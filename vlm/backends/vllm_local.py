"""vLLM offline (`LLM.chat`) trên MỘT GPU server (A30 dùng chung; venv riêng có vllm — xem vlm/README.md). Theo docs vLLM (2026-11-03):
- vLLM báo lỗi khi tổng × `gpu_memory_utilization` > bộ nhớ ĐANG trống ⇒ đọc bộ nhớ trống (nvidia-smi, GPU mà `run_on_free_gpu.py` đặt
  trong CUDA_VISIBLE_DEVICES) rồi đặt utilization = `vllm.gpu_mem_gib` / tổng; trống không đủ thì dừng NGAY với thông báo rõ;
- `kv_cache_memory_bytes` cố định KV cache (bỏ qua utilization); `quantization: fp8_per_tensor` = FP8 chỉ weight (Marlin) trên Ampere;
- `engine.tensor_parallel_size` 2 (Kaggle 2 × T4, `config/kaggle/`): `vllm.gpu_mem_gib` / `kv_cache_gib` tính MỖI card, kiểm card trống ít nhất;
- `LLM.chat` nhận ảnh base64 data URL (chỉ base64); structured outputs JSON theo schema tác giả. Tiền = 0; `sec` = thời gian khối /
  số prompt."""

import os
import subprocess
import time

from vlm.backends import Fatal, result
from vlm.prompt import box_schema

__all__ = ["VLLMLocal", "gpu_free_total_mib", "memory_utilization"]


def gpu_free_total_mib():
    """-> (trống, tổng) MiB, lấy GPU trống ÍT nhất trong CUDA_VISIBLE_DEVICES (hay GPU 0) — tensor parallel cần đủ chỗ trên MỌI card."""
    ids = [i.strip() for i in (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",") if i.strip()]
    out = subprocess.run(["nvidia-smi", "-i", ",".join(ids), "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()
    return min((tuple(float(v) for v in line.split(",")) for line in out), key=lambda ft: ft[0])


def memory_utilization(gib, free_mib, total_mib, margin_mib=256):
    """GiB muốn dùng -> `gpu_memory_utilization`; RuntimeError nếu GPU trống < GiB + lề (vLLM sẽ lỗi lúc khởi động)."""
    need = gib * 1024.0
    if free_mib < need + margin_mib:
        raise RuntimeError(f"GPU trống {free_mib / 1024:.1f} GiB < cần {gib:.1f} GiB (+{margin_mib} MiB lề) — chờ GPU trống hơn, giảm "
                           f"`vllm.gpu_mem_gib` / `engine.max_model_len`, hay hạ bản model")
    return need / total_mib


class VLLMLocal:
    def __init__(self, cfg, log=print):
        from vllm import LLM
        self.cfg, self.log = cfg, log
        v = cfg.get("vllm") or {}
        free, total = gpu_free_total_mib()
        util = memory_utilization(float(v["gpu_mem_gib"]), free, total)
        engine = dict(cfg.get("engine") or {})
        if v.get("kv_cache_gib"):
            engine["kv_cache_memory_bytes"] = int(float(v["kv_cache_gib"]) * 2 ** 30)
        log(f"[vllm] {cfg['model']} | GPU trống {free / 1024:.1f} / {total / 1024:.1f} GiB -> gpu_memory_utilization {util:.3f} | "
            f"engine {engine}")
        t0 = time.time()
        self._so = self._structured()
        self.llm = LLM(model=cfg["model"], gpu_memory_utilization=util, seed=int(cfg.get("seed", 0)), **engine)
        free2, _ = gpu_free_total_mib()
        log(f"[vllm] nạp model {time.time() - t0:.0f} s | GPU bị chiếm thêm {(free - free2) / 1024:.1f} GiB (mọi tiến trình trên GPU)")

    def _structured(self):
        """Ràng buộc JSON theo schema tác giả lúc sinh (structured outputs của vLLM; tên lớp đổi theo bản)."""
        schema = box_schema(self.cfg["k"])["schema"]
        try:
            from vllm.sampling_params import StructuredOutputsParams
            return {"structured_outputs": StructuredOutputsParams(json=schema)}
        except ImportError:
            pass
        try:
            from vllm.sampling_params import GuidedDecodingParams
            return {"guided_decoding": GuidedDecodingParams(json=schema)}
        except ImportError:
            raise Fatal("bản vLLM này không có StructuredOutputsParams / GuidedDecodingParams -> không ràng buộc được schema tác giả, "
                        "cài bản vLLM có structured outputs")

    def params(self, job):
        from vllm import SamplingParams
        s = {**(self.cfg.get("sampling") or {}), **self._so}
        if job.get("temperature") is not None:
            s["temperature"] = job["temperature"]
        return SamplingParams(max_tokens=int(self.cfg["max_tokens"]), **s)

    @staticmethod
    def conversation(job):
        conv = []
        for t in job["messages"]:
            if t["image"]:
                content = [{"type": "text", "text": t["text"]},
                           {"type": "image_url", "image_url": {"url": f"data:{t.get('mime') or 'image/jpeg'};base64,{t['image']}"}}]
            else:
                content = t["text"]
            conv.append({"role": t["role"], "content": content})
        return conv

    def batch(self, jobs):
        t0 = time.time()
        outs = self.llm.chat([self.conversation(j) for j in jobs], sampling_params=[self.params(j) for j in jobs], use_tqdm=False,
                             chat_template_kwargs=self.cfg.get("chat_template_kwargs"))
        el = time.time() - t0
        sec = el / max(len(jobs), 1)                       # thông lượng khối; latency 1 ảnh chỉ đo được khi chạy từng ảnh (`chunk: 1`)
        return [result(o.outputs[0].text, len(o.prompt_token_ids or []), len(o.outputs[0].token_ids), 0, 0.0, sec, None,
                       el if len(jobs) == 1 else None, finish=o.outputs[0].finish_reason) for o in outs]
