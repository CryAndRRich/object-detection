#!/usr/bin/env python3
"""Đo box quality của DiffusionDet trên CE-130, TÁCH RIÊNG khỏi chất lượng score head —
EXPERIMENT D so trực tiếp với CE-LocModel A/B/C.

Vì sao KHÔNG dùng AP làm kết luận chính (xem
``docs/thiet-ke-experiment-d-diffusiondet-ce130.md`` §4 và
``count_editing/CE-LocModel/tools/measure_box_quality.py`` — bản gốc của ý tưởng này,
viết cho CE-LocModel, đo trên A/B/C): A/B đã đo được ``score_AUC = 0,512`` (~ngẫu
nhiên, tung đồng xu), tức AP của chúng phản ánh CẢ ranking lẫn box, không tách được đâu
là lỗi hình học đâu là lỗi xếp hạng. So "AP của D" với "AP của A/B/C" là so hai đại
lượng trộn theo tỉ lệ khác nhau — không có nghĩa.

Ba chỉ số, độc lập với ngưỡng score:

    oracle_recall   tỉ lệ GT có ít nhất 1 box IoU>=0,5 (bỏ qua score hoàn toàn) — TRẦN
                    mà một score head hoàn hảo có thể đạt.
    mean_bestIoU    IoU tốt nhất mỗi GT, lấy trung bình — chất lượng hình học thuần.
    score_AUC       score có xếp hạng đúng box khớp lên trên box không khớp không.
                    0,5 = ngẫu nhiên (tham chiếu TUYỆT ĐỐI, không phải tương đối).

Logic đo (không cần detectron2) nằm ở ``objdet/box_quality_metrics.py`` — tách riêng để
test được ở máy không có torch/detectron2, giống convention ``objdet/mmr.py`` +
``tests/test_mmr.py``. File này chỉ là phần I/O: load model, chạy inference, đọc GT.

Chạy (từ ``object-detection/diffusiondet/``, cần detectron2 — chạy trên GPU
server/Kaggle, không chạy local):

    python tools/measure_box_quality_ce130.py \\
        --config-file configs/diffdet.ce130.res50.yaml \\
        MODEL.WEIGHTS output/ce130_agnostic_res50/model_final.pth

    # so trực tiếp với A/B/C: dùng cùng test split, IoU 0,5, và đọc oracle_recall/
    # mean_bestIoU ở CẢ HAI PHÍA — không so AP hai bên.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from objdet.box_quality_metrics import quality_one_image, summarise  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--dataset", default=None,
                     help="tên dataset đã đăng ký để eval; mặc định lấy TEST[0] trong config")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-proposals", type=int, default=None,
                     help="số box lúc eval (dynamic boxes — đổi tự do, KHÔNG cần train "
                          "lại). CE-130 rất dày (test 48,5 vật/ảnh, max 505) nên 300 CHẶN "
                          "TRẦN recall; quét ít nhất tới 3000. Mặc định lấy từ config.")
    ap.add_argument("--out", default=None)
    # KHÔNG dùng nargs=argparse.REMAINDER cho opts: REMAINDER nuốt mọi thứ sau positional
    # đầu tiên, nên "... MODEL.WEIGHTS x.pth --num-proposals 3000" sẽ đẩy cả
    # "--num-proposals 3000" vào opts rồi merge_from_list vỡ (hoặc tệ hơn: chạy sai số
    # box). parse_known_args cho phép cờ đứng ở BẤT KỲ vị trí nào.
    args, opts = ap.parse_known_args()
    # parse_known_args đẩy MỌI cờ không nhận ra vào opts (gõ nhầm "--num-propsals" cũng
    # lọt) -> bắt ngay, đừng để nó chui vào merge_from_list rồi hỏng khó hiểu.
    stray = [o for o in opts if o.startswith("-")]
    if stray:
        ap.error(f"tham số không nhận ra: {stray} (gõ nhầm tên cờ?). "
                 f"Override config phải ở dạng KEY VALUE, vd MODEL.WEIGHTS path.pth")
    if len(opts) % 2 != 0:
        ap.error(f"override config phải theo cặp KEY VALUE, nhận được {len(opts)} phần "
                 f"tử lẻ: {opts}")
    args.opts = opts

    import torch
    from detectron2.config import get_cfg
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import DatasetCatalog, build_detection_test_loader
    from detectron2.modeling import build_model

    from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config
    from diffusiondet.util.model_ema import add_model_ema_configs
    from objdet import register_all

    # Dùng ĐÚNG bộ hàm mở rộng config mà tools/train_net.py dùng (train_net.py:277-281).
    # Thiếu bất kỳ cái nào là merge_from_file vỡ ngay: add_kaggle_configs định nghĩa
    # SOLVER.CHECKPOINT_MAX_TO_KEEP, mà Base-Kaggle-T4x2.yaml (config CE-130 kế thừa nó)
    # có set key đó -> "Non-existent config key". Import từ chính train_net.py thay vì
    # chép lại, để hai bên không bao giờ lệch nhau.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_net import add_kaggle_configs, check_num_classes

    register_all()

    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    add_kaggle_configs(cfg)
    cfg.merge_from_file(args.config_file)
    if args.num_proposals is not None:
        cfg.MODEL.DiffusionDet.NUM_PROPOSALS = args.num_proposals
    cfg.merge_from_list(args.opts)
    # Số class sai không crash mà chỉ cho kết quả rác -> kiểm như train_net.py vẫn làm.
    check_num_classes(cfg)
    cfg.freeze()

    dataset_name = args.dataset or cfg.DATASETS.TEST[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)  # tương đương @torch.no_grad(), đặt trong hàm vì
                                    # import torch cũng nằm trong hàm (xem đầu file: để
                                    # module import được ở máy không có torch/detectron2)

    print("=" * 78)
    print(f"  BOX QUALITY (oracle_recall / mean_bestIoU / score_AUC) — EXPERIMENT D")
    print("-" * 78)
    for k, v in [("config", args.config_file), ("weights", cfg.MODEL.WEIGHTS),
                 ("dataset", dataset_name), ("device", str(device)),
                 ("iou_thr", args.iou_thr)]:
        print(f"  {k:12s} {v}")
    print("=" * 78)

    model = build_model(cfg).to(device)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    # QUAN TRỌNG: DiffusionDetDatasetMapper(is_train=False) XOÁ "annotations" khỏi
    # dataset_dict trước khi trả về (dataset_mapper.py: "if not self.is_train:
    # dataset_dict.pop('annotations', None); return dataset_dict") — mapper eval không
    # mang theo GT. Nếu đọc `inp["instances"]` ở đây sẽ luôn None -> oracle_recall/
    # mean_bestIoU câm lặng thành vô nghĩa (n_gt=0 mọi ảnh). Phải đọc GT trực tiếp từ
    # chính COCO json qua DatasetCatalog, khớp theo "image_id" -> KHÔNG phụ thuộc mapper.
    dataset_dicts = {d["image_id"]: d for d in DatasetCatalog.get(dataset_name)}

    def gt_xyxy_of(image_id):
        if image_id not in dataset_dicts:
            # Không bao giờ nên xảy ra (cùng một DatasetCatalog). Nếu xảy ra thì im lặng
            # coi là "ảnh không có GT" sẽ làm oracle_recall sai mà không ai biết -> dừng.
            raise KeyError(
                f"image_id={image_id!r} có trong loader nhưng không có trong "
                f"DatasetCatalog.get({dataset_name!r}) — dataset đã bị đăng ký hai lần "
                f"với nội dung khác nhau?")
        anns = dataset_dicts[image_id].get("annotations", [])
        boxes = []
        for a in anns:
            if a.get("iscrowd", 0):        # bỏ vùng ignore, giống cách train bỏ qua
                continue
            # Hai giả định dưới đây đã đối chiếu SOURCE detectron2 (bản trong
            # refs/repos/DiffuDETR/detrex/detectron2/, file data/datasets/coco.py),
            # không phải nhớ áng chừng:
            #   dòng 163: record["image_id"] = img_dict["id"]  -> khớp GT bằng image_id
            #             chính là "id" trong mục images của json converter sinh ra;
            #   dòng 210: obj["bbox_mode"] = BoxMode.XYWH_ABS  -> bbox giữ nguyên dạng
            #             XYWH góc trên-trái của json, nên đổi sang xyxy như dưới.
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
        return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)

    mapper = DiffusionDetDatasetMapper(cfg, is_train=False)
    loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    acc = {"best": [], "hit": 0, "n_gt": 0, "auc": [], "scored_hit": 0}
    seen_ids = []                      # để cảnh báo "trần recall" chỉ tính trên ảnh đã chạy
    t0 = time.time()
    n_images = 0
    limit = args.limit or len(loader.dataset)

    for batched_inputs in loader:
        if n_images >= limit:
            break
        outputs = model(batched_inputs)
        for inp, out in zip(batched_inputs, outputs):
            if n_images >= limit:      # đừng vượt --limit ở batch cuối (test loader hiện
                break                   # là batch=1, nhưng đừng dựa vào giả định đó)
            n_images += 1
            inst = out["instances"].to("cpu")
            # model tự postprocess pred_boxes về hệ toạ độ ẢNH GỐC (height/width trong
            # inp), nên so trực tiếp với GT đọc từ COCO json (cũng ở hệ toạ độ ảnh gốc,
            # vì convert_ce130.py ghi bbox theo width/height gốc của ground_truth.jpg,
            # KHÔNG qua resize_and_pad) là nhất quán, không cần quy đổi gì thêm.
            pred_xyxy = inst.pred_boxes.tensor.numpy() if len(inst) else np.zeros((0, 4))
            scores = inst.scores.numpy() if len(inst) else np.zeros((0,))
            gt_xyxy = gt_xyxy_of(inp["image_id"])
            seen_ids.append(inp["image_id"])

            best, hit, ngt, auc, scored_hit = quality_one_image(
                pred_xyxy, scores, gt_xyxy, iou_thr=args.iou_thr)
            acc["best"].append(best)
            acc["hit"] += hit
            acc["n_gt"] += ngt
            acc["scored_hit"] += scored_hit
            if not np.isnan(auc):
                acc["auc"].append(auc)

        if n_images % max(limit // 10, 1) < len(batched_inputs):
            el = time.time() - t0
            print(f"  [{n_images:5d}/{limit}] {1000*el/max(n_images,1):.0f}ms/img "
                  f"ETA {el/max(n_images,1)*(limit-n_images):.0f}s", flush=True)

    res = summarise(acc["best"], acc["hit"], acc["n_gt"], acc["auc"], acc["scored_hit"])
    print("\n" + "=" * 78)
    print(f"RESULTS  n_images={n_images}  n_gt={res['n_gt']}")
    print("-" * 78)
    for k, v in res.items():
        print(f"  {k:20s} {v}")
    print("=" * 78)

    # Trần recall do SỐ BOX, không phải do model: nếu ảnh có nhiều GT hơn số proposal thì
    # dù model hoàn hảo cũng không thể phủ hết. Cảnh báo bằng chính dữ liệu vừa chạy, để
    # không ai đọc nhầm "recall thấp = model kém" khi thật ra là thiếu box.
    # Chỉ đếm trên các ảnh THỰC SỰ đã chạy (tôn trọng --limit), không phải cả dataset.
    n_prop = cfg.MODEL.DiffusionDet.NUM_PROPOSALS
    gt_per_image = res["n_gt"] / max(n_images, 1)
    n_over = sum(1 for iid in seen_ids
                 if len([a for a in dataset_dicts[iid].get("annotations", [])
                         if not a.get("iscrowd", 0)]) > n_prop)
    print(f"  NUM_PROPOSALS = {n_prop} | GT trung bình {gt_per_image:.1f}/ảnh | "
          f"{n_over}/{n_images} ảnh có nhiều GT hơn số box")
    if n_over > 0 or n_prop < 4 * gt_per_image:
        print(f"  ⚠️ SỐ BOX CÓ THỂ ĐANG CHẶN TRẦN RECALL. RESULTS.md §4: số box tối ưu bám "
              f"mật độ vật thể\n"
              f"     (CrowdHuman 22,8 vật/ảnh vẫn tăng tới 3000 box). CE-130 dày hơn — "
              f"chạy lại với --num-proposals 1000/2000/3000 rồi so.")

    if np.isnan(res["score_AUC"]) or res["score_AUC"] < 0.60:
        print("  SCORE HEAD gần ngẫu nhiên -> đọc oracle_recall, đừng đọc AP.")
    print(f"  So với A/B/C: đọc oracle_recall/mean_bestIoU của CẢ HAI PHÍA, "
          f"KHÔNG so AP trực tiếp (xem docstring đầu file).")

    # Tên file kết quả PHẢI mang num_proposals: sweep 300/1000/2000/3000 dùng cùng một
    # checkpoint nên nếu không có, các lần chạy ghi đè lẫn nhau và mất hết trừ lần cuối.
    # Cũng không ghi cạnh MODEL.WEIGHTS khi đó là URL detectron2:// (chưa train xong /
    # dùng weight pretrain) — rơi vào đường dẫn vô nghĩa.
    if args.out:
        out_path = args.out
    else:
        w = cfg.MODEL.WEIGHTS
        stem = (os.path.join(cfg.OUTPUT_DIR, "boxquality")
                if ("://" in w or not w) else os.path.splitext(w)[0])
        out_path = f"{stem}_boxquality_{dataset_name}_N{n_prop}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": res,
                   "settings": {"config": args.config_file, "dataset": dataset_name,
                                "iou_thr": args.iou_thr, "n_images": n_images,
                                "num_proposals": n_prop,
                                "sample_step": cfg.MODEL.DiffusionDet.SAMPLE_STEP,
                                "n_images_over_num_proposals": n_over,
                                "weights": cfg.MODEL.WEIGHTS}}, f, indent=1)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
