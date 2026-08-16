"""Minimal python-file config system.

A config is a plain ``.py`` file of module-level assignments. Reading it is just
an ``exec``; the value here is the small amount of glue around that: ``_base_``
inheritance, dotted-key overrides from the command line, and a reproducible dump.

    cfg = Config.fromfile("config/cfg_odvg_diffusion.py")
    cfg.merge_from_dict({"diff.sampling_timesteps": 5})
    cfg.dump(output_dir / "config_cfg.py")

Written for this project rather than depending on mmcv/addict, so a config file
stays readable python with no magic.
"""

import argparse
import copy
import os
import pprint
from pathlib import Path
from typing import Any, Dict

BASE_KEY = "_base_"


def _load_py(path: str) -> Dict[str, Any]:
    """Exec a python file and return its public module-level names."""
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config file not found: {path}")

    namespace: Dict[str, Any] = {"__file__": path, "__name__": "_config_"}
    with open(path, "r", encoding="utf-8") as f:
        code = compile(f.read(), path, "exec")
    exec(code, namespace)  # noqa: S102 - configs are trusted project files

    import types

    return {
        k: v
        for k, v in namespace.items()
        if not k.startswith("__") and not isinstance(v, types.ModuleType) and not callable(v)
    }


class Config:
    """Attribute-and-dict access over a flat config namespace."""

    def __init__(self, cfg_dict: Dict[str, Any] = None, filename: str = None):
        super().__setattr__("_cfg", dict(cfg_dict or {}))
        super().__setattr__("_filename", filename)

    # ---------------------------------------------------------------- #
    @classmethod
    def fromfile(cls, filename: str) -> "Config":
        """Load a config, resolving ``_base_`` first so the child wins.

        ``_base_`` may be a single path or a list of paths, relative to the file
        that declares it. This is what lets ``cfg_odvg_diffusion.py`` be a short
        delta on top of the non-diffusion baseline instead of a full copy that
        silently drifts out of sync.
        """
        cfg_dict = _load_py(filename)
        bases = cfg_dict.pop(BASE_KEY, None)

        if bases is None:
            return cls(cfg_dict, filename=filename)

        if isinstance(bases, str):
            bases = [bases]
        merged: Dict[str, Any] = {}
        parent_dir = Path(filename).resolve().parent
        for base in bases:
            base_path = base if os.path.isabs(base) else str(parent_dir / base)
            merged.update(cls.fromfile(base_path).to_dict())
        merged.update(cfg_dict)
        return cls(merged, filename=filename)

    # ---------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._cfg)

    def merge_from_dict(self, options: Dict[str, Any]):
        """Apply ``{"a.b": value}`` overrides in place.

        Dotted keys descend into nested dicts; a plain key sets a top-level
        field. Unknown keys are allowed -- configs are open, and rejecting them
        would make it impossible to add a field from the command line.
        """
        for key, value in (options or {}).items():
            parts = key.split(".")
            target = self._cfg
            for part in parts[:-1]:
                if part not in target or not isinstance(target[part], dict):
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value

    def dump(self, path) -> str:
        """Write the resolved config as executable python. Returns the text."""
        lines = [
            "# Auto-generated dump of the resolved config -- do not edit.",
            f"# source: {self._filename}",
            "",
        ]
        for key in sorted(self._cfg):
            lines.append(f"{key} = {pprint.pformat(self._cfg[key], width=100, sort_dicts=True)}")
        text = "\n".join(lines) + "\n"

        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return text

    # ---------------------------------------------------------------- #
    def __getattr__(self, name):
        try:
            return self._cfg[name]
        except KeyError as exc:
            raise AttributeError(f"config has no field {name!r}") from exc

    def __setattr__(self, name, value):
        self._cfg[name] = value

    def __contains__(self, name):
        return name in self._cfg

    def __repr__(self):
        return f"Config(filename={self._filename!r}, {len(self._cfg)} fields)"

    def items(self):
        return self._cfg.items()

    def get(self, name, default=None):
        return self._cfg.get(name, default)


class DictAction(argparse.Action):
    """argparse action for ``--options key=value key2=value2``.

    Values are parsed as python literals when possible (so ``5``, ``0.1``,
    ``True``, ``[1,2]`` arrive with the right type), and kept as strings
    otherwise.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for item in values:
            if "=" not in item:
                raise argparse.ArgumentTypeError(f"expected key=value, got {item!r}")
            key, value = item.split("=", 1)
            options[key.strip()] = self._parse(value.strip())
        setattr(namespace, self.dest, options)

    @staticmethod
    def _parse(value: str):
        import ast

        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
