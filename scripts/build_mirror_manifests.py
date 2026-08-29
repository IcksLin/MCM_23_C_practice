# -*- coding: utf-8 -*-
"""为三问当前结果镜像生成Config、代码、输入和参考结果哈希清单。"""
from __future__ import annotations
import hashlib, json, platform, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = {
    "q1": ROOT / "q1_test/configs/current_result_mirror/config.yaml",
    "q2": ROOT / "q2_test/configs/current_result_mirror/config.yaml",
    "q3": ROOT / "q3_test/configs/current_result_mirror/config.yaml",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def main() -> int:
    for problem, config_path in CONFIGS.items():
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        code_root = ROOT / f"{problem}_test"
        code_files = sorted((code_root / "algorithms").glob("*.py"))
        code_files += [code_root / "scripts" / f"run_{problem}.py",
                       ROOT / "scripts/run_with_config.py",
                       ROOT / "run_problem.ps1"]
        reference = ROOT / cfg["artifacts"]["reference_dir"]
        manifest = {
            "schema_version": 1,
            "problem": problem,
            "config": {"path": str(config_path.relative_to(ROOT)),
                       "sha256": sha(config_path)},
            "environment": {"python": sys.version, "platform": platform.platform()},
            "inputs": {}, "code": {}, "reference_outputs": {},
            "command": f".\\run_problem.ps1 -ConfigPath {config_path.relative_to(ROOT)}",
            "validation_command": f".\\run_problem.ps1 -ConfigPath {config_path.relative_to(ROOT)} -ValidateOnly",
            "reproduction_level": "feasible_incumbent; timeout MIP may not be byte-identical",
        }
        for rel in (cfg.get("input_hashes") or {}):
            p = ROOT / rel
            manifest["inputs"][rel] = {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in code_files:
            if p.exists():
                rel = str(p.relative_to(ROOT)).replace("\\", "/")
                manifest["code"][rel] = sha(p)
        if reference.exists():
            for p in sorted(reference.glob("*")):
                if p.is_file():
                    rel = str(p.relative_to(ROOT)).replace("\\", "/")
                    manifest["reference_outputs"][rel] = {
                        "sha256": sha(p), "bytes": p.stat().st_size}
        target = config_path.parent / "manifest.json"
        target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(target.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
