# -*- coding: utf-8 -*-
"""统一Config启动器：校验独立镜像输入后调用对应问题主程序。"""
from __future__ import annotations
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
RUNNERS = {
    "q1": ROOT / "q1_test" / "scripts" / "run_q1.py",
    "q2": ROOT / "q2_test" / "scripts" / "run_q2.py",
    "q3": ROOT / "q3_test" / "scripts" / "run_q3.py",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args()
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    problem = str(cfg.get("problem", "")).lower()
    if problem not in RUNNERS:
        raise SystemExit("Config必须声明 problem: q1/q2/q3")
    if int(cfg.get("schema_version", 0)) != 1:
        raise SystemExit("不支持的Config schema_version")
    for rel, expected in (cfg.get("input_hashes") or {}).items():
        path = Path(rel)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise SystemExit(f"镜像输入缺失: {path}")
        actual = sha256(path)
        if actual != str(expected).upper():
            raise SystemExit(f"镜像输入哈希不匹配: {path}\n{actual}\n{expected}")
        print(f"[OK] {path.relative_to(ROOT)} {actual[:16]}...", flush=True)
    if args.validate_only:
        print(f"Config验证通过: problem={problem}, path={config_path}")
        return 0
    cmd = [sys.executable, "-u", str(RUNNERS[problem]),
           "--config", str(config_path)]
    print("启动:", subprocess.list2cmdline(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
