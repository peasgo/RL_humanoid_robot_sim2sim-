#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_franka_local_assets_v2.py

两种模式：
A) 默认：生成一个本地“壳”USD (panda_instanceable_local.usd) 引用远程官方 USD，再把 franka.py 指向它。
B) --mirror-core：下载一批 Franka 核心 USD（按 manifest），并把 franka.py 指向本地副本。

用法：
  # 方案 A（默认，推荐）
  python tools/setup_franka_local_assets_v2.py

  # 方案 B（下载核心 USD）
  python tools/setup_franka_local_assets_v2.py --mirror-core

  # 还原 franka.py
  python tools/setup_franka_local_assets_v2.py --revert
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from urllib.request import urlretrieve, URLError

# —— 你环境里打印出来的根（保持与 Finding_USD.py 打印一致）——
ASSET_BASE_HTTP = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab"
REMOTE_FRANKA_DIR = f"{ASSET_BASE_HTTP}/Robots/FrankaEmika"

# 仓库路径
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_STORE = REPO_ROOT / "z_used_models_storage" / "FrankaEmika"
LOCAL_STORE.mkdir(parents=True, exist_ok=True)

# 目标：修改这个文件里的 usd_path
FRANKA_PY = REPO_ROOT / "source" / "isaaclab_assets" / "isaaclab_assets" / "robots" / "franka.py"

FRANKA_PY_BAK = FRANKA_PY.with_suffix(".py.bak")

# Manifest（方案 B 用）：按经验整理的一批常用 USD（若后续缺啥再补）
CORE_FILES = [
    "panda_instanceable.usd",
    "panda.usd",
    "franka.usd",
    "franka_hand.usd",
    "franka_hand_instanceable.usd",
    # 可能的手爪/碰撞/关节定义的引用（不同版本命名有所差异，先尽量覆盖）
    "panda_arm.usd",
    "panda_arm_instanceable.usd",
]

def info(msg): print("[INFO]", msg)
def warn(msg): print("[WARN]", msg)
def err(msg):  print("[ERR ]", msg)

def download_file(remote_url: str, local_path: Path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        info(f"Downloading: {remote_url}")
        urlretrieve(remote_url, str(local_path))
    except URLError as e:
        warn(f"Download failed: {remote_url} -> {e}")

def make_local_redirect_usd(local_usd: Path, remote_target_url: str):
    """
    生成一个很小的 USD 文件，本地路径固定，内部只引用（reference）远程官方 USD。
    """
    text = f'''#usda 1.0
(
    upAxis = "Z"
)

def "FrankaRedirect"
{{
    # 通过 subLayer / references 都可以，这里用 references
    prepend references = @<{remote_target_url}>@
}}
'''
    local_usd.write_text(text, encoding="utf-8")
    info(f"Created local redirect USD: {local_usd}")

def patch_franka_py(new_usd_path: Path):
    if not FRANKA_PY.exists():
        err(f"franka.py not found: {FRANKA_PY}")
        sys.exit(1)

    src = FRANKA_PY.read_text(encoding="utf-8")
    if not FRANKA_PY_BAK.exists():
        shutil.copy2(FRANKA_PY, FRANKA_PY_BAK)
        info(f"Backed up to {FRANKA_PY_BAK}")

    # 将 usd_path=... 替换为本地绝对路径
    pattern = re.compile(
        r'(usd_path\s*=\s*)f?["\'].*panda_instanceable\.usd["\']'
    )
    new_line = rf'\1"{str(new_usd_path.resolve())}"'
    new_src, n = pattern.subn(new_line, src)
    if n == 0:
        warn("No usd_path pattern matched. Please check franka.py manually.")
    else:
        FRANKA_PY.write_text(new_src, encoding="utf-8")
        info(f"Patched franka.py -> usd_path = {new_usd_path.resolve()}")

def revert_franka_py():
    if FRANKA_PY_BAK.exists():
        shutil.copy2(FRANKA_PY_BAK, FRANKA_PY)
        info("Reverted franka.py from backup.")
    else:
        warn("Backup franka.py.bak not found; nothing to revert.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror-core", dest="mirror_core", action="store_true",
                    help="下载一批核心 USD 到本地，并指向本地主文件")
    ap.add_argument("--revert", action="store_true", help="还原 franka.py")
    args = ap.parse_args()

    if args.revert:
        revert_franka_py()
        return

    if args.mirror_core:
        # 方案 B：下载清单里的文件
        for name in CORE_FILES:
            download_file(f"{REMOTE_FRANKA_DIR}/{name}", LOCAL_STORE / name)

        target = LOCAL_STORE / "panda_instanceable.usd"
        if not target.exists():
            err("核心文件 panda_instanceable.usd 下载失败，无法继续。请检查网络或更换版本号。")
            sys.exit(1)

        patch_franka_py(target)
        print("\n✅ Done (mirror-core). 现在可直接运行训练以验证本地 USD 是否可用。\n")
        return

    # 默认方案 A：生成本地重定向 USD
    local_redirect = LOCAL_STORE / "panda_instanceable_local.usd"
    remote_main = f"{REMOTE_FRANKA_DIR}/panda_instanceable.usd"
    make_local_redirect_usd(local_redirect, remote_main)
    patch_franka_py(local_redirect)
    print("\n✅ Done (redirect). 现在已通过本地路径引用远程官方 USD。\n"
          "如果将来需要完全离线，请运行：\n"
          "  python tools/setup_franka_local_assets_v2.py --mirror-core\n")

if __name__ == "__main__":
    main()
