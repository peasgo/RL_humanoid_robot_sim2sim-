# Finding_USD.py
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app   # 启动 Omniverse/Kit

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
print("ISAACLAB_NUCLEUS_DIR =", ISAACLAB_NUCLEUS_DIR)

# 可选：演示访问/拷贝
# import omni.client, os
# src = f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
# dst = os.path.expanduser("~/assets/franka/panda_instanceable.usd")
# os.makedirs(os.path.dirname(dst), exist_ok=True)
# print("copy:", omni.client.copy(src, dst))

app.close()
