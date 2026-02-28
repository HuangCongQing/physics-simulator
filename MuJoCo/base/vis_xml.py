
import os
import time
import numpy as np
import mujoco
import mujoco.viewer

# 模型路径（相对本脚本所在目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hello.xml")
MODEL_PATH = os.path.join(SCRIPT_DIR, "../model/trs_so_arm100/so_arm100.xml")


# 加载模型与数据
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
# 启动可视化窗口（关闭窗口即可退出）
mujoco.viewer.launch(model, data)