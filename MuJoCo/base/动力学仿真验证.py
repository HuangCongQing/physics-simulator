import mujoco
import mujoco.viewer
import time
import os

# 最小可用的内置模型（无 humanoid.xml 时使用）
_FALLBACK_XML = """
<mujoco model="fallback">
  <worldbody>
    <light pos="0 0 2"/>
    <geom type="sphere" size="0.1" pos="0 0 0.5"/>
    <body pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

def _load_model():
    pkg_dir = os.path.dirname(mujoco.__file__)
    candidates = [
        os.path.join(pkg_dir, "model", "humanoid", "humanoid.xml"),
        os.path.join(pkg_dir, "humanoid.xml"),
        os.path.join(os.getcwd(), "humanoid.xml"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return mujoco.MjModel.from_xml_path(p)
    # 未找到文件时使用内置简单模型
    return mujoco.MjModel.from_xml_string(_FALLBACK_XML)

# model = _load_model()
model_path = "../model/humanoid/humanoid.xml"
print(model_path)
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 启动可视化，模拟关节运动
with mujoco.viewer.launch(model, data) as viewer:
    # 控制机械臂关节运动
    for _ in range(1000):
        time.sleep(0.01)
        mujoco.mj_step(model, data)
        viewer.sync()