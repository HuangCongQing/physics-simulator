import mujoco
import mujoco.viewer

# 加载最简测试模型（小球自由落体）
model_xml = """
<mujoco>
  <worldbody>
    <body name="ball">
      <joint type="free"/>  <!-- 自由关节，允许小球自由运动 -->
      <geom type="sphere" size="0.1" rgba="1 0 0 1"/>  <!-- 红色小球，半径0.1 -->
    </body>
  </worldbody>
</mujoco>
"""
# 加载模型与数据
model = mujoco.MjModel.from_xml_string(model_xml)
data = mujoco.MjData(model)
# 启动可视化窗口（关闭窗口即可退出）
mujoco.viewer.launch(model, data)