import mujoco
import mujoco.viewer

# 加载最简测试模型（小球自由落体）
# model_xml = """
# <mujoco>
#   <worldbody>
#     <body name="ball">
#       <joint type="free"/>  <!-- 自由关节，允许小球自由运动 -->
#       <geom type="sphere" size="0.1" rgba="1 0 0 1"/>  <!-- 红色小球，半径0.1 -->
#     </body>
#   </worldbody>
# </mujoco>
# """
model_xml = """
<mujoco>
  <worldbody>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""

# 加载模型与数据
model = mujoco.MjModel.from_xml_string(model_xml)
print(model.geom('green_sphere'))
data = mujoco.MjData(model)
# 启动可视化窗口（关闭窗口即可退出）
mujoco.viewer.launch(model, data)