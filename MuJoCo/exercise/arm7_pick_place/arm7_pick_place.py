#!/usr/bin/env python3
"""
七轴机械臂 Pick & Place 演示。
加载 arm7_pick_place.xml，执行：移动到物体上方 -> 下降 -> 夹取 -> 抬起 -> 移动到放物区 -> 下降 -> 松开 -> 回 home。
"""
import os
import time
import numpy as np
import mujoco
import mujoco.viewer

# 模型路径（相对本脚本所在目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "arm7_pick_place.xml")

# 动作阶段时长（秒）
DURATION_MOVE = 1.2
DURATION_GRASP = 0.4

# 关节目标：7 个臂关节 + 1 个夹爪（夹爪 0=张开，-0.035=闭合）
# 顺序与 XML 中 actuator 一致：j1..j7, gripper_left
HOME = np.array([0.0, 0.25, 0.5, 0.0, -0.4, 0.0, 0.0, 0.0])
ABOVE_PICK = np.array([0.0, -0.15, 0.75, 0.0, -0.5, 0.0, 0.0, 0.0])
AT_PICK = np.array([0.0, -0.35, 1.0, 0.0, -0.55, 0.0, 0.0, 0.0])
GRASP = np.array([0.0, -0.35, 1.0, 0.0, -0.55, 0.0, 0.0, -0.035])
LIFT = np.array([0.0, -0.15, 0.75, 0.0, -0.5, 0.0, 0.0, -0.035])
# 放物区在 y=-0.35，需要 base 旋转约 -0.7 rad
ABOVE_PLACE = np.array([-0.7, 0.1, 0.5, 0.0, -0.35, 0.0, 0.0, -0.035])
AT_PLACE = np.array([-0.7, 0.05, 0.45, 0.0, -0.3, 0.0, 0.0, -0.035])
RELEASE = np.array([-0.7, 0.05, 0.45, 0.0, -0.3, 0.0, 0.0, 0.0])


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """线性插值，t in [0, 1]."""
    return a + (b - a) * np.clip(t, 0.0, 1.0)


def run_pick_place():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # 初始夹爪张开、臂在 home
    data.ctrl[:] = HOME

    with mujoco.viewer.launch(model, data) as viewer:
        dt = model.opt.timestep
        step = 0

        def set_phase(waypoints, duration, label):
            nonlocal step
            n_steps = int(duration / dt)
            if n_steps <= 0:
                return
            start_ctrl = np.array(data.ctrl.copy())
            for i in range(n_steps):
                t = (i + 1) / n_steps
                data.ctrl[:] = lerp(start_ctrl, waypoints, t)
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(dt * 0.5)  # 适当放慢便于观察
                step += 1

        # 1. 回 home
        set_phase(HOME, 0.8, "home")

        # 2. 移动到物体上方 -> 下降 -> 夹取
        set_phase(ABOVE_PICK, DURATION_MOVE, "above_pick")
        set_phase(AT_PICK, DURATION_MOVE, "at_pick")
        set_phase(GRASP, DURATION_GRASP, "grasp")

        # 3. 抬起 -> 移动到放物区上方 -> 下降 -> 松开
        set_phase(LIFT, DURATION_MOVE, "lift")
        set_phase(ABOVE_PLACE, DURATION_MOVE, "above_place")
        set_phase(AT_PLACE, DURATION_MOVE, "at_place")
        set_phase(RELEASE, DURATION_GRASP, "release")

        # 4. 回 home
        set_phase(HOME, DURATION_MOVE, "home")

        # 保持窗口打开
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)


if __name__ == "__main__":
    run_pick_place()
