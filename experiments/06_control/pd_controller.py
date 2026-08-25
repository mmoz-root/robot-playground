from pathlib import Path

import mujoco
import numpy as np


MODEL_PATH = (
    Path(__file__).parent.parent
    / "05_mujoco"
    / "models"
    / "pendulum.xml"
)

TARGET_ANGLE_DEGREES = 30.0
KP = 8.0
KD = 2.0

CONTROL_MIN = -1.0
CONTROL_MAX = 1.0

SIMULATION_DURATION_SECONDS = 5.0
PRINT_INTERVAL_SECONDS = 0.25


model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)

target_angle_radians = np.deg2rad(
    TARGET_ANGLE_DEGREES
)

number_of_steps = int(
    SIMULATION_DURATION_SECONDS
    / model.opt.timestep
)

print_interval_steps = int(
    PRINT_INTERVAL_SECONDS
    / model.opt.timestep
)


print(f"Target angle: {TARGET_ANGLE_DEGREES}°")
print(f"Kp: {KP}")
print(f"Kd: {KD}")
print()
print(
    "time | angle | velocity | error | "
    "P term | D term | control"
)


for step in range(number_of_steps + 1):
    current_angle = data.qpos[0]
    current_velocity = data.qvel[0]

    error = (
        target_angle_radians
        - current_angle
    )

    p_term = KP * error
    d_term = -KD * current_velocity

    raw_control = (
        p_term
        + d_term
    )

    control = np.clip(
        raw_control,
        CONTROL_MIN,
        CONTROL_MAX,
    )

    data.ctrl[0] = control

    if step % print_interval_steps == 0:
        print(
            f"{data.time:4.2f} | "
            f"{np.rad2deg(current_angle):6.2f} | "
            f"{current_velocity:8.3f} | "
            f"{np.rad2deg(error):6.2f} | "
            f"{p_term:6.3f} | "
            f"{d_term:6.3f} | "
            f"{control:7.3f}"
        )

    if step < number_of_steps:
        mujoco.mj_step(model, data)