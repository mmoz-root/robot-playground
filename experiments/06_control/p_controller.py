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
KP = 30.0

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
print()
print(
    "time (s) | angle (deg) | "
    "error (deg) | control"
)


for step in range(number_of_steps + 1):
    current_angle = data.qpos[0]

    error = (
        target_angle_radians
        - current_angle
    )

    raw_control = KP * error

    control = np.clip(
        raw_control,
        CONTROL_MIN,
        CONTROL_MAX,
    )

    data.ctrl[0] = control

    if step % print_interval_steps == 0:
        print(
            f"{data.time:8.3f} | "
            f"{np.rad2deg(current_angle):11.3f} | "
            f"{np.rad2deg(error):11.3f} | "
            f"{control:7.3f}"
        )

    if step < number_of_steps:
        mujoco.mj_step(model, data)