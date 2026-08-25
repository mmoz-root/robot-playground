from pathlib import Path

import mujoco
import numpy as np


MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "pendulum.xml"
)

CONTROL_COMMAND = -1.0
SIMULATION_DURATION_SECONDS = 1.0
PRINT_INTERVAL_SECONDS = 0.1


model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)

data.ctrl[0] = CONTROL_COMMAND


number_of_steps = int(
    SIMULATION_DURATION_SECONDS
    / model.opt.timestep
)

print_interval_steps = int(
    PRINT_INTERVAL_SECONDS
    / model.opt.timestep
)


print(f"Control command: {CONTROL_COMMAND}")
print(
    "time (s) | angle (deg) | "
    "velocity (rad/s) | actuator torque"
)

for step in range(number_of_steps + 1):
    if step % print_interval_steps == 0:
        mujoco.mj_forward(model, data)

        angle_degrees = np.rad2deg(
            data.qpos[0]
        )

        print(
            f"{data.time:8.3f} | "
            f"{angle_degrees:11.3f} | "
            f"{data.qvel[0]:16.3f} | "
            f"{data.qfrc_actuator[0]:15.3f}"
        )

    if step < number_of_steps:
        mujoco.mj_step(model, data)