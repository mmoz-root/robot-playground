from pathlib import Path

import mujoco
import numpy as np


MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "pendulum.xml"
)

SIMULATION_DURATION_SECONDS = 1.0
PRINT_INTERVAL_SECONDS = 0.1


model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)

pendulum_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "pendulum",
)

# Start horizontally at 90 degrees.
data.qpos[0] = np.deg2rad(90.0)

# No motor command.
data.ctrl[0] = 0.0

# Update derived quantities and sensors without
# advancing simulation time.
mujoco.mj_forward(model, data)

print("Initial body pose:")
print(f"  position: {data.xpos[pendulum_body_id]}")
print(
    "  rotation matrix:\n"
    f"{data.xmat[pendulum_body_id].reshape(3, 3)}"
)
print()

number_of_steps = int(
    SIMULATION_DURATION_SECONDS
    / model.opt.timestep
)

print_interval_steps = int(
    PRINT_INTERVAL_SECONDS
    / model.opt.timestep
)


print(
    "time (s) | angle (deg) | "
    "velocity (rad/s) | sensor position"
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
            f"{data.sensordata[0]:15.3f}"
        )

    if step < number_of_steps:
        mujoco.mj_step(model, data)


mujoco.mj_forward(model, data)

print()
print("Final body pose:")
print(f"  position: {data.xpos[pendulum_body_id]}")
print(
    "  rotation matrix:\n"
    f"{data.xmat[pendulum_body_id].reshape(3, 3)}"
)