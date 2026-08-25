from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


MODEL_PATH = (
    Path(__file__).parent.parent
    / "05_mujoco"
    / "models"
    / "pendulum.xml"
)

TARGET_ANGLE_DEGREES = 30.0
SIMULATION_DURATION_SECONDS = 8.0

CONTROL_MIN = -1.0
CONTROL_MAX = 1.0


model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

target_angle_radians = np.deg2rad(
    TARGET_ANGLE_DEGREES
)


def simulate_controller(kp, ki, kd):
    data = mujoco.MjData(model)

    integral_error = 0.0

    number_of_steps = int(
        SIMULATION_DURATION_SECONDS
        / model.opt.timestep
    )

    times = []
    angles_degrees = []
    controls = []

    for step in range(number_of_steps + 1):
        current_angle = data.qpos[0]
        current_velocity = data.qvel[0]

        error = (
            target_angle_radians
            - current_angle
        )

        if ki > 0.0:
            integral_error += (
                error
                * model.opt.timestep
            )

            integral_limit = (
                CONTROL_MAX / ki
            )

            integral_error = np.clip(
                integral_error,
                -integral_limit,
                integral_limit,
            )

        p_term = kp * error
        i_term = ki * integral_error
        d_term = -kd * current_velocity

        control = np.clip(
            p_term + i_term + d_term,
            CONTROL_MIN,
            CONTROL_MAX,
        )

        data.ctrl[0] = control

        times.append(data.time)
        angles_degrees.append(
            np.rad2deg(current_angle)
        )
        controls.append(control)

        if step < number_of_steps:
            mujoco.mj_step(model, data)

    return (
        np.array(times),
        np.array(angles_degrees),
        np.array(controls),
    )

controllers = {
    "P": (8.0, 0.0, 0.0),
    "PD": (8.0, 0.0, 2.0),
    "PID": (8.0, 4.0, 2.0),
}

results = {}

for name, gains in controllers.items():
    results[name] = simulate_controller(
        *gains
    )

    final_angle = results[name][1][-1]

    print(
        f"{name} final angle: "
        f"{final_angle:.3f}°"
    )


figure, (
    angle_axes,
    control_axes,
) = plt.subplots(
    2,
    1,
    figsize=(9, 7),
    sharex=True,
)

for name, (
    times,
    angles,
    controls,
) in results.items():
    angle_axes.plot(
        times,
        angles,
        label=name,
    )

    control_axes.plot(
        times,
        controls,
        label=name,
    )

angle_axes.axhline(
    TARGET_ANGLE_DEGREES,
    color="black",
    linestyle="--",
    label="Target",
)

angle_axes.set_ylabel("Angle (degrees)")
angle_axes.set_title(
    "P vs PD vs PID — Pendulum Angle"
)
angle_axes.grid(True)
angle_axes.legend()

control_axes.axhline(
    CONTROL_MAX,
    color="gray",
    linestyle="--",
)

control_axes.axhline(
    CONTROL_MIN,
    color="gray",
    linestyle="--",
)

control_axes.set_xlabel("Time (seconds)")
control_axes.set_ylabel("Control command")
control_axes.set_title("Controller Output")
control_axes.grid(True)
control_axes.legend()

figure.tight_layout()
plt.show()