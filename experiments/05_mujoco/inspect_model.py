from pathlib import Path

import mujoco


MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "pendulum.xml"
)

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)


def print_element_names(
    label,
    element_count,
    element_type,
):
    print(f"{label}:")

    for element_id in range(element_count):
        element_name = mujoco.mj_id2name(
            model,
            element_type,
            element_id,
        )

        print(
            f"  ID {element_id}: "
            f"{element_name}"
        )

    print()


print(f"Model path: {MODEL_PATH}")
print()

print("Model contents:")
print(f"  Bodies: {model.nbody}")
print(f"  Joints: {model.njnt}")
print(f"  Actuators: {model.nu}")
print(f"  Sensors: {model.nsensor}")
print()

print("State dimensions:")
print(f"  qpos size: {model.nq}")
print(f"  qvel size: {model.nv}")
print(f"  ctrl size: {model.nu}")
print()

print("Initial data:")
print(f"  time: {data.time}")
print(f"  qpos: {data.qpos}")
print(f"  qvel: {data.qvel}")
print(f"  ctrl: {data.ctrl}")
print(f"  sensor data: {data.sensordata}")

print_element_names(
    "Bodies",
    model.nbody,
    mujoco.mjtObj.mjOBJ_BODY,
)

print_element_names(
    "Joints",
    model.njnt,
    mujoco.mjtObj.mjOBJ_JOINT,
)

print_element_names(
    "Actuators",
    model.nu,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
)

print_element_names(
    "Sensors",
    model.nsensor,
    mujoco.mjtObj.mjOBJ_SENSOR,
)