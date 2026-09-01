from pathlib import Path

import cv2
import mujoco
import numpy as np

import time


JOINT_KP = np.array([5.0, 5.0])
JOINT_KD = np.array([0.5, 0.5])

CONTROL_MIN = -1.0
CONTROL_MAX = 1.0

SIMULATION_DURATION_SECONDS = 12.0
CAMERA_FPS = 30.0
PRINT_INTERVAL_SECONDS = 0.25

TARGET_BASE_X = 0.10
TARGET_BASE_Y = 0.05
TARGET_BASE_Z = 0.04

TARGET_X_AMPLITUDE = 0.018
TARGET_Y_AMPLITUDE = 0.018

TARGET_X_FREQUENCY_HZ = 0.08
TARGET_Y_FREQUENCY_HZ = 0.11

UNREACHABLE_START_SECONDS = 5.0
UNREACHABLE_END_SECONDS = 7.0

UNREACHABLE_TARGET_X = 0.20
UNREACHABLE_TARGET_Y = 0.05

LINK_1_LENGTH_METRES = 0.10
LINK_2_LENGTH_METRES = 0.05

def forward_kinematics(
    joint_angles,
):
    theta_1, theta_2 = joint_angles

    end_x = (
        LINK_1_LENGTH_METRES
        * np.cos(theta_1)
        + LINK_2_LENGTH_METRES
        * np.cos(theta_1 + theta_2)
    )

    end_y = (
        LINK_1_LENGTH_METRES
        * np.sin(theta_1)
        + LINK_2_LENGTH_METRES
        * np.sin(theta_1 + theta_2)
    )

    return np.array([end_x, end_y])


def inverse_kinematics(
    target_x,
    target_y,
):
    target_distance = np.hypot(
        target_x,
        target_y,
    )

    minimum_reach = abs(
        LINK_1_LENGTH_METRES
        - LINK_2_LENGTH_METRES
    )

    maximum_reach = (
        LINK_1_LENGTH_METRES
        + LINK_2_LENGTH_METRES
    )

    if not (
        minimum_reach
        <= target_distance
        <= maximum_reach
    ):
        return []

    cos_theta_2 = (
        target_x**2
        + target_y**2
        - LINK_1_LENGTH_METRES**2
        - LINK_2_LENGTH_METRES**2
    ) / (
        2.0
        * LINK_1_LENGTH_METRES
        * LINK_2_LENGTH_METRES
    )

    cos_theta_2 = np.clip(
        cos_theta_2,
        -1.0,
        1.0,
    )

    theta_2_magnitude = np.arccos(
        cos_theta_2
    )

    solutions = []

    for theta_2 in [
        theta_2_magnitude,
        -theta_2_magnitude,
    ]:
        theta_1 = (
            np.arctan2(
                target_y,
                target_x,
            )
            - np.arctan2(
                LINK_2_LENGTH_METRES
                * np.sin(theta_2),
                LINK_1_LENGTH_METRES
                + LINK_2_LENGTH_METRES
                * np.cos(theta_2),
            )
        )

        solutions.append(
            np.array([
                theta_1,
                theta_2,
            ])
        )

    return solutions


def wrapped_joint_distance(
    solution,
    current_joints,
):
    difference = (
        solution
        - current_joints
    )

    wrapped_difference = np.arctan2(
        np.sin(difference),
        np.cos(difference),
    )

    return np.linalg.norm(
        wrapped_difference
    )

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "vision_arm.xml"
)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

CAMERA_HEIGHT_METRES = 0.50
TARGET_SURFACE_HEIGHT_METRES = 0.05

LOWER_GREEN = np.array(
    [40, 80, 80],
    dtype=np.uint8,
)

UPPER_GREEN = np.array(
    [85, 255, 255],
    dtype=np.uint8,
)

MINIMUM_TARGET_AREA = 50


model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)

camera_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_CAMERA,
    "overhead_camera",
)

target_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "target",
)

target_mocap_id = (
    model.body_mocapid[target_body_id]
)

vertical_fov_degrees = (
    model.cam_fovy[camera_id]
)

focal_length_pixels = (
    IMAGE_HEIGHT / 2.0
) / np.tan(
    np.deg2rad(vertical_fov_degrees) / 2.0
)

camera_matrix = np.array([
    [
        focal_length_pixels,
        0.0,
        IMAGE_WIDTH / 2.0,
    ],
    [
        0.0,
        focal_length_pixels,
        IMAGE_HEIGHT / 2.0,
    ],
    [
        0.0,
        0.0,
        1.0,
    ],
])

rotation_robot_from_camera = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
])

translation_robot_from_camera = np.array([
    0.0,
    0.0,
    CAMERA_HEIGHT_METRES,
])

transform_robot_from_camera = np.eye(4)

transform_robot_from_camera[:3, :3] = (
    rotation_robot_from_camera
)

transform_robot_from_camera[:3, 3] = (
    translation_robot_from_camera
)

mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(
    model,
    width=IMAGE_WIDTH,
    height=IMAGE_HEIGHT,
)

renderer.update_scene(
    data,
    camera="overhead_camera",
)

rgb_image = renderer.render()

hsv_image = cv2.cvtColor(
    rgb_image,
    cv2.COLOR_RGB2HSV,
)

mask = cv2.inRange(
    hsv_image,
    LOWER_GREEN,
    UPPER_GREEN,
)

target_area = cv2.countNonZero(mask)
detected = target_area >= MINIMUM_TARGET_AREA

if not detected:
    renderer.close()
    raise RuntimeError(
        "Green target was not detected."
    )

moments = cv2.moments(mask)

target_u = (
    moments["m10"]
    / moments["m00"]
)

target_v = (
    moments["m01"]
    / moments["m00"]
)

pixel_homogeneous = np.array([
    target_u,
    target_v,
    1.0,
])

camera_ray = (
    np.linalg.inv(camera_matrix)
    @ pixel_homogeneous
)

target_depth_camera = (
    CAMERA_HEIGHT_METRES
    - TARGET_SURFACE_HEIGHT_METRES
)

position_camera = (
    camera_ray
    * target_depth_camera
    / camera_ray[2]
)

position_camera_homogeneous = np.append(
    position_camera,
    1.0,
)

position_robot_homogeneous = (
    transform_robot_from_camera
    @ position_camera_homogeneous
)

position_robot = (
    position_robot_homogeneous[:3]
)

target_xy = position_robot[:2]

ik_solutions = inverse_kinematics(
    target_xy[0],
    target_xy[1],
)

shoulder_joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "shoulder",
)

elbow_joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "elbow",
)

joint_limits = np.array([
    model.jnt_range[shoulder_joint_id],
    model.jnt_range[elbow_joint_id],
])

feasible_solutions = []

for solution in ik_solutions:
    inside_lower_limits = (
        solution >= joint_limits[:, 0]
    )

    inside_upper_limits = (
        solution <= joint_limits[:, 1]
    )

    if np.all(
        inside_lower_limits
        & inside_upper_limits
    ):
        feasible_solutions.append(
            solution
        )

if not feasible_solutions:
    renderer.close()
    raise RuntimeError(
        "No feasible IK solution."
    )

current_joints = data.qpos[:2].copy()

selected_solution = min(
    feasible_solutions,
    key=lambda solution: (
        wrapped_joint_distance(
            solution,
            current_joints,
        )
    ),
)

verified_xy = forward_kinematics(
    selected_solution
)

ik_verification_error = np.linalg.norm(
    verified_xy
    - target_xy
)

true_target_position = (
    data.xpos[target_body_id].copy()
)

planar_error_metres = np.linalg.norm(
    position_robot[:2]
    - true_target_position[:2]
)

print("Detected:", detected)
print("Target area:", target_area)
print(
    "Target pixel centre:",
    (target_u, target_v),
)

print()
print("IK solutions:")

for index, solution in enumerate(
    feasible_solutions,
    start=1,
):
    print(
        f"  Solution {index}:",
        np.rad2deg(solution),
        "degrees",
    )

print()
print(
    "Current joints:",
    np.rad2deg(current_joints),
    "degrees",
)

print(
    "Selected solution:",
    np.rad2deg(selected_solution),
    "degrees",
)

print(
    "FK verification position:",
    verified_xy,
    "m",
)

print(
    "IK verification error:",
    ik_verification_error * 1000.0,
    "mm",
)

print()
print("Camera ray:")
print(camera_ray)
print()
print("Estimated camera position (m):")
print(position_camera)
print()
print("Trobot<-camera:")
print(transform_robot_from_camera)
print()
print("Estimated robot position (m):")
print(position_robot)
print()
print("Ground-truth target centre (m):")
print(true_target_position)
print()
print(
    "Planar estimation error:",
    planar_error_metres * 1000.0,
    "mm",
)

camera_update_steps = max(
    1,
    round(
        1.0
        / (CAMERA_FPS * model.opt.timestep)
    ),
)

number_of_steps = int(
    SIMULATION_DURATION_SECONDS
    / model.opt.timestep
)

print_interval_steps = max(
    1,
    round(
        PRINT_INTERVAL_SECONDS
        / model.opt.timestep
    ),
)

print()
print(
    "time | status | actual joints (deg) | "
    "joint error (deg) | ctrl | "
    "task error (mm)"
)

tracking_status = "initializing"

for step in range(number_of_steps):
    target_is_unreachable = (
        UNREACHABLE_START_SECONDS
        <= data.time
        < UNREACHABLE_END_SECONDS
    )

    if target_is_unreachable:
        data.mocap_pos[target_mocap_id] = [
            UNREACHABLE_TARGET_X,
            UNREACHABLE_TARGET_Y,
            TARGET_BASE_Z,
        ]
    else:
        data.mocap_pos[target_mocap_id] = [
            TARGET_BASE_X
            + TARGET_X_AMPLITUDE
            * np.sin(
                2.0
                * np.pi
                * TARGET_X_FREQUENCY_HZ
                * data.time
            ),
            TARGET_BASE_Y
            + TARGET_Y_AMPLITUDE
            * np.sin(
                2.0
                * np.pi
                * TARGET_Y_FREQUENCY_HZ
                * data.time
            ),
            TARGET_BASE_Z,
        ]
    current_joints = data.qpos[:2].copy()
    current_velocities = data.qvel[:2].copy()

    joint_error = (
        selected_solution
        - current_joints
    )

    joint_error = np.arctan2(
        np.sin(joint_error),
        np.cos(joint_error),
    )

    raw_control = (
        JOINT_KP * joint_error
        - JOINT_KD * current_velocities
    )

    data.ctrl[:2] = np.clip(
        raw_control,
        CONTROL_MIN,
        CONTROL_MAX,
    )

    if step % print_interval_steps == 0:
        actual_end_effector = (
            forward_kinematics(
                current_joints
            )
        )

        task_error = np.linalg.norm(
            actual_end_effector
            - target_xy
        )

        print(
            f"{data.time:4.2f} | "
            f"{tracking_status:12s} | "
            f"{np.rad2deg(current_joints)} | "
            f"{np.rad2deg(joint_error)} | "
            f"{data.ctrl[:2]} | "
            f"{task_error * 1000.0:8.3f}"
        )

    if step % camera_update_steps == 0:
        renderer.update_scene(
            data,
            camera="overhead_camera",
        )

        rgb_image = renderer.render()

        hsv_image = cv2.cvtColor(
            rgb_image,
            cv2.COLOR_RGB2HSV,
        )

        mask = cv2.inRange(
            hsv_image,
            LOWER_GREEN,
            UPPER_GREEN,
        )

        target_area = cv2.countNonZero(
            mask
        )

        detected = (
            target_area
            >= MINIMUM_TARGET_AREA
        )

        if detected:
            moments = cv2.moments(mask)

            target_u = (
                moments["m10"]
                / moments["m00"]
            )

            target_v = (
                moments["m01"]
                / moments["m00"]
            )

            pixel_homogeneous = np.array([
                target_u,
                target_v,
                1.0,
            ])

            camera_ray = (
                np.linalg.inv(camera_matrix)
                @ pixel_homogeneous
            )

            position_camera = (
                camera_ray
                * target_depth_camera
                / camera_ray[2]
            )

            position_camera_homogeneous = (
                np.append(
                    position_camera,
                    1.0,
                )
            )

            position_robot_homogeneous = (
                transform_robot_from_camera
                @ position_camera_homogeneous
            )

            position_robot = (
                position_robot_homogeneous[:3]
            )

            new_target_xy = (
                position_robot[:2]
            )
            target_xy = (
                new_target_xy.copy()
            )

            new_ik_solutions = (
                inverse_kinematics(
                    new_target_xy[0],
                    new_target_xy[1],
                )
            )

            new_feasible_solutions = []

            for solution in new_ik_solutions:
                inside_limits = np.all(
                    (
                        solution
                        >= joint_limits[:, 0]
                    )
                    & (
                        solution
                        <= joint_limits[:, 1]
                    )
                )

                if inside_limits:
                    new_feasible_solutions.append(
                        solution
                    )

            if new_feasible_solutions:
                target_xy = new_target_xy
                tracking_status = "tracking"

                selected_solution = min(
                    new_feasible_solutions,
                    key=lambda solution: (
                        wrapped_joint_distance(
                            solution,
                            current_joints,
                        )
                    ),
                )
            else:
                tracking_status = "unreachable"
                selected_solution = (
                    current_joints.copy()
                )

        else:
            tracking_status = "not detected"
            selected_solution = (
                current_joints.copy()
            )

        bgr_image = cv2.cvtColor(
            rgb_image,
            cv2.COLOR_RGB2BGR,
        )

        cv2.drawMarker(
            bgr_image,
            (
                int(round(target_u)),
                int(round(target_v)),
            ),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            20,
            2,
        )

        cv2.imshow(
            "Vision-guided arm",
            bgr_image,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        time.sleep(
            camera_update_steps
            * model.opt.timestep
        )

    mujoco.mj_step(model, data)


final_joints = data.qpos[:2].copy()

final_end_effector = forward_kinematics(
    final_joints
)

final_estimated_target_error = np.linalg.norm(
    final_end_effector
    - target_xy
)

mujoco.mj_forward(model, data)

final_true_target_position = (
    data.xpos[target_body_id, :2].copy()
)

final_true_target_error = np.linalg.norm(
    final_end_effector
    - final_true_target_position
)

print()
print(
    "Final joints:",
    np.rad2deg(final_joints),
    "degrees",
)

print(
    "Final end-effector:",
    final_end_effector,
    "m",
)

print(
    "Error to estimated target:",
    final_estimated_target_error
    * 1000.0,
    "mm",
)

print(
    "Error to true target:",
    final_true_target_error
    * 1000.0,
    "mm",
)

cv2.destroyAllWindows()
renderer.close()