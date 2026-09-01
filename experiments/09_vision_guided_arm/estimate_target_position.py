from pathlib import Path

import cv2
import mujoco
import numpy as np


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
    (0, 0, 255),
    cv2.MARKER_CROSS,
    20,
    2,
)

cv2.imshow(
    "Target position estimate",
    bgr_image,
)

cv2.imshow(
    "Green target mask",
    mask,
)

cv2.waitKey(0)
cv2.destroyAllWindows()
renderer.close()