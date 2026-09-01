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


model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)

camera_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_CAMERA,
    "overhead_camera",
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

bgr_image = cv2.cvtColor(
    rgb_image,
    cv2.COLOR_RGB2BGR,
)

image_centre = (
    IMAGE_WIDTH // 2,
    IMAGE_HEIGHT // 2,
)

cv2.drawMarker(
    bgr_image,
    image_centre,
    (255, 255, 255),
    cv2.MARKER_CROSS,
    20,
    2,
)

print("Image shape:", rgb_image.shape)
print("Vertical FOV:", vertical_fov_degrees)
print("Focal length:", focal_length_pixels)
print("Camera matrix K:")
print(camera_matrix)

cv2.imshow(
    "Overhead arm camera",
    bgr_image,
)

cv2.waitKey(0)
cv2.destroyAllWindows()
renderer.close()