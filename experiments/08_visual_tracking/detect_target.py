from pathlib import Path

import cv2
import mujoco
import numpy as np


MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "pan_tilt.xml"
)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

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

mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(
    model,
    width=IMAGE_WIDTH,
    height=IMAGE_HEIGHT,
)

renderer.update_scene(
    data,
    camera="tracking_camera",
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

image_centre = (
    IMAGE_WIDTH // 2,
    IMAGE_HEIGHT // 2,
)

bgr_image = cv2.cvtColor(
    rgb_image,
    cv2.COLOR_RGB2BGR,
)

cv2.drawMarker(
    bgr_image,
    image_centre,
    (255, 0, 0),
    markerType=cv2.MARKER_CROSS,
    markerSize=20,
    thickness=2,
)

print("Detected:", detected)
print("Target area:", target_area)

if detected:
    moments = cv2.moments(mask)

    target_u = int(
        moments["m10"]
        / moments["m00"]
    )

    target_v = int(
        moments["m01"]
        / moments["m00"]
    )

    error_u = target_u - image_centre[0]
    error_v = target_v - image_centre[1]

    print(
        "Target centre:",
        (target_u, target_v),
    )

    print(
        "Pixel error:",
        (error_u, error_v),
    )

    cv2.drawMarker(
        bgr_image,
        (target_u, target_v),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=20,
        thickness=2,
    )

cv2.imshow(
    "Target detection",
    bgr_image,
)

cv2.imshow(
    "Green mask",
    mask,
)

cv2.waitKey(0)
cv2.destroyAllWindows()
renderer.close()