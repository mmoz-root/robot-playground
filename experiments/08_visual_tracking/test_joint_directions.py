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

TEST_POSES_DEGREES = [
    ("neutral", 0.0, 0.0),
    ("positive pan", 10.0, 0.0),
    ("negative pan", -10.0, 0.0),
    ("positive tilt", 0.0, 10.0),
    ("negative tilt", 0.0, -10.0),
]


model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)

renderer = mujoco.Renderer(
    model,
    width=IMAGE_WIDTH,
    height=IMAGE_HEIGHT,
)


for name, pan_degrees, tilt_degrees in TEST_POSES_DEGREES:
    data.qpos[:] = np.deg2rad(
        [pan_degrees, tilt_degrees]
    )

    mujoco.mj_forward(model, data)

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

    moments = cv2.moments(mask)

    if moments["m00"] == 0:
        print(f"{name}: target not detected")
        continue

    target_u = int(
        moments["m10"]
        / moments["m00"]
    )

    target_v = int(
        moments["m01"]
        / moments["m00"]
    )

    print(
        f"{name:13s}: "
        f"pan={pan_degrees:5.1f}°, "
        f"tilt={tilt_degrees:5.1f}°, "
        f"target=({target_u}, {target_v})"
    )


renderer.close()