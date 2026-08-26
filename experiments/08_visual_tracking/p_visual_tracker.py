from pathlib import Path
import time

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
IMAGE_CENTRE = (
    IMAGE_WIDTH // 2,
    IMAGE_HEIGHT // 2,
)

CAMERA_FPS = 30.0
SIMULATION_DURATION_SECONDS = 8.0
PRINT_INTERVAL_SECONDS = 0.5

KP_PAN = 0.003
KP_TILT = 0.003

CONTROL_MIN = -1.0
CONTROL_MAX = 1.0

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

renderer = mujoco.Renderer(
    model,
    width=IMAGE_WIDTH,
    height=IMAGE_HEIGHT,
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

next_print_time = 0.0
last_error_u = 0
last_error_v = 0

mujoco.mj_forward(model, data)


for step in range(number_of_steps):
    if step % camera_update_steps == 0:
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
        detected = (
            target_area
            >= MINIMUM_TARGET_AREA
        )

        bgr_image = cv2.cvtColor(
            rgb_image,
            cv2.COLOR_RGB2BGR,
        )

        cv2.drawMarker(
            bgr_image,
            IMAGE_CENTRE,
            (255, 0, 0),
            cv2.MARKER_CROSS,
            20,
            2,
        )

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

            last_error_u = (
                target_u
                - IMAGE_CENTRE[0]
            )

            last_error_v = (
                target_v
                - IMAGE_CENTRE[1]
            )

            raw_pan_control = (
                -KP_PAN * last_error_u
            )

            raw_tilt_control = (
                KP_TILT * last_error_v
            )

            data.ctrl[0] = np.clip(
                raw_pan_control,
                CONTROL_MIN,
                CONTROL_MAX,
            )

            data.ctrl[1] = np.clip(
                raw_tilt_control,
                CONTROL_MIN,
                CONTROL_MAX,
            )

            cv2.drawMarker(
                bgr_image,
                (target_u, target_v),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                20,
                2,
            )

            cv2.line(
                bgr_image,
                IMAGE_CENTRE,
                (target_u, target_v),
                (0, 255, 255),
                2,
            )

        else:
            data.ctrl[:] = 0.0

        if data.time >= next_print_time:
            print(
                f"time={data.time:4.1f}s | "
                f"error=({last_error_u:4d}, "
                f"{last_error_v:4d}) px | "
                f"ctrl=({data.ctrl[0]:6.3f}, "
                f"{data.ctrl[1]:6.3f}) | "
                f"detected={detected}"
            )

            next_print_time += (
                PRINT_INTERVAL_SECONDS
            )

        cv2.imshow(
            "P visual tracking",
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


cv2.destroyAllWindows()
renderer.close()