from pathlib import Path
import time

import cv2
import mujoco
import numpy as np


MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "pan_tilt_moving_target.xml"
)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_CENTRE = (
    IMAGE_WIDTH // 2,
    IMAGE_HEIGHT // 2,
)

CAMERA_FPS = 30.0
SIMULATION_DURATION_SECONDS = 12.0
PRINT_INTERVAL_SECONDS = 0.5

KP_PAN = 0.003
KP_TILT = 0.003
KD_PAN = 0.0005
KD_TILT = 0.0005

CONTROL_MIN = -1.0
CONTROL_MAX = 1.0

TARGET_BASE_X = 2.0
TARGET_BASE_Y = -0.6
TARGET_BASE_Z = 1.6

HORIZONTAL_AMPLITUDE = 0.6
VERTICAL_AMPLITUDE = 0.3

HORIZONTAL_FREQUENCY_HZ = 0.12
VERTICAL_FREQUENCY_HZ = 0.17

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

target_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "target",
)

target_mocap_id = (
    model.body_mocapid[target_body_id]
)

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

camera_period_seconds = (
    camera_update_steps
    * model.opt.timestep
)

number_of_steps = int(
    SIMULATION_DURATION_SECONDS
    / model.opt.timestep
)

next_print_time = 0.0
last_error_u = 0
last_error_v = 0

previous_error_u = None
previous_error_v = None

error_rate_u = 0.0
error_rate_v = 0.0

mujoco.mj_forward(model, data)


for step in range(number_of_steps):
    data.mocap_pos[target_mocap_id] = [
        TARGET_BASE_X,
        TARGET_BASE_Y
        + HORIZONTAL_AMPLITUDE
        * np.sin(
            2.0
            * np.pi
            * HORIZONTAL_FREQUENCY_HZ
            * data.time
        ),
        TARGET_BASE_Z
        + VERTICAL_AMPLITUDE
        * np.sin(
            2.0
            * np.pi
            * VERTICAL_FREQUENCY_HZ
            * data.time
        ),
    ]
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

            if previous_error_u is None:
                error_rate_u = 0.0
                error_rate_v = 0.0
            else:
                error_rate_u = (
                    last_error_u
                    - previous_error_u
                ) / camera_period_seconds

                error_rate_v = (
                    last_error_v
                    - previous_error_v
                ) / camera_period_seconds

            previous_error_u = last_error_u
            previous_error_v = last_error_v

            pan_p_term = (
                KP_PAN * last_error_u
            )

            pan_d_term = (
                KD_PAN * error_rate_u
            )

            tilt_p_term = (
                KP_TILT * last_error_v
            )

            tilt_d_term = (
                KD_TILT * error_rate_v
            )

            raw_pan_control = -(
                pan_p_term
                + pan_d_term
            )

            raw_tilt_control = (
                tilt_p_term
                + tilt_d_term
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
            previous_error_u = None
            previous_error_v = None

            error_rate_u = 0.0
            error_rate_v = 0.0

        if data.time >= next_print_time:
            print(
                f"time={data.time:4.1f}s | "
                f"error=({last_error_u:4d}, "
                f"{last_error_v:4d}) px | "
                f"rate=({error_rate_u:7.1f}, "
                f"{error_rate_v:7.1f}) px/s | "
                f"ctrl=({data.ctrl[0]:6.3f}, "
                f"{data.ctrl[1]:6.3f}) | "
                f"joints=({np.rad2deg(data.qpos[0]):6.1f}, "
                f"{np.rad2deg(data.qpos[1]):6.1f})° | "
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