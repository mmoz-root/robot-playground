import json
from pathlib import Path

import cv2
import numpy as np


CAMERA_INDEX = 0
EXPERIMENT_DIRECTORY = Path(__file__).resolve().parent
PARAMETER_FILE = EXPERIMENT_DIRECTORY / "camera_parameters.json"
WINDOW_NAME = "Undistortion Demo"


calibration_data = json.loads(
    PARAMETER_FILE.read_text(encoding="utf-8")
)

camera_matrix = np.array(
    calibration_data["camera_matrix"],
    dtype=np.float64,
)

distortion_coefficients = np.array(
    calibration_data["distortion_coefficients"],
    dtype=np.float64,
)

expected_width = calibration_data["image_size"]["width"]
expected_height = calibration_data["image_size"]["height"]


camera = cv2.VideoCapture(CAMERA_INDEX)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    expected_width,
)
camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    expected_height,
)

if not camera.isOpened():
    raise RuntimeError("Could not open the camera.")


show_undistorted = False


try:
    while True:
        frame_received, frame = camera.read()

        if not frame_received:
            print("Could not receive a camera frame.")
            break

        frame_height, frame_width = frame.shape[:2]

        if (
            frame_width != expected_width
            or frame_height != expected_height
        ):
            raise RuntimeError(
                "Camera resolution does not match calibration: "
                f"received {frame_width}x{frame_height}, "
                f"expected {expected_width}x{expected_height}."
            )

        undistorted_frame = cv2.undistort(
            frame,
            camera_matrix,
            distortion_coefficients,
        )

        if show_undistorted:
            display_frame = undistorted_frame.copy()
            label = "UNDISTORTED"
            label_color = (0, 255, 0)
        else:
            display_frame = frame.copy()
            label = "ORIGINAL"
            label_color = (0, 255, 255)

        cv2.putText(
            display_frame,
            label,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            label_color,
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            display_frame,
            "Press U to toggle",
            (30, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            WINDOW_NAME,
            display_frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("u"):
            show_undistorted = not show_undistorted
finally:
    camera.release()
    cv2.destroyAllWindows()
