import json
from pathlib import Path

import cv2
import numpy as np

np.set_printoptions(
    precision=4,
    suppress=True,
)

CAMERA_INDEX = 0
WINDOW_NAME = "Coordinate Transform Inspector"

DICTIONARY_TYPE = cv2.aruco.DICT_4X4_50
TARGET_MARKER_ID = 7

MARKER_LENGTH_MM = 60.0
AXIS_LENGTH_MM = 30.0

CALIBRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "01_camera_calibration"
    / "camera_parameters.json"
)


with CALIBRATION_PATH.open("r", encoding="utf-8") as file:
    calibration = json.load(file)

camera_matrix = np.asarray(
    calibration["camera_matrix"],
    dtype=np.float64,
)

distortion_coefficients = np.asarray(
    calibration["distortion_coefficients"],
    dtype=np.float64,
)

expected_frame_size = (
    calibration["image_size"]["width"],
    calibration["image_size"]["height"],
)


half_length = MARKER_LENGTH_MM / 2.0

marker_object_points = np.array(
    [
        [-half_length, +half_length, 0.0],  # top-left
        [+half_length, +half_length, 0.0],  # top-right
        [+half_length, -half_length, 0.0],  # bottom-right
        [-half_length, -half_length, 0.0],  # bottom-left
    ],
    dtype=np.float32,
)


dictionary = cv2.aruco.getPredefinedDictionary(
    DICTIONARY_TYPE
)

detector_parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    dictionary,
    detector_parameters,
)


camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    raise RuntimeError("Could not open the camera.")

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    expected_frame_size[0],
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    expected_frame_size[1],
)

frame_size_checked = False

try:
    while True:
        frame_received, frame = camera.read()

        if not frame_received:
            print("Could not receive a camera frame.")
            break

        if not frame_size_checked:
            frame_height, frame_width = frame.shape[:2]
            actual_frame_size = (frame_width, frame_height)

            print(f"Expected frame size: {expected_frame_size}")
            print(f"Actual frame size:   {actual_frame_size}")

            if actual_frame_size != expected_frame_size:
                raise RuntimeError(
                    "Camera resolution does not match calibration."
                )

            frame_size_checked = True

        marker_corners, marker_ids, rejected_candidates = (
            detector.detectMarkers(frame)
        )

        display_frame = frame.copy()
        target_pose_found = False
        latest_transforms = None

        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(
                display_frame,
                marker_corners,
                marker_ids,
            )

            for detected_corners, marker_id in zip(
                marker_corners,
                marker_ids.flatten(),
            ):
                if int(marker_id) != TARGET_MARKER_ID:
                    continue

                image_points = (
                    detected_corners
                    .reshape(4, 2)
                    .astype(np.float32)
                )

                pose_found, rvec, tvec = cv2.solvePnP(
                    marker_object_points,
                    image_points,
                    camera_matrix,
                    distortion_coefficients,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )

                if not pose_found:
                    continue

                target_pose_found = True
                rotation_matrix, _ = cv2.Rodrigues(rvec)

                camera_from_marker = np.eye(
                    4,
                    dtype=np.float64,
                )

                camera_from_marker[:3, :3] = rotation_matrix
                camera_from_marker[:3, 3] = tvec.reshape(3)


                marker_from_camera = np.eye(
                    4,
                    dtype=np.float64,
                )

                marker_from_camera[:3, :3] = rotation_matrix.T

                marker_from_camera[:3, 3] = (
                    -rotation_matrix.T @ tvec.reshape(3)
                )


                identity_check = (
                    marker_from_camera @ camera_from_marker
                )

                latest_transforms = (
                    rvec.copy(),
                    tvec.copy(),
                    rotation_matrix.copy(),
                    camera_from_marker.copy(),
                    marker_from_camera.copy(),
                    identity_check.copy(),
)

                cv2.drawFrameAxes(
                    display_frame,
                    camera_matrix,
                    distortion_coefficients,
                    rvec,
                    tvec,
                    AXIS_LENGTH_MM,
                    3,
                )

                tx, ty, tz = tvec.reshape(3)
                total_distance = float(np.linalg.norm(tvec))

                cv2.putText(
                    display_frame,
                    (
                        f"tvec mm: "
                        f"x={tx:.1f}, y={ty:.1f}, z={tz:.1f}"
                    ),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    display_frame,
                    f"Distance: {total_distance:.1f} mm",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        if not target_pose_found:
            cv2.putText(
                display_frame,
                f"Marker {TARGET_MARKER_ID} not detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("p") and latest_transforms is not None:
            (
                current_rvec,
                current_tvec,
                current_rotation,
                current_camera_from_marker,
                current_marker_from_camera,
                current_identity_check,
            ) = latest_transforms

            print()
            print("rvec:")
            print(current_rvec.reshape(3))

            print()
            print("tvec:")
            print(current_tvec.reshape(3))

            print()
            print("Rotation matrix R:")
            print(current_rotation)

            print()
            print("Tcamera<-marker:")
            print(current_camera_from_marker)

            print()
            print("Tmarker<-camera:")
            print(current_marker_from_camera)

            print()
            print("Inverse composition check:")
            print(current_identity_check)

        if key == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
