import cv2


CAMERA_INDEX = 0
EXPECTED_FRAME_SIZE = (1920, 1080)
WINDOW_NAME = "ArUco Detection"
DICTIONARY_TYPE = cv2.aruco.DICT_4X4_50


dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_TYPE)
detector_parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(
    dictionary,
    detector_parameters,
)

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    raise RuntimeError("Could not open the camera.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, EXPECTED_FRAME_SIZE[0])
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, EXPECTED_FRAME_SIZE[1])

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

            print(f"Actual frame size: {actual_frame_size}")

            if actual_frame_size != EXPECTED_FRAME_SIZE:
                raise RuntimeError(
                    "Camera resolution does not match calibration. "
                    f"Expected {EXPECTED_FRAME_SIZE}, "
                    f"received {actual_frame_size}."
                )

            frame_size_checked = True

        marker_corners, marker_ids, rejected_candidates = (
            detector.detectMarkers(frame)
        )

        display_frame = frame.copy()

        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(
                display_frame,
                marker_corners,
                marker_ids,
            )

            status_text = f"Detected markers: {len(marker_ids)}"
            status_color = (0, 255, 0)
        else:
            status_text = "No marker detected"
            status_color = (0, 0, 255)

        cv2.putText(
            display_frame,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            status_color,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()