from pathlib import Path

import cv2


CAMERA_INDEX = 0
WINDOW_NAME = "Calibration Capture"
PATTERN_SIZE = (9, 6)
IMAGE_DIRECTORY = Path("calibration_images")

REFINEMENT_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)


IMAGE_DIRECTORY.mkdir(exist_ok=True)

saved_image_count = 0

while (
    IMAGE_DIRECTORY / f"calibration_{saved_image_count:02d}.png"
).exists():
    saved_image_count += 1


camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    raise RuntimeError("Could not open the camera.")

try:
    while True:
        frame_received, frame = camera.read()

        if not frame_received:
            print("Could not receive a camera frame.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        pattern_found, corners = cv2.findChessboardCorners(
            gray_frame,
            PATTERN_SIZE,
        )

        display_frame = frame.copy()

        if pattern_found:
            refined_corners = cv2.cornerSubPix(
                gray_frame,
                corners,
                (11, 11),
                (-1, -1),
                REFINEMENT_CRITERIA,
            )

            cv2.drawChessboardCorners(
                display_frame,
                PATTERN_SIZE,
                refined_corners,
                pattern_found,
            )

            status_text = "DETECTED - SPACE TO SAVE"
            status_color = (0, 255, 0)
        else:
            status_text = "PATTERN NOT DETECTED"
            status_color = (0, 0, 255)

        cv2.putText(
            display_frame,
            status_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display_frame,
            f"Saved: {saved_image_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            if not pattern_found:
                print("Image not saved: pattern is not detected.")
                continue

            image_path = (
                IMAGE_DIRECTORY
                / f"calibration_{saved_image_count:02d}.png"
            )

            image_saved = cv2.imwrite(str(image_path), frame)

            if image_saved:
                print(f"Saved {image_path}")
                saved_image_count += 1
            else:
                print(f"Could not save {image_path}")
finally:
    camera.release()
    cv2.destroyAllWindows()