from pathlib import Path

import cv2


DICTIONARY_TYPE = cv2.aruco.DICT_4X4_50
MARKER_ID = 7
MARKER_PIXELS = 1000
BORDER_BITS = 1

OUTPUT_PATH = Path(__file__).with_name("aruco_marker_7.png")


dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_TYPE)

marker_image = cv2.aruco.generateImageMarker(
    dictionary,
    MARKER_ID,
    MARKER_PIXELS,
    borderBits=BORDER_BITS,
)

saved = cv2.imwrite(str(OUTPUT_PATH), marker_image)

if not saved:
    raise RuntimeError("OpenCV could not save the marker image.")

print(f"Marker ID: {MARKER_ID}")
print(f"Image shape: {marker_image.shape}")
print(f"Saved to: {OUTPUT_PATH.resolve()}")