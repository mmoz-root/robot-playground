from pathlib import Path

import cv2
import numpy as np

import json


PATTERN_SIZE = (9, 6)
SQUARE_SIZE_MM = 23.79
IMAGE_DIRECTORY = Path("calibration_images")

REFINEMENT_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)
EXCLUDED_IMAGE_NAMES = {
    "calibration_06.png",
    "calibration_09.png",
    "calibration_10.png",
    "calibration_18.png",
}
PARAMETER_FILE = Path("camera_parameters.json")


object_point_template = np.zeros(
    (PATTERN_SIZE[0] * PATTERN_SIZE[1], 3),
    dtype=np.float32,
)

object_point_template[:, :2] = (
    np.mgrid[
        0:PATTERN_SIZE[0],
        0:PATTERN_SIZE[1],
    ]
    .T
    .reshape(-1, 2)
)

object_point_template *= SQUARE_SIZE_MM


object_points_per_image = []
image_points_per_image = []
used_image_names = []
image_size = None


for image_path in sorted(IMAGE_DIRECTORY.glob("*.png")):
    if image_path.name in EXCLUDED_IMAGE_NAMES:
        print(f"Excluded: {image_path.name}")
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read {image_path.name}")
        continue

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    current_image_size = (
        gray_image.shape[1],
        gray_image.shape[0],
    )

    if image_size is None:
        image_size = current_image_size
    elif current_image_size != image_size:
        raise RuntimeError("Calibration images have different sizes.")

    pattern_found, corners = cv2.findChessboardCorners(
        gray_image,
        PATTERN_SIZE,
    )

    if not pattern_found:
        print(f"Pattern not found: {image_path.name}")
        continue

    refined_corners = cv2.cornerSubPix(
        gray_image,
        corners,
        (11, 11),
        (-1, -1),
        REFINEMENT_CRITERIA,
    )

    object_points_per_image.append(
        object_point_template.copy()
    )
    image_points_per_image.append(refined_corners)
    used_image_names.append(image_path.name)

    print(f"Detected: {image_path.name}")


if not used_image_names:
    raise RuntimeError("No usable calibration images found.")


print()
print(f"Usable images: {len(used_image_names)}")
print(f"Image size: {image_size}")

print(
    "World-point array shape:",
    object_points_per_image[0].shape,
)

print(
    "Image-point array shape:",
    image_points_per_image[0].shape,
)

print("First correspondence:")
print("World:", object_points_per_image[0][0])
print("Pixel:", image_points_per_image[0][0])

rms_error, camera_matrix, distortion_coefficients, \
rotation_vectors, translation_vectors = cv2.calibrateCamera(
    object_points_per_image,
    image_points_per_image,
    image_size,
    None,
    None,
)


np.set_printoptions(
    precision=6,
    suppress=True,
)

print()
print("RMS reprojection error:")
print(rms_error)

print()
print("Camera matrix K:")
print(camera_matrix)

print()
print("Distortion coefficients:")
print(distortion_coefficients.ravel())

print()
print("Estimated board poses:")
print(f"Rotation vectors: {len(rotation_vectors)}")
print(f"Translation vectors: {len(translation_vectors)}")


per_image_errors = []

for (
    image_name,
    world_points,
    detected_points,
    rotation_vector,
    translation_vector,
) in zip(
    used_image_names,
    object_points_per_image,
    image_points_per_image,
    rotation_vectors,
    translation_vectors,
):
    projected_points, _ = cv2.projectPoints(
        world_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion_coefficients,
    )

    detected_points_2d = detected_points.reshape(-1, 2)
    projected_points_2d = projected_points.reshape(-1, 2)

    residuals = (
        detected_points_2d
        - projected_points_2d
    )

    image_rmse = np.sqrt(
        np.mean(
            np.sum(
                residuals ** 2,
                axis=1,
            )
        )
    )

    per_image_errors.append(
        (image_name, image_rmse)
    )


print()
print("Per-image reprojection RMSE, worst first:")

for image_name, image_rmse in sorted(
    per_image_errors,
    key=lambda item: item[1],
    reverse=True,
):
    print(f"{image_name}: {image_rmse:.4f} px")

calibration_data = {
    "image_size": {
        "width": image_size[0],
        "height": image_size[1],
    },
    "pattern_size": {
        "columns": PATTERN_SIZE[0],
        "rows": PATTERN_SIZE[1],
    },
    "square_size_mm": SQUARE_SIZE_MM,
    "rms_reprojection_error_px": float(rms_error),
    "camera_matrix": camera_matrix.tolist(),
    "distortion_coefficients": (
        distortion_coefficients.ravel().tolist()
    ),
    "used_images": used_image_names,
    "excluded_images": sorted(EXCLUDED_IMAGE_NAMES),
    "per_image_reprojection_rmse_px": {
        image_name: float(image_rmse)
        for image_name, image_rmse in per_image_errors
    },
}

PARAMETER_FILE.write_text(
    json.dumps(
        calibration_data,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print()
print(f"Saved calibration to {PARAMETER_FILE}")