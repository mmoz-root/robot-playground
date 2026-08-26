from pathlib import Path

import cv2
import mujoco


MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "pan_tilt.xml"
)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


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

print("Image shape:", rgb_image.shape)
print(
    "Image centre:",
    (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2),
)

bgr_image = cv2.cvtColor(
    rgb_image,
    cv2.COLOR_RGB2BGR,
)

cv2.imshow(
    "Tracking camera",
    bgr_image,
)

cv2.waitKey(0)
cv2.destroyAllWindows()
renderer.close()