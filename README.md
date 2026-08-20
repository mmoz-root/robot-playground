# Robot Playground

Learning experiments for geometric computer vision and robotics.

## Project structure

```text
experiments/
  01_camera_calibration/
    calibration_images/
    capture_calibration_images.py
    calibrate_camera.py
    camera_parameters.json
    undistort_demo.py
  02_aruco_pose/
    generate_aruco_marker.py
    aruco_marker_7.png
    detect_aruco.py
    estimate_aruco_pose.py
notes/
  section_01.pdf
```

The ArUco pose experiment reads the camera parameters produced by the
calibration experiment. Scripts resolve their data paths from their own
locations, so they can be launched from the project root.

## Run an experiment

Activate the existing environment from the project root:

```bash
source .venv/bin/activate
```

Camera calibration examples:

```bash
python experiments/01_camera_calibration/capture_calibration_images.py
python experiments/01_camera_calibration/calibrate_camera.py
python experiments/01_camera_calibration/undistort_demo.py
```

ArUco examples:

```bash
python experiments/02_aruco_pose/generate_aruco_marker.py
python experiments/02_aruco_pose/detect_aruco.py
python experiments/02_aruco_pose/estimate_aruco_pose.py
```
