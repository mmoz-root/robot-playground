# Robot Playground

Learning experiments that progress from geometric computer vision to robot
kinematics, physics simulation, and control.

## Current checkpoint

Completed:

1. Camera calibration and image undistortion
2. ArUco detection and camera-relative pose estimation
3. Coordinate-frame transforms, composition, and inversion
4. Two-link forward and analytical inverse kinematics
5. MuJoCo model inspection, simulation state, and actuator commands

Next: Section 6 — control theory through simulation, beginning with a
proportional controller for the MuJoCo pendulum.

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
  03_coordinate_frames/
    inspect_coordinate_transforms.py
  04_robot_kinematics/
    forward_kinematics.py
    inverse_kinematics.py
  05_mujoco/
    models/
      pendulum.xml
    inspect_model.py
    read_simulation_state.py
    command_actuator.py
notes/
  section_01.pdf
  section_02.pdf
  section_03.pdf
  section_04.pdf
```

Scripts resolve data and model paths from their own locations, so they can be
launched from the project root. The ArUco experiment uses the camera parameters
produced by the calibration experiment.

## Environment

The current learning environment uses:

```text
Python 3.9.6
OpenCV 5.0.0
Matplotlib 3.9.4
MuJoCo 3.3.7
```

Activate the existing environment from the project root:

```bash
source .venv/bin/activate
```

## Run the experiments

Camera calibration:

```bash
python experiments/01_camera_calibration/capture_calibration_images.py
python experiments/01_camera_calibration/calibrate_camera.py
python experiments/01_camera_calibration/undistort_demo.py
```

ArUco pose estimation:

```bash
python experiments/02_aruco_pose/generate_aruco_marker.py
python experiments/02_aruco_pose/detect_aruco.py
python experiments/02_aruco_pose/estimate_aruco_pose.py
```

Coordinate frames:

```bash
python experiments/03_coordinate_frames/inspect_coordinate_transforms.py
```

Robot kinematics:

```bash
python experiments/04_robot_kinematics/forward_kinematics.py
python experiments/04_robot_kinematics/inverse_kinematics.py
```

MuJoCo basics:

```bash
python -m mujoco.viewer --mjcf=experiments/05_mujoco/models/pendulum.xml
python experiments/05_mujoco/inspect_model.py
python experiments/05_mujoco/read_simulation_state.py
python experiments/05_mujoco/command_actuator.py
```

The MuJoCo pendulum demonstrates the chain:

```text
control signal
-> actuator torque
-> physics step
-> joint velocity
-> joint position
-> sensor and body pose data
```
