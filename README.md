# MAVSDK-Drone-Codes

Python scripts for commanding an ArduPilot-based quadrotor through [MAVSDK-Python](https://github.com/mavlink/MAVSDK-Python), using OptiTrack motion capture as the position source for indoor flight.

[Demo video](https://youtu.be/kBTWviXE7wE)

## Overview

Indoor flight requires an external position estimate, since GPS is unavailable. Here, an OptiTrack motion capture system streams rigid-body pose into ArduPilot as visual odometry, and these scripts sit on top of that estimate to run autonomous flights: arming, mode switching, takeoff, waypoint following, and landing.

The scripts communicate with the flight controller over MAVLink through MAVSDK's async Python API, so each mission is a short, self-contained script rather than a mission file uploaded to the autopilot.

## Contents

| Path | Description |
| --- | --- |
| `Experiment/` | Flight scripts used in hardware experiments (waypoint missions, offboard setpoints) |
| `test/` | Smaller scripts for checking connection, telemetry, and individual commands before flying |
| `mocap_optitrack.zip` | ROS package configuration used to bridge OptiTrack pose data to MAVLink |

## Requirements

- ArduPilot (ArduCopter) flight controller, configured to accept an external position estimate (`EK3_SRC*` / `VISO_*` parameters)
- Python 3.8+ with `mavsdk`
- MAVSDK server or a MAVLink router forwarding to the companion computer
- OptiTrack system with Motive, plus a ROS bridge publishing rigid-body pose (see `mocap_optitrack.zip`)

```bash
pip install mavsdk
```

## Usage

Start the MAVLink connection to the vehicle, confirm that the position estimate from motion capture is healthy, then run a script:

```bash
python Experiment/<script_name>.py
```

Each script connects, waits for a valid global position and home position, arms, and executes its flight sequence.

## Notes

The motion capture side runs in ROS and is only partially included here as a configuration archive; the full ROS workspace lives outside this repository. Scripts were tested in a motion capture arena and assume the flight volume and coordinate frame of that setup, so waypoint coordinates need to be adjusted for a different space.

Test in simulation (SITL) before flying hardware, and keep a pilot ready to take manual control.
