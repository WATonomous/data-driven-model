# Car MPC Control

A ROS package implementing Model Predictive Control (MPC) for autonomous vehicle trajectory tracking. This is a simplified version adapted from a data-driven MPC approach, using a basic bicycle model for vehicle dynamics.

## Features

- Model Predictive Control using CasADi optimizer
- Bicycle model dynamics
- Circular trajectory generation
- RViz visualization
- ROS integration

## Prerequisites

- ROS (tested on Melodic/Noetic)
- Python 3
- CasADi
- NumPy
- SciPy

## Installation

1. Create a catkin workspace (if you don't have one):
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin init
```

2. Clone this repository into the src directory:
```bash
cd ~/catkin_ws/src
git clone <repository_url> car_mpc_control
```

3. Install Python dependencies:
```bash
cd car_mpc_control
pip3 install -r requirements.txt
```

4. Build the package:
```bash
cd ~/catkin_ws
catkin build
source devel/setup.bash
```

## Usage

1. Launch the MPC controller and trajectory generator:
```bash
roslaunch car_mpc_control car_mpc.launch
```

This will start:
- The MPC controller node
- The trajectory generator node (generating a circular reference path)
- RViz for visualization

## Node Information

### MPC Controller (`car_mpc.py`)
- Subscribes to:
  - `/odom` (nav_msgs/Odometry): Current vehicle state
  - `/reference` (std_msgs/Float64MultiArray): Reference trajectory points
- Publishes to:
  - `/cmd_vel` (geometry_msgs/Twist): Control commands

### Trajectory Generator (`trajectory_generator.py`)
- Publishes to:
  - `/reference` (std_msgs/Float64MultiArray): Reference trajectory points
  - `/reference_path` (nav_msgs/Path): Visualization of reference path

## Parameters

### MPC Parameters (in `car_mpc.py`)
- Time step (dt): 0.1s
- Prediction horizon (N): 10 steps
- Maximum velocity: 20.0 m/s
- Maximum steering angle: π/4 rad
- Maximum acceleration: 3.0 m/s²

### Trajectory Parameters (in `trajectory_generator.py`)
- Circle radius: 10.0m
- Reference velocity: 5.0 m/s
- Update rate: 10 Hz

## Customization

To modify the reference trajectory:
1. Edit the `TrajectoryGenerator` class in `trajectory_generator.py`
2. Implement your own trajectory generation method
3. Update the parameters as needed

To tune the MPC controller:
1. Adjust the weights in the `CarMPC` class (`self.Q` and `self.R`)
2. Modify the constraints (`v_max`, `delta_max`, `a_max`)
3. Change the prediction horizon or time step if needed 