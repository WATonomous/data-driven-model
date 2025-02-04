# Data Driven Model

## Training the model

- Setup the db with the Drone Simuation Setup 
  - instructions: [https://github.com/TUM-AAS/neural-mpc](https://github.com/TUM-AAS/neural-mpc)
- Setup ml-casadi repository
- Setup acados on Linux locally
- set `export PYTHONPATH="/home/{username}/neural-mpc/ml-casadi:/home/{username}/neural-mpc/ros_dd_mpc"`
- set `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/acados/lib && export ACADOS_SOURCE_DIR=~/acados`

- python src/model_fitting/train.py --csv data/simplified_sim_dataset/train/dataset_001.csv
- python src/experiments/eval.py

## ROS Environment (WIP)

1. Build and start the Docker container:
```bash
./run.sh build
./run.sh up
```

2. Connect Foxglove Studio:
- Open Foxglove Studio
- Click "Open Connection"
- Select "WebSocket"
- Enter URL: `ws://localhost:8765`
- Click "Open"

3. Add visualization panels in Foxglove:
- Click "+" to add a new panel
- Add "3D" panel to see the car moving
- Add "Plot" panel to see velocities
- Select topics:
  - `/car/odom` for car position
  - `/car/reference_path` for the circular path
  - `/car/cmd_vel` for velocity commands

## Stopping the Simulation

```bash
./run.sh down
``` 

If there is an error on build, prune the docker system.

```bash
docker system prune -f
```