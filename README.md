# Data Driven Model

## Quick Start

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