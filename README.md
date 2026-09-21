# MolmoAct2 Franka Controller

This repository connects the **MolmoAct2-DROID** vision-language-action model to a Franka robot equipped with a Robotiq 2F-85 gripper and two ZED cameras. The model runs behind an HTTP inference server; the robot controller captures observations, requests an action trajectory, validates it, and executes a configurable part of the trajectory in a closed loop.



## System overview

The controller sends the following observation to `POST /act`:

- external/side camera image: RGB `uint8`
- wrist camera image: RGB `uint8`
- natural-language task instruction
- robot state: eight `float32` values (`q1..q7` plus gripper position)

The server returns an `N x 8` action array containing seven absolute Franka joint targets and one normalized gripper command. Before execution, the controller rejects non-finite actions, joint-limit violations, large initial offsets, and large waypoint jumps.

During a run, the Franka control loop runs in a separate process. The main process repeatedly captures observations, performs MolmoAct2 inference, and executes up to `actions_per_chunk` actions at `action_rate_hz`.

## Repository layout

```text
.
├── cameras/                         # Dual ZED camera capture and tests
├── gripper/                         # Robotiq Modbus RTU driver and status tools
├── inference/                       # MolmoAct2 HTTP client and test client
├── robot/
│   ├── main_controller.py           # Main closed-loop controller
│   ├── config.yaml                  # Task, robot, policy, and safety settings
│   ├── robot_init.py                # Unlock Franka joints and activate FCI
│   ├── home_robot.py                # Move to the configured home pose
│   ├── read_jointpose.py            # Print current Franka joint positions
│   └── experiments/                 # Images, videos, configs, and run logs
└── model_server/official_molmoact2/ # Bundled upstream model/server project
```

## Hardware and software requirements

- Linux workstation connected to a Franka robot
- Franka Control Interface (FCI) and a compatible `pylibfranka` installation
- Robotiq 2F-85 available as `/dev/ttyUSB0` at 115200 baud
- two ZED cameras with the serial numbers configured in `cameras/zed_camera.py`
- ZED SDK and its matching Python wheel
- NVIDIA GPU suitable for MolmoAct2-DROID inference
- recent NVIDIA driver compatible with the CUDA 12.8 PyTorch build used by the model server
- Python 3.12 for the robot-side environment
- [`uv`](https://docs.astral.sh/uv/)

The current hardware defaults are:

| Device | Default |
| --- | --- |
| Franka | `192.168.103.1` |
| Robotiq | `/dev/ttyUSB0`, slave ID `0x09` |
| Side ZED | serial `39337350` |
| Wrist ZED | serial `19928076` |
| Molmo endpoint | `http://127.0.0.1:8000/act` |

Change these values for your installation before running the controller.

## Installation

### 1. Robot-side environment

From the repository root:

```bash
uv venv --python 3.12
uv pip install -e .
uv pip install numpy opencv-python pillow pyyaml requests json-numpy pyserial python-dotenv pylibfranka franka-desk
uv pip install cameras/pyzed-5.4-cp312-cp312-linux_x86_64.whl
```

The local ZED wheel is built for CPython 3.12 on x86-64 Linux and requires the corresponding ZED SDK to be installed. Franka library compatibility also depends on the robot system version; follow the Franka documentation if `pylibfranka` cannot connect.

### 2. Model-server environment

The model server has its own project and lockfile:

```bash
cd model_server/official_molmoact2
uv sync
```

Optionally download the model before the first server start:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
uv run hf download allenai/MolmoAct2-DROID
```

The checkpoint is large (approximately 22 GB). Hugging Face authentication may be required depending on the checkpoint's access settings. Set `HF_HOME` before downloading and launching the server if its cache should live on another disk.

## Configuration

Edit `robot/config.yaml` before each run. The main settings are:

- `task.instruction`: instruction sent to MolmoAct2
- `franka.robot_ip`: robot address
- `franka.joint_lower_limit` / `joint_upper_limit`: hard validation bounds
- `franka.max_joint_velocity`, `max_joint_acceleration`, `max_joint_jerk`: motion limits
- `molmo.server_url`: complete `/act` endpoint
- `molmo.max_vla_steps`: maximum inference cycles
- `molmo.actions_per_chunk`: predictions executed before observing again
- `molmo.action_rate_hz`: action publication rate
- `molmo.max_model_waypoint_delta`: maximum allowed change between predictions
- `molmo.max_initial_target_offset`: maximum first-target offset from current joints
- `gripper.enabled`, `speed`, and `force`: Robotiq behavior
- `runtime.dry_run`: suppress sending predicted joint and gripper targets

> [!CAUTION]
> `dry_run: true` is not a hardware-free simulation. The program still connects to the Franka, starts its joint-position control process, activates the Robotiq gripper, and opens both ZED cameras. It only prevents the predicted targets from being sent.

If the model server is on another machine, replace `127.0.0.1` in `molmo.server_url` with that machine's LAN IP, for example `http://192.168.1.20:8000/act`.

## Running the system

Use two terminals. Start the model server first.

### Terminal 1: run MolmoAct2-DROID

```bash
cd model_server/official_molmoact2
uv run python examples/droid/host_server_droid.py \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16
```

This is the command used to run the Molmo model for this project. `0.0.0.0` makes it reachable through all network interfaces; access to port 8000 should therefore be restricted to trusted networks.

Check that the server is ready:

```bash
curl http://127.0.0.1:8000/act
```

For a remote server, use its LAN IP instead of `127.0.0.1`.

### Terminal 2: prepare and run the robot

Create `robot/.env` with the Franka Desk credentials:

```dotenv
DESK_USERNAME=your_username
DESK_PASSWORD=your_password
```

Then initialize the robot:

```bash
uv run python robot/robot_init.py
```

Optionally inspect the pose and move to the repository's predefined home pose:

```bash
uv run python robot/read_jointpose.py
uv run python robot/home_robot.py
```

Finally, start the controller **from the `robot` directory**. This working directory is required because `main_controller.py` currently loads `config.yaml` and writes `experiments/` relative to the current directory.

```bash
cd robot
../.venv/bin/python main_controller.py
```

Review the printed settings and workspace one last time. The controller waits for Enter before beginning policy execution.

## Experiment outputs

Each launch creates the next available directory under `robot/experiments/`, such as `iteration_3/`, containing:

```text
iteration_N/
├── config.yaml                 # Snapshot of the run configuration
├── robot_run.txt               # Observed state and executed action chunks
├── images/
│   ├── step_001_side.png
│   └── step_001_wrist.png
└── videos/
    ├── side_trajectory.mp4
    └── wrist_trajectory.mp4
```

An experiment directory is created even if initialization later fails. With `dry_run: true`, the log records that no actions were executed.

## Useful diagnostics

Run these commands from the repository root unless noted otherwise:

```bash
# Open both configured ZED cameras (GUI required; press q to exit)
uv run python -m cameras.zed_camera

# Poll and decode Robotiq status (Ctrl+C to exit)
uv run python gripper/robotiq_status_test.py

# Print the current Franka joint pose
uv run python robot/read_jointpose.py
```

`inference/test_model_client.py` is a standalone protocol test, but it expects `external.jpg` and `wrist.jpg` in the current directory. Update its `SERVER_URL` when testing a remote server.

## Troubleshooting

- **`config.yaml` not found:** start `main_controller.py` from the `robot/` directory as shown above.
- **Cannot reach `/act`:** confirm the server has completed model loading, check `curl`, firewall rules, and `molmo.server_url`.
- **CUDA out of memory:** keep `--dtype bfloat16`, stop other GPU workloads, and avoid `--cuda-graph`, which uses additional VRAM.
- **ZED camera open failure:** verify the ZED SDK, USB connectivity, permissions, and the two serial numbers in `cameras/zed_camera.py`.
- **Robotiq serial failure:** verify `/dev/ttyUSB0`, user permissions, baud rate, and the Modbus slave ID.
- **Franka connection failure:** verify network reachability, Desk credentials, brakes, FCI activation, and Franka/libfranka compatibility.
- **Trajectory rejected:** review the reported initial offset, waypoint delta, and joint-limit error before changing any safety threshold.

Stop the controller with `Ctrl+C`. It requests a smooth robot stop, closes the cameras and gripper connection, releases video writers, and joins the Franka control process.

