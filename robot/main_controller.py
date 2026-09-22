#!/usr/bin/env python3

import os
import time
import yaml
import shutil
import ctypes
import traceback
import multiprocessing as mp
from queue import Empty

import cv2
import numpy as np
from PIL import Image

from pylibfranka import Robot, ControllerMode, JointPositions, RealtimeConfig
from cameras.zed_camera import DualZEDCameras
from gripper.robotiq_gripper import RobotiqGripper
from inference.molmo_client import MolmoActClient


# ============================================================
# CONFIG
# ============================================================

CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)

EXPERIMENT_NAME = CONFIG["experiment"]["name"]
EXPERIMENT_ROOT = CONFIG["experiment"]["root_dir"]
INSTRUCTION = CONFIG["task"]["instruction"]

ROBOT_IP = CONFIG["franka"]["robot_ip"]
JOINT_LOWER_LIMIT = np.asarray(CONFIG["franka"]["joint_lower_limit"], dtype=np.float64)
JOINT_UPPER_LIMIT = np.asarray(CONFIG["franka"]["joint_upper_limit"], dtype=np.float64)
JOINT_VELOCITY_LIMIT = np.asarray(CONFIG["franka"]["joint_velocity_limit"], dtype=np.float64)

MOLMO_SERVER = CONFIG["molmo"]["server_url"]
MAX_VLA_STEPS = int(CONFIG["molmo"]["max_vla_steps"])
ACTIONS_PER_CHUNK = int(CONFIG["molmo"]["actions_per_chunk"])
ACTION_RATE_HZ = float(CONFIG["molmo"]["action_rate_hz"])
ACTION_DT = 1.0 / ACTION_RATE_HZ

ENABLE_GRIPPER = bool(CONFIG["gripper"]["enabled"])
GRIPPER_SPEED = int(CONFIG["gripper"]["speed"])
GRIPPER_FORCE = int(CONFIG["gripper"]["force"])

DRY_RUN = bool(CONFIG["runtime"]["dry_run"])
MAX_INITIAL_TARGET_OFFSET = float(
    CONFIG["molmo"]["max_initial_target_offset"]
)


# ============================================================
# EXPERIMENT DIRECTORY
# ============================================================

def create_experiment_dir():
    os.makedirs(EXPERIMENT_ROOT, exist_ok=True)

    iteration = 1
    while True:
        experiment_dir = os.path.join(EXPERIMENT_ROOT, f"iteration_{iteration}")
        if not os.path.exists(experiment_dir):
            break
        iteration += 1

    os.makedirs(os.path.join(experiment_dir, "images"))
    os.makedirs(os.path.join(experiment_dir, "videos"))
    shutil.copy2(CONFIG_PATH, os.path.join(experiment_dir, "config.yaml"))

    return experiment_dir


# ============================================================
# MODEL OUTPUT ERROR
# ============================================================

class ModelTrajectoryRejected(RuntimeError):
    pass


# ============================================================
# SHARED MEMORY
# ============================================================

def write_shared_vector(shared_array, version_counter, values):
    """Publish a 7-D vector using a simple version counter."""
    values = np.asarray(values, dtype=np.float64)

    version_counter.value += 1  # odd = write in progress
    for joint_index in range(7):
        shared_array[joint_index] = float(values[joint_index])
    version_counter.value += 1  # even = complete


def read_shared_vector(shared_array, version_counter):
    """Read a consistent shared-memory snapshot."""
    while True:
        version_before = version_counter.value
        if version_before % 2:
            continue

        values = np.array([shared_array[i] for i in range(7)], dtype=np.float64)
        version_after = version_counter.value

        if version_before == version_after and version_after % 2 == 0:
            return values


def read_shared_vector_if_updated(shared_array, version_counter, previous_version, previous_values):
    """Return a new vector only if a complete newer value has been published."""
    current_version = version_counter.value

    if current_version == previous_version or current_version % 2:
        return previous_values, previous_version, False

    values = np.array([shared_array[i] for i in range(7)], dtype=np.float64)
    version_after = version_counter.value

    if current_version == version_after and version_after % 2 == 0:
        return values, version_after, True

    return previous_values, previous_version, False


# ============================================================
# CONTROL PERIOD
# ============================================================

def control_period_seconds(control_period):
    """Convert pylibfranka control period to seconds."""
    try:
        return float(control_period.to_sec())
    except Exception:
        return 0.001  # Franka nominal control period


# ============================================================
# FRANKA PROCESS
# ============================================================

def franka_control_process(
    shared_target, target_version,
    shared_state, state_version,
    state_ready, stop_event, error_queue,
):
    robot = None

    try:
        print(f"Connecting Franka to {ROBOT_IP}...", flush=True)

        robot = Robot(ROBOT_IP, RealtimeConfig.kIgnore)
        control = robot.start_joint_position_control(ControllerMode.JointImpedance)

        robot_state, _ = control.readOnce()
        measured_q = np.asarray(robot_state.q, dtype=np.float64)

        # Start command exactly from the measured configuration.
        command_q = measured_q.copy()
        start_q = command_q.copy()
        target_q = command_q.copy()

        target_elapsed = ACTION_DT
        last_target_version = target_version.value

        write_shared_vector(shared_state, state_version, measured_q)
        control.writeOnce(JointPositions(command_q.tolist()))

        state_ready.set()
        print("Franka control process ready.", flush=True)

        while not stop_event.is_set():
            robot_state, control_period = control.readOnce()
            dt = control_period_seconds(control_period)

            measured_q = np.asarray(robot_state.q, dtype=np.float64)
            write_shared_vector(shared_state, state_version, measured_q)

            new_target, last_target_version, target_updated = read_shared_vector_if_updated(
                shared_target,
                target_version,
                last_target_version,
                target_q,
            )

            if target_updated:
                # Interpolate from the currently commanded pose to the new Molmo waypoint.
                start_q = command_q.copy()
                target_q = new_target.copy()
                target_elapsed = 0.0

                if np.any(target_q < JOINT_LOWER_LIMIT) or np.any(target_q > JOINT_UPPER_LIMIT):
                    raise RuntimeError("Target exceeds Franka joint limits.")


            # Molmo gives waypoints at 15 Hz; interpolate between them at the Franka control rate.
            if target_elapsed < ACTION_DT:
                target_elapsed += dt
                interpolation = min(target_elapsed / ACTION_DT, 1.0)
                command_q = start_q + interpolation * (target_q - start_q)
            else:
                command_q = target_q.copy()

            if np.any(command_q < JOINT_LOWER_LIMIT) or np.any(command_q > JOINT_UPPER_LIMIT):
                raise RuntimeError("Generated command exceeds Franka joint limits.")

            control.writeOnce(JointPositions(command_q.tolist()))

        # Mark the final position command as finished.
        final_command = JointPositions(command_q.tolist())
        final_command.motion_finished = True
        control.writeOnce(final_command)

    except Exception as exception:
        error_text = (
            f"Franka control error:\n"
            f"{repr(exception)}\n"
            f"{traceback.format_exc()}"
        )

        try:
            error_queue.put_nowait(error_text)
        except Exception:
            pass

        stop_event.set()

    finally:
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass


# ============================================================
# MAIN VLA CONTROLLER
# ============================================================

class VLAController:
    def __init__(
        self,
        shared_target, target_version,
        shared_state, state_version,
        stop_event, error_queue,
        experiment_dir,
    ):
        self.shared_target = shared_target
        self.target_version = target_version
        self.shared_state = shared_state
        self.state_version = state_version
        self.stop_event = stop_event
        self.error_queue = error_queue

        self.experiment_dir = experiment_dir
        self.image_dir = os.path.join(experiment_dir, "images")
        self.video_dir = os.path.join(experiment_dir, "videos")
        self.log_path = os.path.join(experiment_dir, "robot_run.txt")

        print(f"Experiment directory: {self.experiment_dir}")

        print("Connecting to Robotiq gripper...")
        self.gripper = RobotiqGripper()
        self.gripper.activate()
        print("Robotiq gripper ready.")

        print("Opening ZED cameras...")
        self.cameras = DualZEDCameras()
        print("Cameras ready.")

        self.video_fps = 5.0
        self.side_video_writer = None
        self.wrist_video_writer = None

        self.molmo = MolmoActClient(server_url=MOLMO_SERVER)
        print("MolmoAct2 client ready.")


    # ========================================================
    # VIDEO
    # ========================================================

    def _write_rgb_to_video(self, camera_name, rgb_frame):
        if rgb_frame is None:
            return

        writer_attr = f"{camera_name}_video_writer"
        writer = getattr(self, writer_attr, None)

        # Create the writer on the first frame so the real image size is known.
        if writer is None:
            height, width = rgb_frame.shape[:2]
            output_path = os.path.join(self.video_dir, f"{camera_name}_trajectory.mp4")

            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.video_fps,
                (width, height),
                True,
            )

            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {output_path}")

            setattr(self, writer_attr, writer)

        bgr_frame = cv2.cvtColor(
            np.asarray(rgb_frame, dtype=np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        writer.write(bgr_frame)


    # ========================================================
    # LOGGING
    # ========================================================

    def write_log(self, text):
        with open(self.log_path, "a", encoding="utf-8") as log_file:
            log_file.write(text + "\n")


    # ========================================================
    # ROBOT STATE
    # ========================================================

    def get_joint_state(self):
        return read_shared_vector(
            self.shared_state,
            self.state_version,
        ).astype(np.float32)


    def get_model_state(self):
        joint_pos = self.get_joint_state()
        gripper_pos = self.gripper.get_raw_position()

        # Molmo DROID state = 7 Franka joints + 1 gripper value.
        return np.concatenate([
            joint_pos,
            np.array([gripper_pos], dtype=np.float32),
        ])


    # ========================================================
    # FRANKA HEALTH
    # ========================================================

    def check_franka_error(self):
        if not self.stop_event.is_set():
            return

        try:
            error_text = self.error_queue.get_nowait()
        except Empty:
            error_text = "Franka control process stopped unexpectedly."

        raise RuntimeError(error_text)


    # ========================================================
    # OBSERVATION
    # ========================================================

    def get_observation(self):
        side_rgb, wrist_rgb = self.cameras.get_frames_rgb()

        self._write_rgb_to_video("side", side_rgb)
        self._write_rgb_to_video("wrist", wrist_rgb)

        return side_rgb, wrist_rgb, self.get_model_state()


    # ========================================================
    # TARGET
    # ========================================================

    def set_joint_target(self, target_joint_pos):
        target_joint_pos = np.asarray(target_joint_pos, dtype=np.float64)

        if target_joint_pos.shape != (7,):
            raise ValueError(f"Expected joint target shape (7,), got {target_joint_pos.shape}")

        if not np.all(np.isfinite(target_joint_pos)):
            raise ValueError("Joint target contains NaN or Inf.")

        if np.any(target_joint_pos < JOINT_LOWER_LIMIT) or np.any(target_joint_pos > JOINT_UPPER_LIMIT):
            raise ModelTrajectoryRejected("Molmo target exceeds Franka joint limits.")

        write_shared_vector(
            self.shared_target,
            self.target_version,
            target_joint_pos,
        )


    # ========================================================
    # MODEL OUTPUT VALIDATION
    # ========================================================

    def validate_action_trajectory(self, predicted_actions):
        predicted_actions = np.asarray(predicted_actions, dtype=np.float32)

        if predicted_actions.ndim != 2 or predicted_actions.shape[1] != 8:
            raise ModelTrajectoryRejected(
                f"Expected Molmo action shape (N, 8), got {predicted_actions.shape}"
            )

        if not np.all(np.isfinite(predicted_actions)):
            raise ModelTrajectoryRejected("Molmo returned NaN or Inf.")

        predicted_q = predicted_actions[:, :7].astype(np.float64)
        current_q = self.get_joint_state().astype(np.float64)

        # Hard robot-position-limit check.
        if np.any(predicted_q < JOINT_LOWER_LIMIT) or np.any(predicted_q > JOINT_UPPER_LIMIT):
            raise ModelTrajectoryRejected(
                "Molmo trajectory contains a target outside Franka joint limits."
            )

        # First predicted absolute pose should not be excessively far from
        # the robot's actual current configuration.
        initial_delta = predicted_q[0] - current_q
        max_initial_offset = float(np.max(np.abs(initial_delta)))

        # Diagnostic: implied velocities between Molmo's 15 Hz waypoints.
        if len(predicted_q) > 1:
            waypoint_delta = np.diff(predicted_q, axis=0)
            implied_velocity = np.abs(waypoint_delta) / ACTION_DT
            max_velocity_ratio = np.max(implied_velocity / JOINT_VELOCITY_LIMIT)
        else:
            waypoint_delta = np.empty((0, 7))
            max_velocity_ratio = 0.0

        print(f"Current q:          {np.array2string(current_q, precision=4)}")
        print(f"Molmo first q:      {np.array2string(predicted_q[0], precision=4)}")
        print(f"Initial delta:      {np.array2string(initial_delta, precision=4)}")
        print(f"Max initial offset: {max_initial_offset:.4f} rad")

        if len(waypoint_delta):
            print(f"Max waypoint delta: {np.max(np.abs(waypoint_delta)):.4f} rad")
            print(f"Max velocity ratio: {max_velocity_ratio:.2f}x robot limit")

        if max_initial_offset > MAX_INITIAL_TARGET_OFFSET:
            raise ModelTrajectoryRejected(
                f"Initial Molmo target offset {max_initial_offset:.4f} rad exceeds "
                f"{MAX_INITIAL_TARGET_OFFSET:.4f} rad."
            )

        return predicted_actions


    # ========================================================
    # GRIPPER
    # ========================================================

    def execute_gripper_action(self, gripper_value):
        if not ENABLE_GRIPPER:
            return

        normalized_value = float(np.clip(gripper_value, 0.0, 1.0))
        robotiq_position = int(round(normalized_value * 255.0))

        print(f"Gripper: model={gripper_value:.4f} -> Robotiq={robotiq_position}")

        self.gripper.set_position(
            robotiq_position,
            speed=GRIPPER_SPEED,
            force=GRIPPER_FORCE,
        )


    # ========================================================
    # MOLMO INFERENCE
    # ========================================================

    def infer(self, vla_step):
        self.check_franka_error()

        side_rgb, wrist_rgb, robot_state = self.get_observation()

        print()
        print("=" * 70)
        print(f"VLA STEP {vla_step}/{MAX_VLA_STEPS}")
        print("=" * 70)
        print(f"Robot state: {np.array2string(robot_state, precision=4)}")

        self.current_vla_state = robot_state.copy()

        # Save exactly the images passed to Molmo for this inference step.
        Image.fromarray(side_rgb.astype(np.uint8)).save(
            os.path.join(self.image_dir, f"step_{vla_step:03d}_side.png")
        )
        Image.fromarray(wrist_rgb.astype(np.uint8)).save(
            os.path.join(self.image_dir, f"step_{vla_step:03d}_wrist.png")
        )

        inference_start = time.perf_counter()

        predicted_actions, inference_ms = self.molmo.predict(
            external_image=side_rgb,
            wrist_image=wrist_rgb,
            instruction=INSTRUCTION,
            robot_state=robot_state,
        )

        wall_time_ms = (time.perf_counter() - inference_start) * 1000.0

        if inference_ms is not None:
            print(f"Inference: server={inference_ms:.1f} ms, wall={wall_time_ms:.1f} ms")
        else:
            print(f"Inference wall time: {wall_time_ms:.1f} ms")

        return self.validate_action_trajectory(predicted_actions)


    # ========================================================
    # ACTION CHUNK
    # ========================================================

    def execute_action_chunk(self, predicted_actions, vla_step):
        action_count = min(ACTIONS_PER_CHUNK, len(predicted_actions))
        print(f"Executing {action_count}/{len(predicted_actions)} Molmo actions...")

        executed_actions = []

        # Absolute scheduling prevents timing drift across the 15-action chunk.
        next_action_time = time.perf_counter()

        for action_index in range(action_count):
            self.check_franka_error()

            target_joint_pos = np.asarray(
                predicted_actions[action_index, :7],
                dtype=np.float64,
            )
            gripper_value = float(predicted_actions[action_index, 7])

            complete_action = np.concatenate([
                target_joint_pos,
                np.array([gripper_value], dtype=np.float64),
            ])

            print(
                f"Action {action_index + 1}/{action_count}: "
                f"{np.array2string(complete_action, precision=6)}"
            )

            if not DRY_RUN:
                # Arm and gripper are commanded during the same policy step.
                self.set_joint_target(target_joint_pos)
                self.execute_gripper_action(gripper_value)
                executed_actions.append(complete_action.copy())

            next_action_time += ACTION_DT
            sleep_time = next_action_time - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)

        # Store the state sent to Molmo and all actions actually executed.
        self.write_log("=" * 80)
        self.write_log(f"VLA STEP = {vla_step}")
        self.write_log(
            "Robot current state = "
            + np.array2string(
                self.current_vla_state,
                precision=6,
                separator=", ",
            )
        )

        self.write_log(f"Executed actions for this {action_count}-action chunk =")

        if executed_actions:
            self.write_log(
                np.array2string(
                    np.asarray(executed_actions, dtype=np.float64),
                    precision=6,
                    separator=", ",
                )
            )
        else:
            self.write_log("No actions executed (DRY_RUN=True).")

        self.write_log("")


    # ========================================================
    # CLOSED LOOP
    # ========================================================

    def run(self):
        print()
        print("=" * 70)
        print("MolmoAct2 Franka Controller")
        print("=" * 70)

        print(f"Experiment: {EXPERIMENT_NAME}")
        print(f"Instruction: {INSTRUCTION}")
        print(f"Actions per chunk: {ACTIONS_PER_CHUNK}")
        print(f"Policy action rate: {ACTION_RATE_HZ:.1f} Hz")
        print(f"Dry run: {DRY_RUN}")

        for vla_step in range(1, MAX_VLA_STEPS + 1):
            self.check_franka_error()
            step_start = time.perf_counter()

            try:
                predicted_actions = self.infer(vla_step)
            except ModelTrajectoryRejected as exception:
                print(f"Skipping VLA step {vla_step}: {exception}")
                continue

            self.execute_action_chunk(predicted_actions, vla_step)

            print(f"VLA step time: {time.perf_counter() - step_start:.3f} s")


    # ========================================================
    # SHUTDOWN
    # ========================================================

    def close(self):
        for camera_name in ("side", "wrist"):
            writer = getattr(self, f"{camera_name}_video_writer", None)

            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass

        if self.cameras is not None:
            try:
                self.cameras.close()
            except Exception as exception:
                print(f"Camera shutdown warning: {exception}")

        if self.gripper is not None:
            try:
                self.gripper.close_connection()
            except Exception as exception:
                print(f"Gripper shutdown warning: {exception}")


# ============================================================
# MAIN
# ============================================================

def main():
    # Spawn avoids inheriting potentially unsafe robot/camera resources.
    mp.set_start_method("spawn", force=True)

    experiment_dir = create_experiment_dir()

    print()
    print("=" * 70)
    print(f"Starting experiment: {EXPERIMENT_NAME}")
    print(f"Output directory: {experiment_dir}")
    print("=" * 70)

    # Main process -> Franka process.
    shared_target = mp.RawArray(ctypes.c_double, 7)
    target_version = mp.RawValue(ctypes.c_uint64, 0)

    # Franka process -> main process.
    shared_state = mp.RawArray(ctypes.c_double, 7)
    state_version = mp.RawValue(ctypes.c_uint64, 0)

    state_ready = mp.Event()
    stop_event = mp.Event()
    error_queue = mp.Queue(maxsize=1)

    franka_process = mp.Process(
        target=franka_control_process,
        args=(
            shared_target,
            target_version,
            shared_state,
            state_version,
            state_ready,
            stop_event,
            error_queue,
        ),
        daemon=False,
    )

    franka_process.start()
    print("Waiting for Franka control process...")

    if not state_ready.wait(timeout=10.0):
        stop_event.set()
        franka_process.join(timeout=2.0)

        if franka_process.is_alive():
            franka_process.terminate()
            franka_process.join(timeout=2.0)

        raise RuntimeError("Franka control process did not initialize.")

    controller = None

    try:
        controller = VLAController(
            shared_target,
            target_version,
            shared_state,
            state_version,
            stop_event,
            error_queue,
            experiment_dir,
        )

        print()
        print("WARNING: robot motion is enabled.")
        print("Keep the Franka user-stop available and verify that the workspace is clear.")

        input("Press Enter to start MolmoAct2 control...")
        controller.run()

    except KeyboardInterrupt:
        print("\nStopped by user.")

    except Exception as exception:
        print(f"\nController error: {exception}")
        raise

    finally:
        print("Shutting down...")

        stop_event.set()

        if controller is not None:
            controller.close()

        franka_process.join(timeout=5.0)

        if franka_process.is_alive():
            print("Franka process did not stop normally. Terminating it.")
            franka_process.terminate()
            franka_process.join(timeout=2.0)

        try:
            while True:
                print("\nFranka process error:")
                print(error_queue.get_nowait())
        except Empty:
            pass

        print("Shutdown complete.")


if __name__ == "__main__":
    main()