#!/usr/bin/env python3

import ctypes
import multiprocessing as mp
import os
import shutil
import time
import traceback
from queue import Empty

import cv2
import numpy as np
import yaml
from PIL import Image
from pylibfranka import ControllerMode, JointPositions, RealtimeConfig, Robot
from ruckig import InputParameter, OutputParameter, Ruckig

from cameras.zed_camera import DualZEDCameras
from gripper.robotiq_gripper import RobotiqGripper
from inference.molmo_client import MolmoActClient


# Configuration
CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

EXPERIMENT_NAME = CONFIG["experiment"]["name"]
EXPERIMENT_ROOT = CONFIG["experiment"]["root_dir"]
INSTRUCTION = CONFIG["task"]["instruction"]

ROBOT_IP = CONFIG["franka"]["robot_ip"]
JOINT_LOWER_LIMIT = np.asarray(CONFIG["franka"]["joint_lower_limit"], dtype=np.float64)
JOINT_UPPER_LIMIT = np.asarray(CONFIG["franka"]["joint_upper_limit"], dtype=np.float64)
MAX_JOINT_VELOCITY = np.full(7, float(CONFIG["franka"]["max_joint_velocity"]), dtype=np.float64)
MAX_JOINT_ACCELERATION = np.full(7, float(CONFIG["franka"]["max_joint_acceleration"]), dtype=np.float64)
MAX_JOINT_JERK = np.full(7, float(CONFIG["franka"]["max_joint_jerk"]), dtype=np.float64)

MOLMO_SERVER = CONFIG["molmo"]["server_url"]
MAX_VLA_STEPS = int(CONFIG["molmo"]["max_vla_steps"])
ACTIONS_PER_CHUNK = int(CONFIG["molmo"]["actions_per_chunk"])
ACTION_RATE_HZ = float(CONFIG["molmo"]["action_rate_hz"])
ACTION_DT = 1.0 / ACTION_RATE_HZ
MAX_MODEL_WAYPOINT_DELTA = float(CONFIG["molmo"]["max_model_waypoint_delta"])
MAX_INITIAL_TARGET_OFFSET = float(CONFIG["molmo"]["max_initial_target_offset"])

ENABLE_GRIPPER = bool(CONFIG["gripper"]["enabled"])
GRIPPER_SPEED = int(CONFIG["gripper"]["speed"])
GRIPPER_FORCE = int(CONFIG["gripper"]["force"])
DRY_RUN = bool(CONFIG["runtime"]["dry_run"])


def create_experiment_dir():
    """Create the next iteration directory and copy the active config into it."""
    os.makedirs(EXPERIMENT_ROOT, exist_ok=True)

    iteration = 1
    while os.path.exists(os.path.join(EXPERIMENT_ROOT, f"iteration_{iteration}")):
        iteration += 1

    experiment_dir = os.path.join(EXPERIMENT_ROOT, f"iteration_{iteration}")
    os.makedirs(os.path.join(experiment_dir, "images"))
    os.makedirs(os.path.join(experiment_dir, "videos"))
    shutil.copy2(CONFIG_PATH, os.path.join(experiment_dir, "config.yaml"))
    return experiment_dir


class ModelTrajectoryRejected(RuntimeError):
    """Raised when a model trajectory is unsafe or malformed."""


# Shared-memory helpers use an odd/even version counter to avoid partial reads.
def write_shared_vector(shared_array, version_counter, values):
    values = np.asarray(values, dtype=np.float64)

    version_counter.value += 1
    for index in range(7):
        shared_array[index] = float(values[index])
    version_counter.value += 1


def read_shared_vector(shared_array, version_counter):
    while True:
        version_before = version_counter.value
        if version_before % 2:
            continue

        values = np.array([shared_array[i] for i in range(7)], dtype=np.float64)
        version_after = version_counter.value

        if version_before == version_after and version_after % 2 == 0:
            return values


def read_shared_vector_if_updated(
    shared_array, version_counter, previous_version, previous_values
):
    current_version = version_counter.value
    if current_version == previous_version or current_version % 2:
        return previous_values, previous_version

    values = np.array([shared_array[i] for i in range(7)], dtype=np.float64)
    version_after = version_counter.value

    if current_version == version_after and version_after % 2 == 0:
        return values, version_after

    return previous_values, previous_version


def franka_control_process(
    shared_target,
    target_version,
    shared_state,
    state_version,
    state_ready,
    stop_event,
    error_queue,
):
    """Run Franka joint control in a separate process with Ruckig smoothing."""
    robot = None
    control = None

    try:
        robot = Robot(ROBOT_IP, RealtimeConfig.kIgnore)

        while not stop_event.is_set():
            try:
                control = robot.start_joint_position_control(ControllerMode.JointImpedance)

                robot_state, _ = control.readOnce()
                measured_joints = np.asarray(robot_state.q, dtype=np.float64)
                write_shared_vector(shared_state, state_version, measured_joints)

                # Franka runs at roughly 1 kHz, so Ruckig is configured for 1 ms steps.
                trajectory_generator = Ruckig(7, 0.001)
                ruckig_input = InputParameter(7)
                ruckig_output = OutputParameter(7)

                ruckig_input.current_position = measured_joints.tolist()
                ruckig_input.current_velocity = [0.0] * 7
                ruckig_input.current_acceleration = [0.0] * 7
                ruckig_input.target_position = measured_joints.tolist()
                ruckig_input.target_velocity = [0.0] * 7
                ruckig_input.target_acceleration = [0.0] * 7
                ruckig_input.max_velocity = MAX_JOINT_VELOCITY.tolist()
                ruckig_input.max_acceleration = MAX_JOINT_ACCELERATION.tolist()
                ruckig_input.max_jerk = MAX_JOINT_JERK.tolist()

                target_joints = measured_joints.copy()
                last_target_version = target_version.value

                # Start from the measured pose to avoid a discontinuity at controller startup.
                control.writeOnce(JointPositions(measured_joints.tolist()))
                state_ready.set()
                print("Franka control ready.", flush=True)

                while not stop_event.is_set():
                    robot_state, _ = control.readOnce()
                    measured_joints = np.asarray(robot_state.q, dtype=np.float64)
                    write_shared_vector(shared_state, state_version, measured_joints)

                    new_target, new_version = read_shared_vector_if_updated(
                        shared_target,
                        target_version,
                        last_target_version,
                        target_joints,
                    )

                    if new_version != last_target_version:
                        target_joints = new_target.copy()
                        last_target_version = new_version

                        if np.any(target_joints < JOINT_LOWER_LIMIT) or np.any(
                            target_joints > JOINT_UPPER_LIMIT
                        ):
                            raise RuntimeError("Molmo target exceeded Franka joint limits.")

                        # Only update the target. Ruckig keeps its current velocity and acceleration.
                        ruckig_input.target_position = target_joints.tolist()
                        ruckig_input.target_velocity = [0.0] * 7
                        ruckig_input.target_acceleration = [0.0] * 7

                    result = trajectory_generator.update(ruckig_input, ruckig_output)
                    if result < 0:
                        raise RuntimeError(
                            f"Ruckig trajectory generation failed: {result}"
                        )

                    command_joints = np.asarray(
                        ruckig_output.new_position, dtype=np.float64
                    )
                    if np.any(command_joints < JOINT_LOWER_LIMIT) or np.any(
                        command_joints > JOINT_UPPER_LIMIT
                    ):
                        raise RuntimeError(
                            "Ruckig generated a position outside Franka limits."
                        )

                    control.writeOnce(JointPositions(command_joints.tolist()))

                    # Carry the generated position, velocity, and acceleration forward.
                    ruckig_output.pass_to_input(ruckig_input)

                final_command = JointPositions(list(ruckig_input.current_position))
                final_command.motion_finished = True
                control.writeOnce(final_command)

            except Exception as exception:
                if stop_event.is_set():
                    break

                print(f"Franka control error: {exception}", flush=True)

                try:
                    robot.stop()
                except Exception:
                    pass

                if control is not None:
                    try:
                        del control
                    except Exception:
                        pass
                    control = None

                try:
                    robot.automatic_error_recovery()
                    time.sleep(0.5)
                except Exception as recovery_exception:
                    error_text = (
                        "Franka recovery failed:\n"
                        f"{recovery_exception!r}\n"
                        f"{traceback.format_exc()}"
                    )
                    try:
                        error_queue.put_nowait(error_text)
                    except Exception:
                        pass

                    stop_event.set()
                    break
    finally:
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass


class VLAController:
    """Coordinate cameras, MolmoAct2 inference, Franka targets, and the gripper."""

    def __init__(
        self,
        shared_target,
        target_version,
        shared_state,
        state_version,
        stop_event,
        error_queue,
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

        self.gripper = RobotiqGripper()
        self.gripper.activate()
        self.cameras = DualZEDCameras()
        self.molmo = MolmoActClient(server_url=MOLMO_SERVER)

        self.video_fps = 2.0
        self.side_video_writer = None
        self.wrist_video_writer = None
        self.current_vla_state = None

    def _write_rgb_to_video(self, camera_name, rgb_frame):
        """Append one RGB frame to the selected experiment video."""
        if rgb_frame is None:
            return

        writer_attr = f"{camera_name}_video_writer"
        writer = getattr(self, writer_attr, None)

        if writer is None:
            height, width = rgb_frame.shape[:2]
            video_path = os.path.join(
                self.video_dir, f"{camera_name}_trajectory.mp4"
            )
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.video_fps,
                (width, height),
                True,
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {video_path}")
            setattr(self, writer_attr, writer)

        bgr_frame = cv2.cvtColor(
            np.asarray(rgb_frame, dtype=np.uint8), cv2.COLOR_RGB2BGR
        )
        writer.write(bgr_frame)

    def write_log(self, text):
        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(text + "\n")

    def get_joint_state(self):
        return read_shared_vector(
            self.shared_state, self.state_version
        ).astype(np.float32)

    def get_model_state(self):
        joint_state = self.get_joint_state()
        gripper_position = self.gripper.get_raw_position()
        return np.concatenate(
            [joint_state, np.array([gripper_position], dtype=np.float32)]
        )

    def check_franka_error(self):
        if not self.stop_event.is_set():
            return

        try:
            error_text = self.error_queue.get_nowait()
        except Empty:
            error_text = "Franka control process stopped unexpectedly."

        raise RuntimeError(error_text)

    def get_observation(self):
        side_rgb, wrist_rgb = self.cameras.get_frames_rgb()
        self._write_rgb_to_video("side", side_rgb)
        self._write_rgb_to_video("wrist", wrist_rgb)
        return side_rgb, wrist_rgb, self.get_model_state()

    def set_joint_target(self, target_joints):
        """Validate and publish a new seven-joint Franka target."""
        target_joints = np.asarray(target_joints, dtype=np.float64)

        if target_joints.shape != (7,):
            raise ValueError(
                f"Expected joint target shape (7,), got {target_joints.shape}"
            )
        if not np.all(np.isfinite(target_joints)):
            raise ValueError("Joint target contains NaN or Inf.")
        if np.any(target_joints < JOINT_LOWER_LIMIT) or np.any(
            target_joints > JOINT_UPPER_LIMIT
        ):
            raise ModelTrajectoryRejected(
                "Molmo joint target is outside Franka joint limits."
            )

        write_shared_vector(
            self.shared_target, self.target_version, target_joints
        )

    def validate_action_trajectory(self, predicted_actions):
        """Reject malformed, discontinuous, or out-of-limit Molmo trajectories."""
        predicted_actions = np.asarray(predicted_actions, dtype=np.float32)

        if predicted_actions.ndim != 2 or predicted_actions.shape[1] != 8:
            raise ModelTrajectoryRejected(
                f"Expected Molmo action shape (N, 8), got {predicted_actions.shape}"
            )
        if not np.all(np.isfinite(predicted_actions)):
            raise ModelTrajectoryRejected("Molmo returned NaN or Inf.")

        predicted_joints = predicted_actions[:, :7].astype(np.float64)
        current_joints = self.get_joint_state().astype(np.float64)

        if np.any(predicted_joints < JOINT_LOWER_LIMIT) or np.any(
            predicted_joints > JOINT_UPPER_LIMIT
        ):
            raise ModelTrajectoryRejected(
                "Molmo trajectory contains a target outside Franka joint limits."
            )

        max_initial_offset = float(
            np.max(np.abs(predicted_joints[0] - current_joints))
        )
        max_waypoint_delta = (
            float(np.max(np.abs(np.diff(predicted_joints, axis=0))))
            if len(predicted_joints) > 1
            else 0.0
        )

        if max_initial_offset > MAX_INITIAL_TARGET_OFFSET:
            raise ModelTrajectoryRejected(
                f"Initial target offset {max_initial_offset:.4f} rad exceeds "
                f"{MAX_INITIAL_TARGET_OFFSET:.4f} rad."
            )
        if max_waypoint_delta > MAX_MODEL_WAYPOINT_DELTA:
            raise ModelTrajectoryRejected(
                f"Waypoint delta {max_waypoint_delta:.4f} rad exceeds "
                f"{MAX_MODEL_WAYPOINT_DELTA:.4f} rad."
            )

        return predicted_actions

    def execute_gripper_action(self, gripper_value):
        if not ENABLE_GRIPPER:
            return

        normalized_value = float(np.clip(gripper_value, 0.0, 1.0))
        robotiq_position = int(round(normalized_value * 255.0))
        self.gripper.set_position(
            robotiq_position, speed=GRIPPER_SPEED, force=GRIPPER_FORCE
        )

    def infer(self, vla_step):
        """Capture the latest observation and request one action chunk."""
        self.check_franka_error()
        side_rgb, wrist_rgb, robot_state = self.get_observation()
        self.current_vla_state = robot_state.copy()

        Image.fromarray(side_rgb.astype(np.uint8)).save(
            os.path.join(self.image_dir, f"step_{vla_step:03d}_side.png")
        )
        Image.fromarray(wrist_rgb.astype(np.uint8)).save(
            os.path.join(self.image_dir, f"step_{vla_step:03d}_wrist.png")
        )

        predicted_actions, _ = self.molmo.predict(
            external_image=side_rgb,
            wrist_image=wrist_rgb,
            instruction=INSTRUCTION,
            robot_state=robot_state,
        )
        return self.validate_action_trajectory(predicted_actions)

    def execute_action_chunk(self, predicted_actions, vla_step):
        """
        Stream Molmo actions at the configured policy rate.

        Franka smooths each published target in its own control process, so the
        main process does not wait for individual waypoints to settle. The arm
        and gripper command for each policy step are issued together.
        """
        action_count = min(ACTIONS_PER_CHUNK, len(predicted_actions))
        executed_actions = []
        next_action_time = time.perf_counter()

        for action_index in range(action_count):
            self.check_franka_error()

            action = predicted_actions[action_index]
            target_joints = np.asarray(action[:7], dtype=np.float64)
            gripper_value = float(action[7])
            full_action = np.concatenate(
                [target_joints, np.array([gripper_value], dtype=np.float64)]
            )

            if not DRY_RUN:
                self.set_joint_target(target_joints)
                self.execute_gripper_action(gripper_value)
                executed_actions.append(full_action.copy())

            next_action_time += ACTION_DT
            sleep_time = next_action_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Keep one compact record per VLA step for experiment analysis.
        self.write_log("=" * 80)
        self.write_log(f"VLA STEP = {vla_step}")
        self.write_log(
            "Robot current state = "
            + np.array2string(
                self.current_vla_state, precision=6, separator=", "
            )
        )
        self.write_log(f"Executed actions for this {action_count}-action chunk =")

        if executed_actions:
            self.write_log(
                np.array2string(
                    np.asarray(executed_actions, dtype=np.float64),
                    precision=6,
                    separator=", ",
                    suppress_small=False,
                )
            )
        else:
            self.write_log("No actions executed (DRY_RUN=True).")

        self.write_log("")

    def run(self):
        print(
            f"Running '{EXPERIMENT_NAME}' | instruction: {INSTRUCTION} | "
            f"{ACTIONS_PER_CHUNK} actions/chunk at {ACTION_RATE_HZ:.1f} Hz"
        )

        for vla_step in range(1, MAX_VLA_STEPS + 1):
            self.check_franka_error()

            try:
                predicted_actions = self.infer(vla_step)
            except ModelTrajectoryRejected as exception:
                print(f"Skipping VLA step {vla_step}: {exception}")
                continue

            self.execute_action_chunk(predicted_actions, vla_step)

    def close(self):
        """Release video writers, cameras, and the gripper connection."""
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


def main():
    mp.set_start_method("spawn", force=True)
    experiment_dir = create_experiment_dir()

    # Main process publishes targets; Franka process publishes measured state.
    shared_target = mp.RawArray(ctypes.c_double, 7)
    target_version = mp.RawValue(ctypes.c_uint64, 0)
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

        print(
            f"Experiment output: {experiment_dir}\n"
            "WARNING: robot motion is enabled. Keep the Franka user-stop "
            "available and verify that the workspace is clear."
        )
        controller.run()

    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as exception:
        print(f"Controller error: {exception}")
        raise
    finally:
        stop_event.set()

        if controller is not None:
            controller.close()

        franka_process.join(timeout=5.0)
        if franka_process.is_alive():
            franka_process.terminate()
            franka_process.join(timeout=2.0)

        try:
            while True:
                print(f"Franka process error:\n{error_queue.get_nowait()}")
        except Empty:
            pass

        print("Shutdown complete.")


if __name__ == "__main__":
    main()