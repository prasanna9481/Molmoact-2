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
# CONFIGURATION
# ============================================================

CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)

EXPERIMENT_NAME = CONFIG["experiment"]["name"]
EXPERIMENT_ROOT = CONFIG["experiment"]["root_dir"]

INSTRUCTION = CONFIG["task"]["instruction"]

ROBOT_IP = CONFIG["franka"]["robot_ip"]

MOLMO_SERVER = CONFIG["molmo"]["server_url"]
MAX_VLA_STEPS = CONFIG["molmo"]["max_vla_steps"]
ACTIONS_PER_CHUNK = CONFIG["molmo"]["actions_per_chunk"]
ACTION_RATE_HZ = float(CONFIG["molmo"]["action_rate_hz"])
ACTION_DT = 1.0 / ACTION_RATE_HZ

MAX_MODEL_WAYPOINT_DELTA = float(CONFIG["molmo"]["max_model_waypoint_delta"])
MAX_INITIAL_TARGET_OFFSET = float(CONFIG["molmo"]["max_initial_target_offset"])

JOINT_LOWER_LIMIT = np.array(CONFIG["franka"]["joint_lower_limit"], dtype=np.float64)
JOINT_UPPER_LIMIT = np.array(CONFIG["franka"]["joint_upper_limit"], dtype=np.float64)

MAX_JOINT_VELOCITY = np.full(
    7, float(CONFIG["franka"]["max_joint_velocity"]), dtype=np.float64
)

MAX_JOINT_ACCELERATION = np.full(
    7, float(CONFIG["franka"]["max_joint_acceleration"]), dtype=np.float64
)

MAX_JOINT_JERK = np.full(
    7, float(CONFIG["franka"]["max_joint_jerk"]), dtype=np.float64
)

POSITION_TIME_CONSTANT = float(CONFIG["franka"]["position_time_constant"])
VELOCITY_TIME_CONSTANT = float(CONFIG["franka"]["velocity_time_constant"])

ENABLE_GRIPPER = bool(CONFIG["gripper"]["enabled"])
GRIPPER_SPEED = int(CONFIG["gripper"]["speed"])
GRIPPER_FORCE = int(CONFIG["gripper"]["force"])

DRY_RUN = bool(CONFIG["runtime"]["dry_run"])


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

    os.makedirs(experiment_dir)
    os.makedirs(os.path.join(experiment_dir, "images"))
    os.makedirs(os.path.join(experiment_dir, "videos"))

    shutil.copy2(CONFIG_PATH, os.path.join(experiment_dir, "config.yaml"))

    return experiment_dir


# ============================================================
# CUSTOM EXCEPTION
# ============================================================

class ModelTrajectoryRejected(RuntimeError):
    pass


# ============================================================
# SHARED MEMORY HELPERS
# ============================================================

def write_shared_vector(shared_array, version_counter, values):
    values = np.asarray(values, dtype=np.float64)

    version_counter.value += 1

    for joint_index in range(7):
        shared_array[joint_index] = float(values[joint_index])

    version_counter.value += 1


def read_shared_vector(shared_array, version_counter):
    while True:
        version_before = version_counter.value

        if version_before % 2 != 0:
            continue

        values = np.array(
            [shared_array[joint_index] for joint_index in range(7)],
            dtype=np.float64,
        )

        version_after = version_counter.value

        if version_before == version_after and version_after % 2 == 0:
            return values


def read_shared_vector_if_updated(
    shared_array,
    version_counter,
    previous_version,
    previous_values,
):
    current_version = version_counter.value

    if current_version == previous_version or current_version % 2 != 0:
        return previous_values, previous_version

    values = np.array(
        [shared_array[joint_index] for joint_index in range(7)],
        dtype=np.float64,
    )

    version_after = version_counter.value

    if current_version == version_after and version_after % 2 == 0:
        return values, version_after

    return previous_values, previous_version


# ============================================================
# CONTROL PERIOD HELPER
# ============================================================

def get_dt_seconds(control_period):
    try:
        seconds = float(control_period)

        if np.isfinite(seconds) and 0.0001 <= seconds <= 0.01:
            return seconds
    except Exception:
        pass

    try:
        seconds = float(control_period.to_sec())

        if np.isfinite(seconds) and 0.0001 <= seconds <= 0.01:
            return seconds
    except Exception:
        pass

    return 0.001


# ============================================================
# SMOOTH STOP
# ============================================================

def smoothly_stop_control(
    control,
    cmd_joint_pos,
    cmd_joint_vel,
    cmd_joint_acc,
):
    for _ in range(3000):
        _, control_period = control.readOnce()
        dt = get_dt_seconds(control_period)

        desired_joint_vel = np.zeros(7, dtype=np.float64)

        desired_joint_acc = (
            desired_joint_vel - cmd_joint_vel
        ) / VELOCITY_TIME_CONSTANT

        desired_joint_acc = np.clip(
            desired_joint_acc,
            -MAX_JOINT_ACCELERATION,
            MAX_JOINT_ACCELERATION,
        )

        acc_change = desired_joint_acc - cmd_joint_acc

        acc_change = np.clip(
            acc_change,
            -MAX_JOINT_JERK * dt,
            MAX_JOINT_JERK * dt,
        )

        cmd_joint_acc += acc_change
        cmd_joint_acc = np.clip(
            cmd_joint_acc,
            -MAX_JOINT_ACCELERATION,
            MAX_JOINT_ACCELERATION,
        )

        cmd_joint_vel += cmd_joint_acc * dt
        cmd_joint_vel = np.clip(
            cmd_joint_vel,
            -MAX_JOINT_VELOCITY,
            MAX_JOINT_VELOCITY,
        )

        cmd_joint_pos += cmd_joint_vel * dt

        control.writeOnce(JointPositions(cmd_joint_pos.tolist()))

        if (
            np.max(np.abs(cmd_joint_vel)) < 0.001
            and np.max(np.abs(cmd_joint_acc)) < 0.01
        ):
            final_command = JointPositions(cmd_joint_pos.tolist())
            final_command.motion_finished = True
            control.writeOnce(final_command)
            return


# ============================================================
# FRANKA CONTROL PROCESS
# ============================================================

def franka_control_process(
    shared_target,
    target_version,
    shared_state,
    state_version,
    state_ready,
    stop_event,
    error_queue,
):
    robot = None

    try:
        print(
            f"Connecting Franka control process to {ROBOT_IP}...",
            flush=True,
        )

        robot = Robot(
            ROBOT_IP,
            RealtimeConfig.kIgnore,
        )

        while not stop_event.is_set():
            try:
                control = robot.start_joint_position_control(
                    ControllerMode.JointImpedance
                )

                robot_state, _ = control.readOnce()
                measured_joint_pos = np.asarray(
                    robot_state.q,
                    dtype=np.float64,
                )

                cmd_joint_pos = measured_joint_pos.copy()
                cmd_joint_vel = np.zeros(7, dtype=np.float64)
                cmd_joint_acc = np.zeros(7, dtype=np.float64)

                target_joint_pos = measured_joint_pos.copy()
                last_target_version = target_version.value

                write_shared_vector(
                    shared_state,
                    state_version,
                    measured_joint_pos,
                )

                control.writeOnce(
                    JointPositions(cmd_joint_pos.tolist())
                )

                state_ready.set()

                print(
                    "Franka control process ready.",
                    flush=True,
                )

                while not stop_event.is_set():
                    robot_state, control_period = control.readOnce()
                    dt = get_dt_seconds(control_period)

                    measured_joint_pos = np.asarray(
                        robot_state.q,
                        dtype=np.float64,
                    )

                    write_shared_vector(
                        shared_state,
                        state_version,
                        measured_joint_pos,
                    )

                    target_joint_pos, last_target_version = (
                        read_shared_vector_if_updated(
                            shared_target,
                            target_version,
                            last_target_version,
                            target_joint_pos,
                        )
                    )

                    position_error = (
                        target_joint_pos - cmd_joint_pos
                    )

                    desired_joint_vel = (
                        position_error / POSITION_TIME_CONSTANT
                    )

                    desired_joint_vel = np.clip(
                        desired_joint_vel,
                        -MAX_JOINT_VELOCITY,
                        MAX_JOINT_VELOCITY,
                    )

                    desired_joint_acc = (
                        desired_joint_vel - cmd_joint_vel
                    ) / VELOCITY_TIME_CONSTANT

                    desired_joint_acc = np.clip(
                        desired_joint_acc,
                        -MAX_JOINT_ACCELERATION,
                        MAX_JOINT_ACCELERATION,
                    )

                    acc_change = (
                        desired_joint_acc - cmd_joint_acc
                    )

                    acc_change = np.clip(
                        acc_change,
                        -MAX_JOINT_JERK * dt,
                        MAX_JOINT_JERK * dt,
                    )

                    cmd_joint_acc += acc_change

                    cmd_joint_acc = np.clip(
                        cmd_joint_acc,
                        -MAX_JOINT_ACCELERATION,
                        MAX_JOINT_ACCELERATION,
                    )

                    cmd_joint_vel += cmd_joint_acc * dt

                    cmd_joint_vel = np.clip(
                        cmd_joint_vel,
                        -MAX_JOINT_VELOCITY,
                        MAX_JOINT_VELOCITY,
                    )

                    cmd_joint_pos += cmd_joint_vel * dt

                    if (
                        np.any(cmd_joint_pos < JOINT_LOWER_LIMIT)
                        or np.any(cmd_joint_pos > JOINT_UPPER_LIMIT)
                    ):
                        raise RuntimeError(
                            "Generated trajectory exceeded "
                            "Franka joint limits."
                        )

                    control.writeOnce(
                        JointPositions(
                            cmd_joint_pos.tolist()
                        )
                    )

                smoothly_stop_control(
                    control,
                    cmd_joint_pos,
                    cmd_joint_vel,
                    cmd_joint_acc,
                )

            except Exception as exception:
                if stop_event.is_set():
                    break

                print()
                print(
                    f"Franka control error: {exception}",
                    flush=True,
                )

                try:
                    robot.stop()
                except Exception:
                    pass

                print(
                    "Attempting automatic error recovery...",
                    flush=True,
                )

                try:
                    robot.automatic_error_recovery()

                    print(
                        "Franka automatic error recovery succeeded.",
                        flush=True,
                    )

                    time.sleep(0.5)

                except Exception as recovery_exception:
                    error_text = (
                        f"Franka recovery failed:\n"
                        f"{repr(recovery_exception)}\n"
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


# ============================================================
# MAIN VLA CONTROLLER
# ============================================================

class VLAController:
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

        print(f"Experiment directory: {self.experiment_dir}")
        print(f"Robot log file: {self.log_path}")

        print("Connecting to Robotiq gripper...")
        self.gripper = RobotiqGripper()
        self.gripper.activate()
        print("Robotiq gripper ready.")

        print("Opening ZED cameras...")
        self.cameras = DualZEDCameras()
        print("Cameras ready.")

        self.video_dir = os.path.join(self.experiment_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)
        self.video_fps = 15.0
        self.side_video_writer = None
        self.wrist_video_writer = None

        self.molmo = MolmoActClient(server_url=MOLMO_SERVER)
        print("MolmoAct2 client ready.")

    def _write_rgb_to_video(self, camera_name, rgb_frame):
        """
        Save RGB frames to an MP4 file without showing a GUI window.

        This keeps the existing inference and trajectory logic intact,
        while recording both camera streams as a video in the background.
        """
        if rgb_frame is None:
            return

        writer_attr = f"{camera_name}_video_writer"
        writer = getattr(self, writer_attr, None)

        if writer is None:
            height, width = rgb_frame.shape[:2]
            output_path = os.path.join(
                self.video_dir,
                f"{camera_name}_trajectory.mp4",
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                self.video_fps,
                (width, height),
                True,
            )

            if not writer.isOpened():
                raise RuntimeError(
                    f"Failed to open {camera_name} video writer: {output_path}"
                )

            setattr(self, writer_attr, writer)

        bgr_frame = cv2.cvtColor(
            np.asarray(rgb_frame, dtype=np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        writer.write(bgr_frame)


    # ========================================================
    # SIMPLE LOGGING
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
        robot_state = self.get_model_state()

        return side_rgb, wrist_rgb, robot_state


    # ========================================================
    # TARGET PUBLISHING
    # ========================================================

    def set_joint_target(self, target_joint_pos):
        target_joint_pos = np.asarray(target_joint_pos, dtype=np.float64)

        if target_joint_pos.shape != (7,):
            raise ValueError(
                f"Expected joint target shape (7,), got {target_joint_pos.shape}"
            )

        if not np.all(np.isfinite(target_joint_pos)):
            raise ValueError("Joint target contains NaN or Inf.")

        if np.any(target_joint_pos < JOINT_LOWER_LIMIT) or np.any(
            target_joint_pos > JOINT_UPPER_LIMIT
        ):
            raise ModelTrajectoryRejected(
                "Molmo joint target is outside Franka joint limits."
            )

        write_shared_vector(
            self.shared_target,
            self.target_version,
            target_joint_pos,
        )


    # ========================================================
    # MODEL TRAJECTORY VALIDATION
    # ========================================================

    def validate_action_trajectory(self, predicted_actions):
        predicted_actions = np.asarray(predicted_actions, dtype=np.float32)

        if predicted_actions.ndim != 2 or predicted_actions.shape[1] != 8:
            raise ModelTrajectoryRejected(
                f"Unexpected Molmo action shape: {predicted_actions.shape}"
            )

        if not np.all(np.isfinite(predicted_actions)):
            raise ModelTrajectoryRejected("Molmo returned NaN or Inf.")

        predicted_joint_pos = predicted_actions[:, :7].astype(np.float64)
        current_joint_pos = self.get_joint_state().astype(np.float64)

        if np.any(predicted_joint_pos < JOINT_LOWER_LIMIT) or np.any(
            predicted_joint_pos > JOINT_UPPER_LIMIT
        ):
            raise ModelTrajectoryRejected(
                "Molmo trajectory contains a target outside Franka joint limits."
            )

        initial_difference = predicted_joint_pos[0] - current_joint_pos
        max_initial_difference = float(np.max(np.abs(initial_difference)))

        if len(predicted_joint_pos) > 1:
            waypoint_difference = np.diff(predicted_joint_pos, axis=0)
            max_waypoint_difference = float(
                np.max(np.abs(waypoint_difference))
            )
        else:
            max_waypoint_difference = 0.0

        print(
            f"Current q:     "
            f"{np.array2string(current_joint_pos, precision=4)}"
        )

        print(
            f"Molmo first q: "
            f"{np.array2string(predicted_joint_pos[0], precision=4)}"
        )

        print(
            f"Initial delta: "
            f"{np.array2string(initial_difference, precision=4)}"
        )

        print(
            f"Max initial offset: "
            f"{max_initial_difference:.4f} rad"
        )

        print(
            f"Max waypoint delta: "
            f"{max_waypoint_difference:.4f} rad"
        )

        if max_initial_difference > MAX_INITIAL_TARGET_OFFSET:
            raise ModelTrajectoryRejected(
                f"Initial target offset {max_initial_difference:.4f} rad exceeds "
                f"{MAX_INITIAL_TARGET_OFFSET:.4f} rad."
            )

        if max_waypoint_difference > MAX_MODEL_WAYPOINT_DELTA:
            raise ModelTrajectoryRejected(
                f"Waypoint delta {max_waypoint_difference:.4f} rad exceeds "
                f"{MAX_MODEL_WAYPOINT_DELTA:.4f} rad."
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

        print(
            f"Gripper: model={gripper_value:.4f} "
            f"-> Robotiq={robotiq_position}"
        )

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

        print(
            f"Robot state: "
            f"{np.array2string(robot_state, precision=4)}"
        )

        # ----------------------------------------------------
        # Save current state for this VLA step.
        # This is the state that is sent to the model.
        # ----------------------------------------------------

        self.current_vla_state = robot_state.copy()

        Image.fromarray(
            side_rgb.astype(np.uint8)
        ).save(
            os.path.join(
                self.image_dir,
                f"step_{vla_step:03d}_side.png",
            )
        )

        Image.fromarray(
            wrist_rgb.astype(np.uint8)
        ).save(
            os.path.join(
                self.image_dir,
                f"step_{vla_step:03d}_wrist.png",
            )
        )

        inference_start = time.perf_counter()

        predicted_actions, inference_ms = self.molmo.predict(
            external_image=side_rgb,
            wrist_image=wrist_rgb,
            instruction=INSTRUCTION,
            robot_state=robot_state,
        )

        wall_time_ms = (
            time.perf_counter() - inference_start
        ) * 1000.0

        if inference_ms is not None:
            print(
                f"Inference: server={inference_ms:.1f} ms, "
                f"wall={wall_time_ms:.1f} ms"
            )
        else:
            print(
                f"Inference wall time: "
                f"{wall_time_ms:.1f} ms"
            )

        return self.validate_action_trajectory(
            predicted_actions
        )


    # ========================================================
    # ACTION CHUNK EXECUTION
    # ========================================================

    def execute_action_chunk(self, predicted_actions, vla_step):
        action_count = min(
            ACTIONS_PER_CHUNK,
            len(predicted_actions),
        )

        print(
            f"Executing {action_count}/{len(predicted_actions)} "
            f"Molmo actions..."
        )

        # ----------------------------------------------------
        # Store all actions actually executed in this chunk.
        # ----------------------------------------------------

        executed_actions = []
        next_action_time = time.perf_counter()

        for action_index in range(action_count):
            self.check_franka_error()

            target_joint_pos = np.asarray(
                predicted_actions[action_index, :7],
                dtype=np.float64,
            )

            gripper_value = float(
                predicted_actions[action_index, 7]
            )

            complete_action = np.concatenate([
                target_joint_pos,
                np.array([gripper_value], dtype=np.float64),
            ])

            print(
                f"Action {action_index + 1}/{action_count}: "
                f"{np.array2string(complete_action, precision=6)}"
            )

            if not DRY_RUN:
                self.set_joint_target(target_joint_pos)
                self.execute_gripper_action(gripper_value)

                # Only store it as executed if it was actually sent.
                executed_actions.append(complete_action.copy())

            next_action_time += ACTION_DT

            sleep_time = next_action_time - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)

        # ----------------------------------------------------
        # Write robot state + executed chunk into ONE log file.
        # ----------------------------------------------------

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

        self.write_log(
            f"Executed actions for this "
            f"{action_count}-action chunk ="
        )

        if executed_actions:
            executed_actions = np.asarray(
                executed_actions,
                dtype=np.float64,
            )

            self.write_log(
                np.array2string(
                    executed_actions,
                    precision=6,
                    separator=", ",
                    suppress_small=False,
                )
            )
        else:
            self.write_log(
                "No actions executed (DRY_RUN=True)."
            )

        self.write_log("")


    # ========================================================
    # CLOSED-LOOP VLA
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
        print(f"Max velocity: {MAX_JOINT_VELOCITY[0]:.3f} rad/s")
        print(f"Max acceleration: {MAX_JOINT_ACCELERATION[0]:.3f} rad/s^2")
        print(f"Max jerk: {MAX_JOINT_JERK[0]:.3f} rad/s^3")
        print(f"Dry run: {DRY_RUN}")
        print(f"Saving experiment to: {self.experiment_dir}")

        for vla_step in range(1, MAX_VLA_STEPS + 1):
            self.check_franka_error()

            step_start = time.perf_counter()

            try:
                predicted_actions = self.infer(vla_step)

            except ModelTrajectoryRejected as exception:
                print(
                    f"Skipping VLA step "
                    f"{vla_step}: {exception}"
                )
                continue

            self.execute_action_chunk(
                predicted_actions,
                vla_step,
            )

            print(
                f"VLA step time: "
                f"{time.perf_counter() - step_start:.3f} s"
            )


    # ========================================================
    # SHUTDOWN
    # ========================================================

    def close(self):
        for camera_name in ("side", "wrist"):
            writer = getattr(self, f"{camera_name}_video_writer", None)

            if writer is not None:
                try:
                    writer.release()
                    print(
                        f"Saved {camera_name} trajectory video to "
                        f"{os.path.join(self.video_dir, f'{camera_name}_trajectory.mp4')}"
                    )
                except Exception as exception:
                    print(
                        f"{camera_name} video shutdown warning: "
                        f"{exception}"
                    )

        if self.cameras is not None:
            try:
                self.cameras.close()
            except Exception as exception:
                print(
                    f"Camera shutdown warning: "
                    f"{exception}"
                )

        if self.gripper is not None:
            try:
                self.gripper.close_connection()
            except Exception as exception:
                print(
                    f"Gripper shutdown warning: "
                    f"{exception}"
                )


# ============================================================
# MAIN
# ============================================================

def main():
    mp.set_start_method("spawn", force=True)

    experiment_dir = create_experiment_dir()

    print()
    print("=" * 70)
    print(f"Starting experiment: {EXPERIMENT_NAME}")
    print(f"Output directory: {experiment_dir}")
    print("=" * 70)

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

    print("Waiting for Franka control process...")

    if not state_ready.wait(timeout=10.0):
        stop_event.set()
        franka_process.join(timeout=2.0)

        if franka_process.is_alive():
            franka_process.terminate()
            franka_process.join(timeout=2.0)

        raise RuntimeError(
            "Franka control process did not initialize."
        )

    print("Franka control process initialized.")

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
        print(
            "Keep the Franka user-stop available "
            "and verify that the workspace is clear."
        )

        input(
            "Press Enter to start MolmoAct2 control..."
        )

        controller.run()

    except KeyboardInterrupt:
        print("\nStopped by user.")

    except Exception as exception:
        print(
            f"\nController error: {exception}"
        )
        raise

    finally:
        print("Shutting down...")

        stop_event.set()

        if controller is not None:
            controller.close()

        franka_process.join(timeout=5.0)

        if franka_process.is_alive():
            print(
                "Franka process did not stop normally. "
                "Terminating it."
            )

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