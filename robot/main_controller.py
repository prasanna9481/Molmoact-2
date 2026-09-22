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
GRIPPER_CLOSE_THRESHOLD = float(CONFIG["gripper"]["close_threshold"])

DRY_RUN = bool(CONFIG["runtime"]["dry_run"])


# ============================================================
# EXPERIMENT DIRECTORY
# ============================================================

def create_experiment_dir():
    os.makedirs(EXPERIMENT_ROOT, exist_ok=True)

    iteration = 1
    while os.path.exists(os.path.join(EXPERIMENT_ROOT, f"iteration_{iteration}")):
        iteration += 1

    experiment_dir = os.path.join(EXPERIMENT_ROOT, f"iteration_{iteration}")
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
    """Publish a consistent 7-D vector to shared memory."""
    values = np.asarray(values, dtype=np.float64)

    version_counter.value += 1  # odd = write in progress
    for i in range(7):
        shared_array[i] = float(values[i])
    version_counter.value += 1  # even = complete


def read_shared_vector(shared_array, version_counter):
    """Read a consistent shared-memory snapshot."""
    while True:
        before = version_counter.value

        if before % 2:
            continue

        values = np.array([shared_array[i] for i in range(7)], dtype=np.float64)
        after = version_counter.value

        if before == after and after % 2 == 0:
            return values


def read_shared_vector_if_updated(
    shared_array,
    version_counter,
    previous_version,
    previous_values,
):
    """Read the target only when a complete newer value was published."""
    current_version = version_counter.value

    if current_version == previous_version or current_version % 2:
        return previous_values, previous_version

    values = np.array([shared_array[i] for i in range(7)], dtype=np.float64)
    version_after = version_counter.value

    if current_version == version_after and version_after % 2 == 0:
        return values, version_after

    return previous_values, previous_version


# ============================================================
# CONTROL PERIOD
# ============================================================

def get_dt_seconds(control_period):
    """Convert pylibfranka control period to seconds."""
    try:
        dt = float(control_period)
        if np.isfinite(dt) and 0.0001 <= dt <= 0.01:
            return dt
    except Exception:
        pass

    try:
        dt = float(control_period.to_sec())
        if np.isfinite(dt) and 0.0001 <= dt <= 0.01:
            return dt
    except Exception:
        pass

    return 0.001


# ============================================================
# SMOOTH STOP
# ============================================================

def smoothly_stop_control(control, cmd_q, cmd_vel, cmd_acc):
    """Decelerate the commanded trajectory to zero without time constants."""
    while True:
        _, control_period = control.readOnce()
        dt = get_dt_seconds(control_period)

        # Acceleration required to bring the current velocity toward zero.
        stop_acc = np.clip(-cmd_vel / max(dt, 1e-6), -MAX_JOINT_ACCELERATION, MAX_JOINT_ACCELERATION)

        # Limit acceleration change per cycle so commanded jerk stays bounded.
        cmd_acc += np.clip(stop_acc - cmd_acc, -MAX_JOINT_JERK * dt, MAX_JOINT_JERK * dt)
        cmd_acc = np.clip(cmd_acc, -MAX_JOINT_ACCELERATION, MAX_JOINT_ACCELERATION)

        old_vel = cmd_vel.copy()
        cmd_vel = np.clip(cmd_vel + cmd_acc * dt, -MAX_JOINT_VELOCITY, MAX_JOINT_VELOCITY)

        # Stop at zero rather than allowing the deceleration step to reverse direction.
        crossed_zero = old_vel * cmd_vel <= 0.0
        cmd_vel[crossed_zero] = 0.0
        cmd_acc[crossed_zero] = 0.0
        cmd_q += cmd_vel * dt

        if np.max(np.abs(cmd_vel)) < 0.001 and np.max(np.abs(cmd_acc)) < 0.01:
            final_command = JointPositions(cmd_q.tolist())
            final_command.motion_finished = True
            control.writeOnce(final_command)
            return

        control.writeOnce(JointPositions(cmd_q.tolist()))


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
    """
    High-rate Franka process.

    Molmo only updates target_q.
    cmd_q / cmd_vel / cmd_acc remain continuous and smoothly follow
    the latest target while respecting configured velocity,
    acceleration and jerk limits.
    """

    robot = None
    control = None

    try:
        print(f"Connecting Franka control process to {ROBOT_IP}...", flush=True)

        robot = Robot(ROBOT_IP, RealtimeConfig.kIgnore)

        # Outer loop exists so a fresh control session can be created
        # after a recoverable Franka reflex.
        while not stop_event.is_set():
            try:
                control = robot.start_joint_position_control(
                    ControllerMode.JointImpedance
                )

                # Start from the actual physical robot position at rest.
                robot_state, _ = control.readOnce()
                measured_q = np.asarray(robot_state.q, dtype=np.float64)

                cmd_q = measured_q.copy()
                cmd_vel = np.zeros(7, dtype=np.float64)
                cmd_acc = np.zeros(7, dtype=np.float64)

                target_q = measured_q.copy()
                last_target_version = target_version.value

                write_shared_vector(shared_state, state_version, measured_q)
                control.writeOnce(JointPositions(cmd_q.tolist()))

                state_ready.set()
                print("Franka control process ready.", flush=True)

                # ------------------------------------------------
                # HIGH-RATE CONTROL LOOP
                # ------------------------------------------------

                while not stop_event.is_set():
                    robot_state, control_period = control.readOnce()
                    dt = get_dt_seconds(control_period)

                    measured_q = np.asarray(robot_state.q, dtype=np.float64)
                    write_shared_vector(shared_state, state_version, measured_q)

                    # Only the target changes when Molmo publishes another action.
                    target_q, last_target_version = read_shared_vector_if_updated(
                        shared_target,
                        target_version,
                        last_target_version,
                        target_q,
                    )

                    # Distance and direction to the latest Molmo target.
                    error = target_q - cmd_q
                    direction = np.sign(error)

                    # Largest velocity that can still stop at the target:
                    # v^2 = 2*a*distance. This replaces POSITION_TIME_CONSTANT.
                    stopping_vel = np.sqrt(2.0 * MAX_JOINT_ACCELERATION * np.abs(error))
                    desired_vel = direction * np.minimum(MAX_JOINT_VELOCITY, stopping_vel)

                    # Acceleration required to approach desired velocity directly.
                    # No VELOCITY_TIME_CONSTANT is used.
                    desired_acc = np.clip(
                        (desired_vel - cmd_vel) / max(dt, 1e-6),
                        -MAX_JOINT_ACCELERATION,
                        MAX_JOINT_ACCELERATION,
                    )

                    # Bound acceleration change per cycle -> jerk limiting.
                    cmd_acc += np.clip(
                        desired_acc - cmd_acc,
                        -MAX_JOINT_JERK * dt,
                        MAX_JOINT_JERK * dt,
                    )
                    cmd_acc = np.clip(cmd_acc, -MAX_JOINT_ACCELERATION, MAX_JOINT_ACCELERATION)

                    # Integrate acceleration -> velocity, while respecting both the
                    # configured velocity bound and the target stopping velocity.
                    next_vel = np.clip(cmd_vel + cmd_acc * dt, -MAX_JOINT_VELOCITY, MAX_JOINT_VELOCITY)
                    next_vel = np.clip(next_vel, -stopping_vel, stopping_vel)
                    next_q = cmd_q + next_vel * dt

                    # Do not step through the current Molmo target.
                    crossed_target = (((error >= 0.0) & (next_q >= target_q)) |
                                      ((error < 0.0) & (next_q <= target_q)))
                    next_q[crossed_target] = target_q[crossed_target]
                    next_vel[crossed_target] = 0.0
                    cmd_acc[crossed_target] = 0.0
                    cmd_q, cmd_vel = next_q, next_vel

                    if np.any(cmd_q < JOINT_LOWER_LIMIT) or np.any(cmd_q > JOINT_UPPER_LIMIT):
                        raise RuntimeError(
                            "Generated trajectory exceeded Franka joint limits."
                        )

                    control.writeOnce(JointPositions(cmd_q.tolist()))

                # Normal program shutdown.
                smoothly_stop_control(control, cmd_q, cmd_vel, cmd_acc)

            except Exception as exception:
                if stop_event.is_set():
                    break

                print(f"\nFranka control error: {exception}", flush=True)

                # Stop the failed control session.
                try:
                    robot.stop()
                except Exception:
                    pass

                # IMPORTANT:
                # Release the old ActiveControl object before creating another
                # control session. Otherwise libfranka reports:
                # "another control or read operation is running".
                if control is not None:
                    try:
                        del control
                    except Exception:
                        pass
                    control = None

                print("Attempting automatic error recovery...", flush=True)

                try:
                    robot.automatic_error_recovery()
                    print("Franka automatic error recovery succeeded.", flush=True)
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
# VLA CONTROLLER
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

        self.video_fps = 2.0
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

        if writer is None:
            height, width = rgb_frame.shape[:2]
            path = os.path.join(self.video_dir, f"{camera_name}_trajectory.mp4")

            writer = cv2.VideoWriter(
                path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.video_fps,
                (width, height),
                True,
            )

            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {path}")

            setattr(self, writer_attr, writer)

        writer.write(
            cv2.cvtColor(
                np.asarray(rgb_frame, dtype=np.uint8),
                cv2.COLOR_RGB2BGR,
            )
        )


    # ========================================================
    # LOGGING
    # ========================================================

    def write_log(self, text):
        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(text + "\n")


    # ========================================================
    # ROBOT STATE
    # ========================================================

    def get_joint_state(self):
        return read_shared_vector(
            self.shared_state,
            self.state_version,
        ).astype(np.float32)


    def get_model_state(self):
        joint_q = self.get_joint_state()
        gripper_position = self.gripper.get_raw_position()

        return np.concatenate([
            joint_q,
            np.array([gripper_position], dtype=np.float32),
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
    # FRANKA TARGET
    # ========================================================

    def set_joint_target(self, target_q):
        target_q = np.asarray(target_q, dtype=np.float64)

        if target_q.shape != (7,):
            raise ValueError(f"Expected joint target shape (7,), got {target_q.shape}")

        if not np.all(np.isfinite(target_q)):
            raise ValueError("Joint target contains NaN or Inf.")

        if np.any(target_q < JOINT_LOWER_LIMIT) or np.any(target_q > JOINT_UPPER_LIMIT):
            raise ModelTrajectoryRejected(
                "Molmo joint target is outside Franka joint limits."
            )

        write_shared_vector(
            self.shared_target,
            self.target_version,
            target_q,
        )


    # ========================================================
    # MODEL TRAJECTORY VALIDATION
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

        if np.any(predicted_q < JOINT_LOWER_LIMIT) or np.any(predicted_q > JOINT_UPPER_LIMIT):
            raise ModelTrajectoryRejected(
                "Molmo trajectory contains a target outside Franka joint limits."
            )

        initial_delta = predicted_q[0] - current_q
        max_initial_offset = float(np.max(np.abs(initial_delta)))

        if len(predicted_q) > 1:
            waypoint_delta = np.diff(predicted_q, axis=0)
            max_waypoint_delta = float(np.max(np.abs(waypoint_delta)))
        else:
            max_waypoint_delta = 0.0

        print(f"Current q:          {np.array2string(current_q, precision=4)}")
        print(f"Molmo first q:      {np.array2string(predicted_q[0], precision=4)}")
        print(f"Initial delta:      {np.array2string(initial_delta, precision=4)}")
        print(f"Max initial offset: {max_initial_offset:.4f} rad")
        print(f"Max waypoint delta: {max_waypoint_delta:.4f} rad")

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


    # ========================================================
    # GRIPPER
    # ========================================================

    def execute_gripper_action(self, gripper_value):
        if not ENABLE_GRIPPER:
            return

        value = float(np.clip(gripper_value, 0.0, 1.0))

    

        robotiq_position = int(round(np.clip(value, 0.0, 1.0) * 255.0))

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
        print(f"Robot state: {np.array2string(robot_state, precision=4)}")

        self.current_vla_state = robot_state.copy()

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

        wall_ms = (time.perf_counter() - inference_start) * 1000.0

        if inference_ms is not None:
            print(
                f"Inference: server={inference_ms:.1f} ms, "
                f"wall={wall_ms:.1f} ms"
            )
        else:
            print(f"Inference wall time: {wall_ms:.1f} ms")

        return self.validate_action_trajectory(predicted_actions)


    # ========================================================
    # ACTION CHUNK
    # ========================================================

    def execute_action_chunk(self, predicted_actions, vla_step):
        """
        Execute Molmo's chunk at the configured policy rate.

        Important:
        - Do NOT wait for every waypoint to physically settle.
        - Franka continuously follows the streamed joint targets.
        - Arm and gripper commands are issued in the same policy step.
        """

        action_count = min(ACTIONS_PER_CHUNK, len(predicted_actions))

        print(
            f"Executing {action_count}/{len(predicted_actions)} "
            f"Molmo actions..."
        )

        executed_actions = []
        next_action_time = time.perf_counter()

        for action_index in range(action_count):
            self.check_franka_error()

            target_q = np.asarray(
                predicted_actions[action_index, :7],
                dtype=np.float64,
            )

            gripper_value = float(predicted_actions[action_index, 7])

            full_action = np.concatenate([
                target_q,
                np.array([gripper_value], dtype=np.float64),
            ])

            print(
                f"Action {action_index + 1}/{action_count}: "
                f"{np.array2string(full_action, precision=6)}"
            )

            if not DRY_RUN:
                # Publish both arm and gripper commands in the same action step.
                #
                # The Franka control process smooths the arm target internally;
                # therefore we do not block here waiting for every waypoint.
                self.set_joint_target(target_q)
                self.execute_gripper_action(gripper_value)

                executed_actions.append(full_action.copy())

            # Keep the policy action stream at the configured 15 Hz.
            next_action_time += ACTION_DT
            sleep_time = next_action_time - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)

        # ----------------------------------------------------
        # EXPERIMENT LOG
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
            f"Executed actions for this {action_count}-action chunk ="
        )

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
                print(f"Skipping VLA step {vla_step}: {exception}")
                continue

            self.execute_action_chunk(predicted_actions, vla_step)

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