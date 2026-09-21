#!/usr/bin/env python3

import time
import numpy as np

from pylibfranka import Robot, ControllerMode, JointPositions, RealtimeConfig
from gripper.robotiq_gripper import RobotiqGripper

ROBOT_IP = "192.168.103.1"

HOME_Q = np.array([
    5.19532303e-04,
    -7.85823643e-01,
    3.69584974e-04,
    -2.35402131e+00,
    -4.34993039e-04,
    1.57146442e+00,
    5.69308701e-04,
], dtype=np.float64)

MOVE_TIME = 6.0


def main():
    print("Connecting to Franka...")
    robot = Robot(ROBOT_IP, RealtimeConfig.kIgnore)

    print("Connecting to Robotiq...")
    gripper = RobotiqGripper()
    gripper.activate()

    start_q = np.asarray(
        robot.read_once().q,
        dtype=np.float64,
    )

    print("Moving robot home...")

    control = robot.start_joint_position_control(
        ControllerMode.JointImpedance
    )

    elapsed = 0.0

    while elapsed < MOVE_TIME:
        state, dt = control.readOnce()

        try:
            dt_sec = float(dt)
        except (TypeError, ValueError):
            dt_sec = 0.001

        if dt_sec <= 0 or dt_sec > 0.1:
            dt_sec = 0.001

        elapsed += dt_sec

        u = min(elapsed / MOVE_TIME, 1.0)

        alpha = (
            10.0 * u**3
            - 15.0 * u**4
            + 6.0 * u**5
        )

        q_command = start_q + alpha * (HOME_Q - start_q)

        control.writeOnce(
            JointPositions(q_command.tolist())
        )

    final_command = JointPositions(HOME_Q.tolist())
    final_command.motion_finished = True
    control.writeOnce(final_command)

    print("Robot home.")

    time.sleep(0.5)

    print("Opening gripper...")
    gripper.open()

    time.sleep(2.0)

    gripper.close_connection()

    print("Done.")


if __name__ == "__main__":
    main()