#!/usr/bin/env python3
"""
Franka Robot Initialization

Workflow:
1. Load Desk credentials from robot/.env.
2. Connect to the Franka Desk API and acquire the control token.
3. Read the joint state.
4. If all joints are already unlocked, continue directly.
5. Otherwise request joint unlock and poll until all joints report unlocked.
6. Activate FCI once the robot is unlocked.
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from franka_desk import FrankaDeskClient

ROBOT_IP = "192.168.103.1"
UNLOCK_TIMEOUT = 20.0
POLL_INTERVAL = 0.5

ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_FILE)

DESK_USERNAME = os.getenv("DESK_USERNAME")
DESK_PASSWORD = os.getenv("DESK_PASSWORD")


def joints_are_unlocked(client):
    joint_state = client.get_joints()

    if not isinstance(joint_state, list):
        raise RuntimeError(
            f"Unexpected get_joints() return type: {type(joint_state)} "
            f"value={joint_state}"
        )

    if len(joint_state) == 0:
        raise RuntimeError("get_joints() returned an empty list")

    for joint in joint_state:
        if not isinstance(joint, dict):
            raise RuntimeError(
                f"Unexpected joint entry type: {type(joint)} value={joint}"
            )

        if "brakeStatus" not in joint:
            raise RuntimeError(
                f"Missing brakeStatus in joint entry: {joint}"
            )

        if joint["brakeStatus"] != "Unlocked":
            return False

    return True


def wait_until_unlocked(client):
    print("Waiting for joints to become unlocked...")

    start = time.monotonic()

    while True:
        if joints_are_unlocked(client):
            print("Joints are unlocked.")
            return

        if time.monotonic() - start > UNLOCK_TIMEOUT:
            raise TimeoutError(
                f"Joints did not unlock within {UNLOCK_TIMEOUT} seconds."
            )

        time.sleep(POLL_INTERVAL)


def main():
    if not DESK_USERNAME:
        raise RuntimeError("DESK_USERNAME not found in robot/.env")

    if not DESK_PASSWORD:
        raise RuntimeError("DESK_PASSWORD not found in robot/.env")

    print("=" * 60)
    print("FRANKA INITIALIZATION")
    print("=" * 60)

    robot_url = f"https://{ROBOT_IP}"

    print(f"Connecting to {robot_url}...")

    client = FrankaDeskClient(
        robot_url,
        DESK_USERNAME,
        DESK_PASSWORD,
    )

    try:
        print("Taking control token...")
        client.take_control_token()
        print("Control token acquired.")

        print("Checking joint state...")

        if joints_are_unlocked(client):
            print("Joints are already unlocked.")
        else:
            print("Joints are locked. Requesting unlock...")
            client.unlock_joints()
            wait_until_unlocked(client)

        print("Activating FCI...")
        client.activate_fci()
        print("FCI activated.")

        print()
        print("=" * 60)
        print("ROBOT READY")
        print("=" * 60)

    except Exception as exc:
        print()
        print("Initialization failed:")
        print(exc)
        raise


if __name__ == "__main__":
    main()