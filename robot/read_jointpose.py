#!/usr/bin/env python3


 #this script will explicitely read the joint positions of the robot and print them to the console. 
 # It is useful for debugging and verifying the robot's state.
import numpy as np
from pylibfranka import Robot, RealtimeConfig

ROBOT_IP = "192.168.103.1"

robot = Robot(
    ROBOT_IP,
    RealtimeConfig.kIgnore,
)

state = robot.read_once()

q = np.asarray(
    state.q,
    dtype=np.float64,
)

print("Current joint positions [rad]:")
print(q)

print("\nCurrent joint positions [deg]:")
print(np.degrees(q))