#!/usr/bin/env python3

import numpy as np
import requests
import json_numpy


class MolmoActClient:
    def __init__(
        self,
        server_url="http://127.0.0.1:8000/act",
        timeout=120,
    ):
        self.server_url = server_url
        self.timeout = timeout

    def predict(
        self,
        external_image,
        wrist_image,
        instruction,
        robot_state,
    ):
        """
        external_image:
            np.ndarray RGB uint8, shape (H, W, 3)

        wrist_image:
            np.ndarray RGB uint8, shape (H, W, 3)

        instruction:
            string

        robot_state:
            np.ndarray float32 shape (8,)
            [q1, q2, q3, q4, q5, q6, q7, gripper_state]
        """

        external_image = np.asarray(
            external_image,
            dtype=np.uint8,
        )

        wrist_image = np.asarray(
            wrist_image,
            dtype=np.uint8,
        )

        robot_state = np.asarray(
            robot_state,
            dtype=np.float32,
        )

        if robot_state.shape != (8,):
            raise ValueError(
                f"Expected robot_state shape (8,), "
                f"got {robot_state.shape}"
            )

        payload = {
            "external_cam": external_image,
            "wrist_cam": wrist_image,
            "instruction": instruction,
            "state": robot_state,
        }

        body = json_numpy.dumps(payload)

        response = requests.post(
            self.server_url,
            data=body,
            headers={
                "Content-Type": "application/json"
            },
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"MolmoAct server returned "
                f"{response.status_code}: "
                f"{response.text}"
            )

        result = json_numpy.loads(response.text)

        actions = np.asarray(
            result["actions"],
            dtype=np.float32,
        )

        return actions, result.get("dt_ms")