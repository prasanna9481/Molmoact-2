#!/usr/bin/env python3

import cv2
import numpy as np
import pyzed.sl as sl


CAMERAS = {
    "side": 39337350,
    "wrist": 19928076,
}


class ZEDCamera:
    def __init__(
        self,
        serial_number: int,
        resolution=sl.RESOLUTION.HD720,
        fps=30,
    ):
        self.serial_number = int(serial_number)

        self.zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.camera_resolution = resolution
        init_params.camera_fps = fps
        init_params.set_from_serial_number(self.serial_number)

        print(f"Opening ZED camera {self.serial_number}...")

        status = self.zed.open(init_params)

        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"Failed to open ZED camera "
                f"{self.serial_number}: {status}"
            )

        print(
            f"ZED camera {self.serial_number} "
            f"opened successfully."
        )

        self.image = sl.Mat()

    def get_frame_bgr(self):
        """
        Returns:
            np.ndarray
            shape: (H, W, 3)
            dtype: uint8
            color order: BGR
        """

        status = self.zed.grab()

        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"Failed to grab frame from "
                f"camera {self.serial_number}: {status}"
            )

        self.zed.retrieve_image(
            self.image,
            sl.VIEW.LEFT,
        )

        frame = self.image.get_data()

        # ZED returns BGRA.
        frame_bgr = frame[:, :, :3].copy()

        return frame_bgr

    def get_frame_rgb(self):
        """
        Returns RGB image suitable for MolmoAct2.
        """

        frame_bgr = self.get_frame_bgr()

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        return frame_rgb

    def close(self):
        self.zed.close()


class DualZEDCameras:
    def __init__(self):
        self.side = ZEDCamera(
            CAMERAS["side"]
        )

        self.wrist = ZEDCamera(
            CAMERAS["wrist"]
        )

    def get_frames_rgb(self):
        """
        Returns:
            side_rgb, wrist_rgb
        """

        side_rgb = self.side.get_frame_rgb()
        wrist_rgb = self.wrist.get_frame_rgb()

        return side_rgb, wrist_rgb

    def close(self):
        self.side.close()
        self.wrist.close()


def main():
    """
    Optional standalone camera test.
    """

    cameras = DualZEDCameras()

    try:
        print("Press 'q' to quit")

        while True:
            side_rgb, wrist_rgb = (
                cameras.get_frames_rgb()
            )

            # Convert back to BGR only for OpenCV display
            side_bgr = cv2.cvtColor(
                side_rgb,
                cv2.COLOR_RGB2BGR,
            )

            wrist_bgr = cv2.cvtColor(
                wrist_rgb,
                cv2.COLOR_RGB2BGR,
            )

            cv2.imshow(
                "Side Camera",
                side_bgr,
            )

            cv2.imshow(
                "Wrist Camera",
                wrist_bgr,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        cameras.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()