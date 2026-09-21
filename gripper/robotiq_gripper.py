#!/usr/bin/env python3

import serial
import time


class RobotiqGripper:
    def __init__(
        self,
        port="/dev/ttyUSB0",
        baud=115200,
        slave_id=0x09,
        timeout=1.0,
    ):
        self.port = port
        self.baud = baud
        self.slave_id = slave_id

        print(f"Opening Robotiq serial port {port}...")

        self.ser = serial.Serial(
            port=port,
            baudrate=baud,
            bytesize=8,
            parity=serial.PARITY_NONE,
            stopbits=1,
            timeout=timeout,
        )

        print(f"Robotiq connected on {self.ser.name}")

    # ========================================================
    # MODBUS CRC
    # ========================================================

    @staticmethod
    def crc16(data: bytes) -> int:
        """Compute Modbus RTU CRC16."""
        crc = 0xFFFF

        for byte in data:
            crc ^= byte

            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1

        return crc

    @classmethod
    def add_crc(cls, frame: bytes) -> bytes:
        """Append Modbus CRC bytes to a frame."""
        crc = cls.crc16(frame)

        return frame + bytes([
            crc & 0xFF,
            (crc >> 8) & 0xFF,
        ])

    # ========================================================
    # SERIAL SEND
    # ========================================================

    def send(self, payload, read_len=32):
        """Send one Modbus RTU request and return the response."""
        frame = self.add_crc(payload)

        self.ser.reset_input_buffer()
        self.ser.write(frame)
        self.ser.flush()

        time.sleep(0.1)

        return self.ser.read(read_len)

    # ========================================================
    # STATUS
    # ========================================================

    def read_status_raw(self):
        """
        Read 3 input registers starting at 0x07D0.

        Expected response:
            slave
            function
            byte count
            6 data bytes
            CRC low
            CRC high

        Total: 11 bytes
        """
        req = bytes([
            self.slave_id,
            0x04,
            0x07, 0xD0,
            0x00, 0x03,
        ])

        return self.send(req, read_len=11)

    def read_status(self):
        """
        Read and decode the Robotiq gripper status.
        """
        response = self.read_status_raw()

        if len(response) != 11:
            raise RuntimeError(
                f"Invalid Robotiq response length: "
                f"{len(response)} bytes, expected 11. "
                f"RX={response.hex(' ')}"
            )

        # Basic Modbus response validation
        if response[0] != self.slave_id:
            raise RuntimeError(
                f"Unexpected slave ID: {response[0]}"
            )

        if response[1] != 0x04:
            raise RuntimeError(
                f"Unexpected function code: 0x{response[1]:02X}"
            )

        if response[2] != 6:
            raise RuntimeError(
                f"Unexpected byte count: {response[2]}"
            )

        # Six data bytes returned by the gripper
        data = response[3:9]

        status_byte = data[0]
        fault_byte = data[1]
        requested_position = data[2]
        actual_position = data[3]
        current = data[4]

        # Decode status bits
        gACT = status_byte & 0x01
        gGTO = (status_byte >> 3) & 0x01
        gSTA = (status_byte >> 4) & 0x03
        gOBJ = (status_byte >> 6) & 0x03

        activation_states = {
            0: "reset",
            1: "activation_in_progress",
            2: "reserved",
            3: "activation_complete",
        }

        object_states = {
            0: "moving",
            1: "stopped_while_opening",
            2: "stopped_while_closing",
            3: "position_reached",
        }

        return {
            "raw": response,
            "hex": response.hex(" "),

            # Raw status bytes
            "status_byte": status_byte,
            "fault": fault_byte,

            # Decoded status fields
            "gACT": gACT,
            "gGTO": gGTO,
            "gSTA": gSTA,
            "gOBJ": gOBJ,

            "activated": gACT == 1 and gSTA == 3,
            "activation_state": activation_states[gSTA],
            "object_state": object_states[gOBJ],

            # Position information
            "requested_position": requested_position,
            "actual_position": actual_position,

            # Motor current byte
            "current": current,

            # Useful convenience states
            "moving": gOBJ == 0,
            "object_detected": gOBJ in (1, 2),
            "position_reached": gOBJ == 3,
        }

    # ========================================================
    # COMMANDS
    # ========================================================

    def write_command(
        self,
        action,
        position,
        speed=50,
        force=20,
    ):
        """Write activation / position command registers."""
        position = int(max(0, min(255, position)))
        speed = int(max(0, min(255, speed)))
        force = int(max(0, min(255, force)))

        req = bytes([
            self.slave_id,
            0x10,
            0x03, 0xE8,
            0x00, 0x03,
            0x06,

            action,
            0x00,
            0x00,

            position,
            speed,
            force,
        ])

        return self.send(req)

    # ========================================================
    # HIGH-LEVEL FUNCTIONS
    # ========================================================

    def activate(self):
        """Reset, then activate the gripper."""
        print("Activating Robotiq gripper...")

        self.write_command(
            action=0x00,
            position=0,
            speed=0,
            force=0,
        )

        time.sleep(0.5)

        self.write_command(
            action=0x01,
            position=0,
            speed=0,
            force=0,
        )

        time.sleep(3)

        print("Robotiq activation complete.")

    def set_position(
        self,
        position,
        speed=50,
        force=20,
    ):
        """
        Set gripper position.

        Raw convention:
            0   = open
            255 = closed
        """
        self.write_command(
            action=0x09,
            position=position,
            speed=speed,
            force=force,
        )

    def open(self):
        """Open the gripper."""
        self.set_position(0)

    def close(self):
        """Close the gripper."""
        self.set_position(255)

    # ========================================================
    # MODEL-FRIENDLY STATE
    # ========================================================

    def get_raw_position(self):
        """Return the actual gripper position in the range 0-255."""
        status = self.read_status()

        return status["actual_position"]

    def get_position_normalized(self):
        """
        Return position normalized to [0, 1].

        0.0 = fully open
        1.0 = fully closed
        """
        raw_position = self.get_raw_position()

        return raw_position / 255.0

    def get_state(self):
        """
        Return the full decoded gripper state.
        """
        return self.read_status()

    # ========================================================
    # CONNECTION
    # ========================================================

    def close_connection(self):
        """Close the serial connection."""
        if self.ser.is_open:
            self.ser.close()

        print("Robotiq serial port closed.")