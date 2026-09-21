import serial
import time


PORT = "/dev/ttyUSB0"
BAUD = 115200
SLAVE_ID = 0x09


def crc16(data: bytes) -> int:
    crc = 0xFFFF

    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1

    return crc


def add_crc(frame: bytes) -> bytes:
    crc = crc16(frame)
    return frame + bytes([
        crc & 0xFF,
        (crc >> 8) & 0xFF,
    ])


def send(ser, payload, read_len=32):
    frame = add_crc(payload)

    ser.reset_input_buffer()
    ser.write(frame)
    ser.flush()

    time.sleep(0.1)

    return ser.read(read_len)


def read_status(ser):
    # Read 3 input registers from 0x07D0
    req = bytes([
        SLAVE_ID,
        0x04,
        0x07, 0xD0,
        0x00, 0x03,
    ])

    response = send(ser, req)

    print("STATUS:", response.hex(" "))

    return response


def write_command(ser, action, position, speed=50, force=20):
    # Write 3 registers starting at 0x03E8
    #
    # action:
    #   0x01 = activation
    #   0x09 = activate + go-to
    #
    # position:
    #   0   = open
    #   255 = closed

    req = bytes([
        SLAVE_ID,
        0x10,          # Write multiple registers
        0x03, 0xE8,    # Start register
        0x00, 0x03,    # 3 registers
        0x06,          # 6 bytes

        action,
        0x00,
        0x00,

        position,
        speed,
        force,
    ])

    response = send(ser, req)

    print("COMMAND:", response.hex(" "))


def activate(ser):
    print("Activating gripper...")

    # First reset
    write_command(
        ser,
        action=0x00,
        position=0,
        speed=0,
        force=0,
    )

    time.sleep(0.5)

    # Activate
    write_command(
        ser,
        action=0x01,
        position=0,
        speed=0,
        force=0,
    )

    time.sleep(3)

    read_status(ser)


def open_gripper(ser):
    print("Opening...")
    write_command(
        ser,
        action=0x09,
        position=0,
        speed=80,
        force=20,
    )


def close_gripper(ser):
    print("Closing...")
    write_command(
        ser,
        action=0x09,
        position=255,
        speed=80,
        force=20,
    )


def main():
    with serial.Serial(
        port=PORT,
        baudrate=BAUD,
        bytesize=8,
        parity=serial.PARITY_NONE,
        stopbits=1,
        timeout=1.0,
    ) as ser:

        print("Connected:", ser.name)

        read_status(ser)

        activate(ser)

        input("Press Enter to OPEN...")
        open_gripper(ser)
        time.sleep(3)

        input("Press Enter to CLOSE...")
        close_gripper(ser)
        time.sleep(3)

        read_status(ser)


if __name__ == "__main__":
    main()