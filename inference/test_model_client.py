import requests
import numpy as np
from PIL import Image
import json_numpy


SERVER_URL = "http://127.0.0.1:8000/act"


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def main():
    external = load_rgb("external.jpg")
    wrist = load_rgb("wrist.jpg")

    state = np.zeros(8, dtype=np.float32)

    payload = {
        "external_cam": external,
        "wrist_cam": wrist,
        "instruction": "pick up the object",
        "state": state,
    }

    body = json_numpy.dumps(payload)

    response = requests.post(
        SERVER_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        timeout=120,
    )

    print("HTTP status:", response.status_code)
    print("Raw response:")
    print(response.text)

    response.raise_for_status()

    result = json_numpy.loads(response.text)

    actions = np.asarray(result["actions"])

    print("\nInference time:", result.get("dt_ms"), "ms")
    print("Action shape:", actions.shape)
    print("Action dtype:", actions.dtype)

    print("\nActions:")
    print(actions)


if __name__ == "__main__":
    main()