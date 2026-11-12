"""Shared wire observation decoding; no robot or model assumptions."""
import numpy as np
from .. import colosseum_pb2 as pb
from ..tensors import tensor_to_numpy

def _rgb(image: pb.Image | None, sensor_id: str) -> np.ndarray:
    """RGB uint8 (height, width, 3) from a RAW_RGB, JPEG or PNG sensor."""
    if image is None or image.width < 1 or image.height < 1:
        raise ValueError(f"invalid RGB sensor {sensor_id}")
    if image.encoding == pb.RAW_RGB:
        if len(image.data) != image.width * image.height * 3:
            raise ValueError(f"invalid RGB sensor {sensor_id}")
        return np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, 3).copy()
    if image.encoding not in (pb.JPEG, pb.PNG):
        raise ValueError(f"invalid RGB sensor {sensor_id}")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("JPEG/PNG observations require opencv-python (uv sync --extra demo)") from exc
    decoded = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape[:2] != (image.height, image.width):
        raise ValueError(f"invalid RGB sensor {sensor_id}")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _state(observation: pb.Observation, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name not in observation.state:
        raise ValueError(f"observation is missing {name}")
    value = np.asarray(tensor_to_numpy(observation.state[name]), dtype=np.float32).reshape(-1)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"invalid observation state {name}")
    return value
