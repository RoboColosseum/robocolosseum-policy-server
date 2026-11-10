import cv2
import numpy as np
import pytest

from colosseum_policy_server import colosseum_pb2 as pb
from colosseum_policy_server.model_adapters.observation import _rgb


def sensor(encoding, data, width=8, height=6):
    return pb.Image(sensor_id="head_image", encoding=encoding, width=width, height=height, data=data)


def test_jpeg_and_png_decode_to_the_same_rgb_as_raw():
    rgb = np.zeros((6, 8, 3), np.uint8)
    rgb[:, :4] = (200, 30, 10)
    png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tobytes()
    np.testing.assert_array_equal(_rgb(sensor(pb.PNG, png), "head_image"), rgb)
    np.testing.assert_array_equal(_rgb(sensor(pb.RAW_RGB, rgb.tobytes()), "head_image"), rgb)
    jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    assert np.abs(_rgb(sensor(pb.JPEG, jpeg), "head_image").astype(int) - rgb).mean() < 10  # lossy


@pytest.mark.parametrize("image", [
    sensor(pb.JPEG, b"not a jpeg"),
    sensor(pb.PNG, cv2.imencode(".png", np.zeros((4, 4, 3), np.uint8))[1].tobytes()),  # size mismatch
    sensor(pb.RAW_RGB, b"\0" * 10),
    sensor(pb.IMAGE_ENCODING_UNSPECIFIED, b""),
    None,
])
def test_invalid_images_are_rejected(image):
    with pytest.raises(ValueError, match="invalid RGB sensor"):
        _rgb(image, "head_image")
