from __future__ import annotations

import numpy as np

from colosseum_policy_server import ImageFrame, Observation


def test_observation_state_exposes_standard_policy_inputs():
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    joints = np.arange(7, dtype=np.float32)
    gripper = np.array([0.5], dtype=np.float32)
    observation = Observation(
        session_id="session-1",
        sequence=1,
        deadline_ms=1000,
        robot_time_ns=1,
        control_step=1,
        instruction="test",
        raw_state={"joint_position": joints, "gripper_position": gripper},
        images=(
            ImageFrame(
                image_id="head_image",
                encoding="RAW_RGB",
                capture_time_ns=1,
                width=3,
                height=2,
                data=rgb.tobytes(),
            ),
        ),
        _received_monotonic_ns=1,
    )

    np.testing.assert_array_equal(observation.state.joints, joints)
    np.testing.assert_array_equal(observation.state.gripper, gripper)
    np.testing.assert_array_equal(observation.state.head_image, rgb)
    assert observation.state.left_image is None
    assert observation.state.right_image is None


def test_rover_state_helpers():
    import numpy as np
    from colosseum_policy_server.sdk import ImageFrame, Observation
    front = ImageFrame("front_image", "RAW_RGB", 0, 2, 1, bytes(6))
    goal = ImageFrame("goal_image", "RAW_RGB", 0, 2, 1, bytes(6))
    observation = Observation(session_id="s", sequence=1, deadline_ms=0, robot_time_ns=0, control_step=0, instruction="go",
                              raw_state={"navigation_state": np.arange(9, dtype=np.float64), "goal_gps": np.array([1.0, 2.0])},
                              images=(front, goal), _received_monotonic_ns=0)
    assert observation.state.front_image.shape == (1, 2, 3) and observation.state.rear_image is None
    assert observation.state.goal_image.shape == (1, 2, 3)
    np.testing.assert_array_equal(observation.state.navigation, np.arange(9))
    np.testing.assert_array_equal(observation.state.goal_gps, [1.0, 2.0])
