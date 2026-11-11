"""Colosseum policy server adapter."""

from importlib import import_module

# Model workers use the package in upstream SDK environments (e.g. TensorFlow
# with an older protobuf). Import protocol dependencies only for SDK users.
def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    module = ".server" if name in {"Policy", "PolicyServer"} else ".sdk"
    if name == "ColosseumPolicySDK":
        module = ".sync_sdk"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "AsyncColosseumPolicySDK",
    "ColosseumPolicySDK",
    "ImageFrame",
    "Observation",
    "ObservationState",
    "Policy",
    "PolicyServer",
    "RobotConfiguration",
    "SDKConnectionError",
    "StaleObservationError",
]
