"""
This module is intended for implementing internal serializers for some
site packages.
"""

import sys
from dataclasses import asdict, dataclass

from ray.util.annotations import DeveloperAPI


@dataclass
class JAXTPUTransportMetadata:
    # Required to allocate the destination buffer
    shape: tuple
    dtype: str

    # Required for Actor B to call device_put on the CORRECT local chip
    receiver_global_device_id: int
    # Required for Actor B to call device_put on the CORRECT local chip
    sender_global_device_id: int

    # Required for Ray to match the 'Ready-to-Receive' signal to the right task
    transfer_uuid: str


@DeveloperAPI
def register_starlette_serializer(serialization_context):
    try:
        import starlette.datastructures
    except ImportError:
        return

    # Starlette's app.state object is not serializable
    # because it overrides __getattr__
    serialization_context._register_cloudpickle_serializer(
        starlette.datastructures.State,
        custom_serializer=lambda s: s._state,
        custom_deserializer=lambda s: starlette.datastructures.State(s),
    )


@DeveloperAPI
def register_jax_serializer(serialization_context):
    try:
        # Check if jax is available.
        import jax  # noqa: F401
    except ImportError:
        return

    serialization_context._register_cloudpickle_serializer(
        JAXTPUTransportMetadata,
        custom_serializer=asdict,
        custom_deserializer=lambda d: JAXTPUTransportMetadata(**d),
    )


@DeveloperAPI
def apply(serialization_context):
    from ray._common.pydantic_compat import register_pydantic_serializers

    register_pydantic_serializers(serialization_context)
    register_starlette_serializer(serialization_context)
    register_jax_serializer(serialization_context)

    if sys.platform != "win32":
        from ray._private.arrow_serialization import (
            _register_custom_datasets_serializers,
        )

        _register_custom_datasets_serializers(serialization_context)
