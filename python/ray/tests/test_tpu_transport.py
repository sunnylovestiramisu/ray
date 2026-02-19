import os
import sys

import pytest

import ray

try:
    import jax
    import jax.numpy as jnp

    from ray.experimental.gpu_object_manager.util import register_tensor_transport
    from ray.experimental.tpu_transport import JaxTransport
    from ray.util.tpu import get_tpu_devices
except ImportError:
    jax = None

# Register the custom transport if jax is available.
if jax:
    register_tensor_transport("JAX", ["tpu"], JaxTransport)

# Skip all tests in this file if no TPUs are available, or jax is not installed.
pytestmark = pytest.mark.skipif(
    not jax or len(get_tpu_devices()) == 0,
    reason="TPU not available or JAX not installed.",
)


@ray.remote(num_tpus=1)
class TpuActor:
    def __init__(self):
        # This will fail if no TPU devices are available.
        self.device = jax.devices("tpu")[0]

    def get_device_id(self):
        return self.device.id

    @ray.method(tensor_transport="JAX")
    def create_data(self, shape=(10, 10), dtype=jnp.float32):
        return jnp.ones(shape, dtype=dtype, device=self.device)

    def receive_data(self, data):
        # Check if the data is on the correct device.
        if data.device() != self.device:
            return False
        # Check data integrity.
        return jnp.all(data == 1.0).item()


def test_tpu_transport(shutdown_only):
    # This env var is needed to work around a bug in the test framework.
    os.environ["RAY_testing_asio_delay_us"] = "0"
    ray.init(num_cpus=2, num_tpus=2)
    # The cloudpickle line is important because of how pytest pickles things in tests.
    from ray import cloudpickle

    cloudpickle.register_pickle_by_value(sys.modules[JaxTransport.__module__])

    actor1 = TpuActor.remote()
    actor2 = TpuActor.remote()

    # The create_data method is decorated with @ray.method(tensor_transport="JAX"),
    # so this will create a GPU object managed by the JaxTransport.
    data_on_actor1 = actor1.create_data.remote()

    # Passing the object ref to another actor will trigger the custom transport.
    result_ref = actor2.receive_data.remote(data_on_actor1)
    assert ray.get(result_ref)

    del os.environ["RAY_testing_asio_delay_us"]


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))
