from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import ray
from ray.experimental.gpu_object_manager.tensor_transport_manager import (
    CommunicatorMetadata,
    TensorTransportManager,
    TensorTransportMetadata,
)
from ray.util.annotations import DeveloperAPI
from ray.util.tpu import (
    get_global_device_id_from_actor,
    get_local_device_from_global_id,
)

if TYPE_CHECKING:
    import jax


@DeveloperAPI
@dataclass
class JaxCommunicatorMetadata(CommunicatorMetadata):
    source_device_id: Optional[int] = None
    dst_device_id: Optional[int] = None


@DeveloperAPI
@dataclass
class JaxTransportMetadata(TensorTransportMetadata):
    pass


@DeveloperAPI
class JaxTransport(TensorTransportManager):
    def __init__(self):
        pass

    @property
    def tensor_transport_backend(self) -> str:
        return "tpu_jax"

    @staticmethod
    def is_one_sided() -> bool:
        return False

    @staticmethod
    def can_abort_transport() -> bool:
        return False

    def actor_has_tensor_transport(self, actor: "ray.actor.ActorHandle") -> bool:
        # TODO(songsunny): check if the actor has TPU resources.
        return True

    def extract_tensor_transport_metadata(
        self,
        obj_id: str,
        gpu_object: List["jax.Array"],
    ) -> JaxTransportMetadata:

        tensor_meta = []
        device = None
        if gpu_object:
            device = list(gpu_object[0].devices())[0]
            for t in gpu_object:
                if list(t.devices())[0] != device:
                    raise ValueError(
                        "All tensors in an RDT object must be on the same device."
                    )
                tensor_meta.append((t.shape, t.dtype))

        return JaxTransportMetadata(
            tensor_meta=tensor_meta,
            tensor_device=device.device_kind if device else None,
        )

    def get_communicator_metadata(
        self,
        src_actor: "ray.actor.ActorHandle",
        dst_actor: "ray.actor.ActorHandle",
        backend: Optional[str] = None,
    ) -> JaxCommunicatorMetadata:

        communicator_metadata = JaxCommunicatorMetadata(
            source_device_id=get_global_device_id_from_actor(src_actor),
            dst_device_id=get_global_device_id_from_actor(dst_actor),
        )
        return communicator_metadata

    def recv_multiple_tensors(
        self,
        obj_id: str,
        tensor_transport_metadata: TensorTransportMetadata,
        communicator_metadata: CommunicatorMetadata,
    ) -> List["jax.Array"]:

        assert isinstance(tensor_transport_metadata, JaxTransportMetadata)
        assert isinstance(communicator_metadata, JaxCommunicatorMetadata)

        import jax

        tensors = []
        if tensor_transport_metadata.tensor_meta:
            receiver_device = get_local_device_from_global_id(
                communicator_metadata.dst_device_id
            )
            source_device = get_local_device_from_global_id(
                communicator_metadata.source_device_id
            )
            sharding = jax.sharding.SingleDeviceSharding(source_device)
            for shape, dtype in tensor_transport_metadata.tensor_meta:
                arr = jax.make_array_from_single_device_arrays(
                    shape=shape, sharding=sharding, arrays=[], dtype=dtype
                )
                jax.device_put(arr, receiver_device)
                tensors.append(arr)

        return tensors

    def send_multiple_tensors(
        self,
        tensors: List["jax.Array"],
        tensor_transport_metadata: TensorTransportMetadata,
        communicator_metadata: CommunicatorMetadata,
    ):

        assert isinstance(communicator_metadata, JaxCommunicatorMetadata)

        import jax

        dst_device = get_local_device_from_global_id(
            communicator_metadata.dst_device_id
        )

        for tensor in tensors:
            jax.device_put(tensor, dst_device)

    def garbage_collect(
        self, obj_id: str, tensor_transport_meta: TensorTransportMetadata
    ):
        pass

    def abort_transport(
        self,
        obj_id: str,
        communicator_metadata: CommunicatorMetadata,
    ):
        raise NotImplementedError(
            "JAX transport does not support abort_transport for now."
        )
