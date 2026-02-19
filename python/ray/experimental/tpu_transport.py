from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

# We MUST avoid top-level 'import ray' to prevent circular imports.
from ray.experimental.rdt.tensor_transport_manager import (
    CommunicatorMetadata,
    TensorTransportManager,
    TensorTransportMetadata,
)
from ray.util.annotations import DeveloperAPI

if TYPE_CHECKING:
    import jax

    import ray


# Global caches for metadata. Perfect reuse is MANDATORY to avoid std::bad_alloc.
_MESH_CACHE = {}
_SHARDING_CACHE = {}

# Small deques to protect background threads without exhausting descriptor memory.
_RETAINED_RECV_OBJECTS = deque(maxlen=30)
_RETAINED_SEND_OBJECTS = deque(maxlen=30)


def _get_or_create_sharding(
    device_ids,
    mesh_shape_dict,
    axis_names,
    partition_spec_tuple,
    mesh_axis_types=None,
    force_direct_mesh=False,
):
    import jax

    from ray.util.tpu import get_local_device_from_global_id

    # Use the exact axis names provided.
    norm_axis_names = tuple(axis_names)
    norm_spec_tuple = partition_spec_tuple

    # Strict AxisType matching: ensures JAX reuses internal communication descriptors.
    norm_axis_types = None
    if mesh_axis_types:
        norm_axis_types = tuple(
            jax.sharding.AxisType[a] if isinstance(a, str) else a
            for a in mesh_axis_types
        )
    else:
        # Default to Explicit if not provided, matching typical NamedSharding.
        norm_axis_types = tuple(jax.sharding.AxisType.Explicit for _ in axis_names)

    mesh_key = (
        tuple(sorted(device_ids)),
        tuple(sorted(mesh_shape_dict.items())),
        norm_axis_names,
        norm_axis_types,
        force_direct_mesh,
    )

    if mesh_key not in _MESH_CACHE:
        devices = [get_local_device_from_global_id(gid) for gid in device_ids]
        if any(d is None for d in devices):
            devices = list(jax.local_devices())
            if not devices:
                raise RuntimeError(f"Local JAX devices not found for IDs: {device_ids}")

        shape = tuple(mesh_shape_dict[name] for name in axis_names)

        if force_direct_mesh:
            # Reconstruct Sender Mesh using exact device order.
            mesh = jax.sharding.Mesh(
                devices=np.array(devices).reshape(shape),
                axis_names=norm_axis_names,
                axis_types=norm_axis_types,
            )
        else:
            # Local Mesh for topology-aware performance.
            mesh = jax.make_mesh(
                shape,
                norm_axis_names,
                devices=devices,
                axis_types=norm_axis_types,
            )
        _MESH_CACHE[mesh_key] = mesh

    mesh = _MESH_CACHE[mesh_key]

    sharding_key = (mesh_key, norm_spec_tuple)
    if sharding_key not in _SHARDING_CACHE:
        partition_spec = jax.sharding.PartitionSpec(*norm_spec_tuple)
        sharding = jax.sharding.NamedSharding(mesh, partition_spec)
        _SHARDING_CACHE[sharding_key] = sharding

    return _SHARDING_CACHE[sharding_key]


@DeveloperAPI
@dataclass
class JaxCommunicatorMetadata(CommunicatorMetadata):
    source_device_ids: Optional[List[int]] = None
    dst_device_ids: Optional[List[int]] = None


@DeveloperAPI
@dataclass
class JaxTransportMetadata(TensorTransportMetadata):
    sharding_type: Optional[str] = None
    mesh_shape: Optional[Dict[str, int]] = None
    mesh_axis_names: Optional[Sequence[str]] = None
    mesh_axis_types: Optional[Tuple] = None
    partition_spec: Optional[Tuple] = None
    mesh_devices_ids: Optional[List[int]] = None
    obj_id: Optional[str] = None


@DeveloperAPI
class JaxTransport(TensorTransportManager):
    def __init__(self):
        return

    def tensor_transport_backend(self) -> str:
        return "TPU_JAX"

    @staticmethod
    def is_one_sided() -> bool:
        return False

    @staticmethod
    def can_abort_transport() -> bool:
        return False

    def actor_has_tensor_transport(self, actor: "ray.actor.ActorHandle") -> bool:
        from ray.experimental.collective.collective import (
            get_all_local_device_ids_from_actor,
        )

        try:
            device_ids = get_all_local_device_ids_from_actor(actor)
            return len(device_ids) > 0
        except (KeyError, ValueError):
            return False

    def extract_tensor_transport_metadata(
        self,
        obj_id: str,
        gpu_object: List["jax.Array"],
    ) -> JaxTransportMetadata:
        if not gpu_object:
            return JaxTransportMetadata(tensor_meta=[])

        sharding = gpu_object[0].sharding
        mesh = sharding.mesh
        mesh_devices_ids = [d.id for d in mesh.devices.flatten()]
        tensor_meta = [(t.shape, t.dtype) for t in gpu_object]

        return JaxTransportMetadata(
            tensor_meta=tensor_meta,
            tensor_device=mesh.devices.flatten()[0].device_kind,
            sharding_type="named",
            mesh_shape=mesh.shape,
            mesh_axis_names=mesh.axis_names,
            mesh_axis_types=mesh.axis_types,
            partition_spec=sharding.spec,
            mesh_devices_ids=mesh_devices_ids,
            obj_id=obj_id,
        )

    def get_communicator_metadata(
        self,
        src_actor: "ray.actor.ActorHandle",
        dst_actor: "ray.actor.ActorHandle",
        backend: Optional[str] = None,
    ) -> JaxCommunicatorMetadata:
        from ray.experimental.collective.collective import (
            get_all_local_device_ids_from_actor,
        )

        try:
            return JaxCommunicatorMetadata(
                source_device_ids=get_all_local_device_ids_from_actor(src_actor),
                dst_device_ids=get_all_local_device_ids_from_actor(dst_actor),
            )
        except KeyError as e:
            raise RuntimeError(f"TPU devices cache not primed for actor {e}") from e

    def recv_multiple_tensors(
        self,
        obj_id: str,
        tensor_transport_metadata: JaxTransportMetadata,
        communicator_metadata: JaxCommunicatorMetadata,
        target_buffers: Optional[List["jax.Array"]] = None,
    ) -> List["jax.Array"]:
        import jax

        tensors = []
        if tensor_transport_metadata.tensor_meta:
            src_sharding = _get_or_create_sharding(
                tensor_transport_metadata.mesh_devices_ids,
                tensor_transport_metadata.mesh_shape,
                tensor_transport_metadata.mesh_axis_names,
                tuple(tensor_transport_metadata.partition_spec),
                mesh_axis_types=tensor_transport_metadata.mesh_axis_types,
                force_direct_mesh=True,
            )
            _RETAINED_RECV_OBJECTS.append(src_sharding)

            local_sharding = _get_or_create_sharding(
                [d.id for d in jax.local_devices()],
                tensor_transport_metadata.mesh_shape,
                tensor_transport_metadata.mesh_axis_names,
                tuple(tensor_transport_metadata.partition_spec),
                mesh_axis_types=tensor_transport_metadata.mesh_axis_types,
                force_direct_mesh=False,
            )
            _RETAINED_RECV_OBJECTS.append(local_sharding)

            ghost_arrays = []
            for shape, dtype in tensor_transport_metadata.tensor_meta:
                source_arr = jax.make_array_from_single_device_arrays(
                    shape=shape,
                    sharding=src_sharding,
                    arrays=[],
                    dtype=dtype,
                )
                _RETAINED_RECV_OBJECTS.append(source_arr)
                ghost_arrays.append(source_arr)

            # Batched transfer for coordination stability.
            tensors = jax.device_put(ghost_arrays, local_sharding)

            jax.block_until_ready(tensors)

            if isinstance(tensors, (list, tuple)):
                for t in tensors:
                    _RETAINED_RECV_OBJECTS.append(t)
            else:
                _RETAINED_RECV_OBJECTS.append(tensors)

        return list(tensors) if isinstance(tensors, (list, tuple)) else [tensors]

    def send_multiple_tensors(
        self,
        tensors: List["jax.Array"],
        tensor_transport_metadata: JaxTransportMetadata,
        communicator_metadata: JaxCommunicatorMetadata,
    ):
        import jax

        if not tensors:
            return

        dst_sharding = _get_or_create_sharding(
            communicator_metadata.dst_device_ids,
            tensor_transport_metadata.mesh_shape,
            tensor_transport_metadata.mesh_axis_names,
            tuple(tensor_transport_metadata.partition_spec),
            mesh_axis_types=tensor_transport_metadata.mesh_axis_types,
            force_direct_mesh=False,
        )
        _RETAINED_SEND_OBJECTS.append(dst_sharding)

        for tensor in tensors:
            _RETAINED_SEND_OBJECTS.append(tensor)

        # Batched send for stability.
        proxies = jax.device_put(tensors, dst_sharding)

        jax.block_until_ready(proxies)

        if isinstance(proxies, (list, tuple)):
            for p in proxies:
                _RETAINED_SEND_OBJECTS.append(p)
        else:
            _RETAINED_SEND_OBJECTS.append(proxies)

    def garbage_collect(self, obj_id, tensor_transport_meta, tensors):
        pass

    def abort_transport(self, obj_id, communicator_metadata):
        pass
