from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

# We MUST avoid top-level 'import ray' or 'import jax' to prevent circular imports.
from ray.experimental.rdt.tensor_transport_manager import (
    CommunicatorMetadata,
    TensorTransportManager,
    TensorTransportMetadata,
)

if TYPE_CHECKING:
    import jax

    import ray


def _get_or_create_sharding(
    device_ids: Optional[List[int]],
    mesh_shape_dict: Dict[str, int],
    axis_names: Sequence[str],
    partition_spec_tuple: Tuple,
):
    import jax

    # Resolve physical devices.
    if device_ids is None:
        # Ground truth: use actual hardware devices for our own local sharding.
        devices = list(jax.local_devices())
    else:
        # Mapping: use global IDs provided by Ray for remote shardings.
        pool = {d.id: d for d in jax.devices()}
        devices = [pool.get(gid) for gid in device_ids]
        if any(d is None for d in devices):
            devices = list(jax.local_devices())

    # Use jax.make_mesh for stable, high-level mesh construction.
    mesh = jax.make_mesh(
        axis_shapes=tuple(mesh_shape_dict[name] for name in axis_names),
        axis_names=tuple(axis_names),
        devices=devices,
        axis_types=(jax.sharding.AxisType.Explicit,) * len(axis_names),
    )

    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(*partition_spec_tuple)
    )
    return sharding


@dataclass
class JaxCommunicatorMetadata(CommunicatorMetadata):
    source_device_ids: Optional[List[int]] = None
    dst_device_ids: Optional[List[int]] = None


@dataclass
class JaxTransportMetadata(TensorTransportMetadata):
    # Only essential metadata for logical sharding layout.
    mesh_shape: Optional[Dict[str, int]] = None
    mesh_axis_names: Optional[Sequence[str]] = None
    partition_spec: Optional[Tuple] = None
    obj_id: Optional[str] = None


class JaxTransport(TensorTransportManager):
    def __init__(self):
        pass

    def tensor_transport_backend(self) -> str:
        return "JAX"

    @staticmethod
    def is_one_sided() -> bool:
        return False

    @staticmethod
    def can_abort_transport() -> bool:
        return False

    def actor_has_tensor_transport(self, actor: "ray.actor.ActorHandle") -> bool:
        from ray.experimental.collective import get_all_local_device_ids_from_actor

        try:
            device_ids = get_all_local_device_ids_from_actor(actor)
            return len(device_ids) > 0
        except (KeyError, ValueError, RuntimeError):
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
        tensor_meta = [(t.shape, t.dtype) for t in gpu_object]

        # Convert JAX objects to plain Python types for serialization safety.
        meta = JaxTransportMetadata(
            tensor_meta=tensor_meta,
            tensor_device=str(mesh.devices.flatten()[0].device_kind),
            mesh_shape=dict(mesh.shape),
            mesh_axis_names=list(mesh.axis_names),
            partition_spec=tuple(sharding.spec),
            obj_id=obj_id,
        )
        return meta

    def get_communicator_metadata(
        self,
        src_actor: "ray.actor.ActorHandle",
        dst_actor: "ray.actor.ActorHandle",
        backend: Optional[str] = None,
    ) -> JaxCommunicatorMetadata:
        from ray.experimental.collective import get_all_local_device_ids_from_actor

        return JaxCommunicatorMetadata(
            source_device_ids=get_all_local_device_ids_from_actor(src_actor),
            dst_device_ids=get_all_local_device_ids_from_actor(dst_actor),
        )

    def recv_multiple_tensors(
        self,
        obj_id: str,
        tensor_transport_metadata: JaxTransportMetadata,
        communicator_metadata: JaxCommunicatorMetadata,
        target_buffers: Optional[List["jax.Array"]] = None,
    ) -> List["jax.Array"]:
        import jax

        tensors = []
        meta = tensor_transport_metadata
        comm = communicator_metadata

        if meta.tensor_meta:
            # 1. Reconstruct shardings for BOTH sides.
            # We need the source sharding to create the "Ghost" handles.
            src_sharding = _get_or_create_sharding(
                comm.source_device_ids,
                meta.mesh_shape,
                meta.mesh_axis_names,
                tuple(meta.partition_spec),
            )
            local_sharding = _get_or_create_sharding(
                None,  # Local hardware ground truth.
                meta.mesh_shape,
                meta.mesh_axis_names,
                tuple(meta.partition_spec),
            )

            # 2. Create "Ghost Arrays" (Real objects, 0 local data).
            # These handles tell JAX: "I expect data for this shape on the sender's chips."
            ghost_arrays = []
            for shape, dtype in meta.tensor_meta:
                arr_remote = jax.make_array_from_single_device_arrays(
                    shape=shape,
                    sharding=src_sharding,
                    arrays=[],
                    dtype=dtype,
                )
                ghost_arrays.append(arr_remote)

            # 3. Trigger the ICI pull.
            # JAX sees the ghost arrays and pulls the data to our local chips.
            jax.block_until_ready(ghost_arrays)
            tensors = jax.device_put(ghost_arrays, local_sharding)

            # Wait for JAX to finalize the linkage to incoming ICI data.
            jax.block_until_ready(tensors)

        return tensors

    def send_multiple_tensors(
        self,
        tensors: List["jax.Array"],
        tensor_transport_metadata: JaxTransportMetadata,
        communicator_metadata: JaxCommunicatorMetadata,
    ):
        import jax

        if not tensors:
            return

        meta = tensor_transport_metadata
        comm = communicator_metadata

        # Reconstruct receiver's sharding using provided IDs.
        dst_sharding = _get_or_create_sharding(
            comm.dst_device_ids,
            meta.mesh_shape,
            meta.mesh_axis_names,
            tuple(meta.partition_spec),
        )

        # 1. Initiate P2P push coordination.
        proxies = jax.device_put(tensors, dst_sharding)

        # 2. Dispatch to hardware.
        jax.block_until_ready(proxies)

    def garbage_collect(self, obj_id, tensor_transport_meta, tensors):
        pass

    def abort_transport(self, obj_id, communicator_metadata):
        pass
