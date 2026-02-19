import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import jax
import numpy as np

import ray
from ray.experimental.rdt.tensor_transport_manager import (
    CommunicatorMetadata,
    TensorTransportManager,
    TensorTransportMetadata,
)
from ray.util.annotations import DeveloperAPI
from ray.util.tpu import (
    get_all_local_device_ids_from_actor,
    get_local_device_from_global_id,
)

if TYPE_CHECKING:
    import jax


@DeveloperAPI
@dataclass
class JaxCommunicatorMetadata(CommunicatorMetadata):
    source_device_ids: Optional[List[int]] = None
    dst_device_ids: Optional[List[int]] = None


@DeveloperAPI
@dataclass
class JaxTransportMetadata(TensorTransportMetadata):
    sharding_type: Optional[str] = None
    mesh_shape: Optional[tuple] = None
    mesh_axis_names: Optional[tuple] = None
    mesh_axis_types: Optional[tuple] = None
    partition_spec: Optional[tuple] = None
    mesh_devices_ids: Optional[List[int]] = None


@DeveloperAPI
class JaxTransport(TensorTransportManager):
    def __init__(self):
        print("!!! JaxTransport object is being created !!!", flush=True)
        pass

    @property
    def tensor_transport_backend(self) -> str:
        return "TPU_JAX"

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
        print(
            "!!! Calling extract_tensor_transport_metadata with gpu_object: "
            f"{len(gpu_object)} !!!"
        )
        tensor_meta = []
        if not gpu_object:
            return JaxTransportMetadata(tensor_meta=[])

        sharding = gpu_object[0].sharding
        print(f"!!! Sharding is: {sharding} !!!")

        if isinstance(sharding, jax.sharding.NamedSharding):
            mesh = sharding.mesh
            mesh_devices_ids = [d.id for d in mesh.devices.flatten()]
            for t in gpu_object:
                if t.sharding != sharding:
                    raise ValueError(
                        "All tensors in an RDT object must have the same sharding."
                    )
                tensor_meta.append((t.shape, t.dtype))

            return JaxTransportMetadata(
                tensor_meta=tensor_meta,
                tensor_device=mesh.devices.flatten()[0].device_kind,
                sharding_type="named",
                mesh_shape=mesh.shape,
                mesh_axis_names=mesh.axis_names,
                mesh_axis_types=mesh.axis_types,
                partition_spec=sharding.spec,
                mesh_devices_ids=mesh_devices_ids,
            )
        elif isinstance(sharding, jax.sharding.SingleDeviceSharding):
            device = list(gpu_object[0].devices())[0]
            print(f"!!! gpu_object device is: {device} !!!")
            for t in gpu_object:
                if list(t.devices())[0] != device:
                    raise ValueError(
                        "All tensors in an RDT object must be on the same device."
                    )
                tensor_meta.append((t.shape, t.dtype))

            print(f"!!! Extracted tensor_meta: {tensor_meta} !!!", flush=True)
            print(f"!!! Extracted device: {device} !!!", flush=True)
            return JaxTransportMetadata(
                tensor_meta=tensor_meta,
                tensor_device=device.device_kind if device else None,
                sharding_type="single",
            )
        else:
            # Default to existing behavior if sharding is not recognized
            device = list(gpu_object[0].devices())[0]
            print(f"!!! gpu_object device is: {device} !!!")
            for t in gpu_object:
                if list(t.devices())[0] != device:
                    raise ValueError(
                        "All tensors in an RDT object must be on the same device."
                    )
                tensor_meta.append((t.shape, t.dtype))

            print(f"!!! Extracted tensor_meta: {tensor_meta} !!!", flush=True)
            print(f"!!! Extracted device: {device} !!!", flush=True)
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
        print(
            f"!!! Called get_communicator_metadata for src_actor {src_actor} and dst_actor {dst_actor} !!!"
        )
        communicator_metadata = JaxCommunicatorMetadata(
            source_device_ids=get_all_local_device_ids_from_actor(src_actor),
            dst_device_ids=get_all_local_device_ids_from_actor(dst_actor),
        )
        print(f"!!! JaxCommunicatorMetadata created: {communicator_metadata} !!!")
        return communicator_metadata

    def recv_multiple_tensors(
        self,
        obj_id: str,
        tensor_transport_metadata: TensorTransportMetadata,
        communicator_metadata: CommunicatorMetadata,
        target_buffers: Optional[List[jax.Array]] = None,
    ) -> List["jax.Array"]:
        print(
            "!!! recv_multiple_tensors: src="
            f"{communicator_metadata.source_device_ids}, dst="
            f"{communicator_metadata.dst_device_ids} !!!"
        )

        print(f"!!! tensor_transport_metadata is {tensor_transport_metadata} !!!")

        assert isinstance(tensor_transport_metadata, JaxTransportMetadata)
        assert isinstance(communicator_metadata, JaxCommunicatorMetadata)

        import jax

        tensors = []
        if tensor_transport_metadata.tensor_meta:
            sharding = None  # Sharding from the source devices
            if tensor_transport_metadata.sharding_type == "named":
                # Create the sharding of the sender to create the empty array
                # with the correct sharding info.
                source_devices = [
                    get_local_device_from_global_id(gid)
                    for gid in communicator_metadata.source_device_ids
                ]
                mesh = jax.make_mesh(
                    axis_shapes=tuple(tensor_transport_metadata.mesh_shape.values()),
                    axis_names=tensor_transport_metadata.mesh_axis_names,
                    axis_types=tensor_transport_metadata.mesh_axis_types,
                    devices=np.asarray(source_devices),
                )
                partition_spec = jax.sharding.PartitionSpec(
                    *tensor_transport_metadata.partition_spec
                )
                sharding = jax.sharding.NamedSharding(mesh, partition_spec)
                print(
                    f"!!! Reconstructed receiver's source NamedSharding: {sharding} !!!"
                )
                print(
                    f"!!! Mesh devices in recv_multiple_tensors: {mesh.devices} and lenth is {len(mesh.devices)}!!!"
                )
                print(f"!!! Mesh axis names: {mesh.axis_names} !!!")
                print(f"!!! PartitionSpec: {partition_spec} !!!")
            else:  # single or None
                source_device = get_local_device_from_global_id(
                    communicator_metadata.source_device_ids[0]
                )
                sharding = jax.sharding.SingleDeviceSharding(source_device)
                print(f"!!! Reconstructed SingleDeviceSharding: {sharding} !!!")

            for shape, dtype in tensor_transport_metadata.tensor_meta:
                # This API is supposed to be for single device sharding, but it can be
                # used to create an empty container with the right sharding info,
                # which is what we need here before the actual data transfer.
                arr = jax.make_array_from_single_device_arrays(
                    shape=shape,
                    sharding=sharding,
                    arrays=[],
                    dtype=dtype,
                )
                print(f"!!! array before receiving is {len(arr)} !!!")
                if tensor_transport_metadata.sharding_type == "named":
                    receiver_sharding = None
                    dest_devices = [
                        get_local_device_from_global_id(gid)
                        for gid in communicator_metadata.dst_device_ids
                    ]
                    mesh = jax.make_mesh(
                        axis_shapes=tuple(
                            tensor_transport_metadata.mesh_shape.values()
                        ),
                        axis_names=tensor_transport_metadata.mesh_axis_names,
                        axis_types=tensor_transport_metadata.mesh_axis_types,
                        devices=np.asarray(dest_devices),
                    )
                    partition_spec = jax.sharding.PartitionSpec(
                        *tensor_transport_metadata.partition_spec
                    )
                    receiver_sharding = jax.sharding.NamedSharding(mesh, partition_spec)
                    print(f"!!! receiver_sharding is {receiver_sharding} !!!")
                    arr.block_until_ready()
                    t = time.time()
                    arr = jax.device_put(arr, receiver_sharding)
                    arr.block_until_ready()
                    device_put_time = time.time() - t
                    print(f"!!! device_put_time is {device_put_time} !!!")
                else:
                    receiver_device = get_local_device_from_global_id(
                        communicator_metadata.dst_device_ids[0]
                    )
                    arr = jax.device_put(arr, receiver_device)
                print(f"!!! array after receiving is {len(arr)} !!!")
                tensors.append(arr)

        print(f"!!! tensors result is {len(tensors)} !!!")
        return tensors

    def send_multiple_tensors(
        self,
        tensors: List["jax.Array"],
        tensor_transport_metadata: TensorTransportMetadata,
        communicator_metadata: CommunicatorMetadata,
    ):
        print(
            "!!! send_multiple_tensors: src="
            f"{communicator_metadata.source_device_ids}, dst="
            f"{communicator_metadata.dst_device_ids} !!!"
        )

        assert isinstance(communicator_metadata, JaxCommunicatorMetadata)
        assert isinstance(tensor_transport_metadata, JaxTransportMetadata)

        import jax

        if tensor_transport_metadata.sharding_type == "named":
            # Reconstruct the destination mesh and sharding.
            dest_devices = [
                get_local_device_from_global_id(gid)
                for gid in communicator_metadata.dst_device_ids
            ]
            mesh = jax.make_mesh(
                axis_shapes=tuple(tensor_transport_metadata.mesh_shape.values()),
                axis_names=tensor_transport_metadata.mesh_axis_names,
                axis_types=tensor_transport_metadata.mesh_axis_types,
                devices=np.asarray(dest_devices),
            )
            partition_spec = jax.sharding.PartitionSpec(
                *tensor_transport_metadata.partition_spec
            )
            sharding = jax.sharding.NamedSharding(mesh, partition_spec)

            print(f"!!! Reconstructed sender NamedSharding: {sharding} !!!")
            print(f"!!! Mesh devices: {mesh.devices} !!!")
            print(f"!!! Mesh axis names: {mesh.axis_names} !!!")
            print(f"!!! PartitionSpec: {partition_spec} !!!")
            for tensor in tensors:
                print(f"!!! Sending tensor {len(tensor)} with sharding {sharding} !!!")
                jax.device_put(tensor, sharding)
        else:
            dst_device = get_local_device_from_global_id(
                communicator_metadata.dst_device_ids[0]
            )
            for tensor in tensors:
                print(
                    f"!!! Sending tensor {len(tensor)} to dst_device {dst_device} !!!"
                )
                jax.device_put(tensor, dst_device)

        print("!!! Done send_multiple_tensors !!!")

    def garbage_collect(
        self,
        obj_id: str,
        tensor_transport_meta: TensorTransportMetadata,
        tensors: List["jax.Array"],
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
