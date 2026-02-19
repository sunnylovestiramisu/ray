import logging
import math
import os
import threading
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import ray
from ray._private.accelerators import TPUAcceleratorManager
from ray._private.accelerators.tpu import (
    VALID_TPU_TYPES,
    get_chips_per_host,
    get_num_chips_from_topology,
    reserve_tpu_slice,
)
from ray._private.client_mode_hook import client_mode_wrap
from ray.util.annotations import PublicAPI
from ray.util.placement_group import (
    PlacementGroup,
    placement_group,
    remove_placement_group,
)
from ray.util.tpu import (
    get_local_device_from_global_id,
)

if TYPE_CHECKING:
    import jax

logger = logging.getLogger(__name__)


@PublicAPI(stability="alpha")
def get_tpu_version_from_type(accelerator_type: str) -> str:
    """Extracts the version from the accelerator type.

    Args:
        accelerator_type: The full accelerator type string (e.g. "TPU-V6E").

    Returns:
        The version string (e.g. "v6e").

    Raises:
        ValueError: If the accelerator type is invalid.
    """
    accel_type_lower = accelerator_type.lower()

    if accel_type_lower.startswith("tpu-"):
        version = accel_type_lower.replace("tpu-", "")
    elif accel_type_lower.startswith("tpu"):
        version = accel_type_lower.replace("tpu", "v")
    else:
        version = accel_type_lower

    if version not in VALID_TPU_TYPES:
        raise ValueError(
            f"Invalid accelerator_type: {accelerator_type}. "
            f"Must be one of {list(VALID_TPU_TYPES)} or start with 'TPU-' followed by a valid type."
        )

    return version


@PublicAPI(stability="alpha")
def get_current_pod_name() -> Optional[str]:
    """
    Return the name of the TPU pod that the worker is a part of.

    Returns:
        The name of the TPU pod. Returns None if not part of a TPU pod.
    """
    tpu_name = TPUAcceleratorManager.get_current_node_tpu_name()
    if tpu_name == "":
        tpu_name = None
    return tpu_name


@PublicAPI(stability="alpha")
def get_current_pod_worker_count() -> Optional[int]:
    """
    Count the number of workers associated with the TPU pod that the worker belongs to.

    Returns:
        The total number of workers in the TPU pod. Returns None if the worker is not
        part of a TPU pod.
    """
    return TPUAcceleratorManager.get_num_workers_in_current_tpu_pod()


@PublicAPI(stability="alpha")
def get_num_tpu_chips_on_node() -> int:
    """
    Return the number of TPU chips on the node.
    Returns:
        The total number of chips on the TPU node. Returns 0 if none are found.
    """
    return TPUAcceleratorManager.get_current_node_num_accelerators()


@PublicAPI(stability="alpha")
def get_tpu_num_slices_for_workers(
    topology: str,
    accelerator_type: str,
    num_workers: int,
    resources_per_worker: Optional[Dict[str, float]] = None,
) -> int:
    """
    Calculates the number of slices needed to accommodate the specified number of workers.

    Args:
        topology: The TPU topology string.
        accelerator_type: The accelerator type string.
        num_workers: The desired number of workers.
        resources_per_worker: Optional dict of resources per worker.

    Returns:
        The number of slices required. Returns 1 if inputs are invalid or incomplete.
    """
    if not topology or not accelerator_type:
        return 1

    try:
        # Calculate how many workers fit in a single slice (num_slices=1)
        # given the topology and resources per worker.
        workers_per_slice, _ = get_tpu_worker_resources(
            topology=topology,
            accelerator_type=accelerator_type,
            resources_per_unit=resources_per_worker,
            num_slices=1,
        )

        if workers_per_slice == 0:
            return 1

        return max(1, math.ceil(num_workers / workers_per_slice))
    except Exception:
        # Fallback to 1 if calculation fails.
        return 1


@PublicAPI(stability="alpha")
def get_tpu_worker_resources(
    topology: str,
    accelerator_type: str,
    resources_per_unit: Optional[Dict[str, float]] = None,
    num_slices: int = 1,
) -> Tuple[int, Dict[str, float]]:
    """
    Calculates the number of workers and the resources required for each worker
    to run based on a TPU topology.

    Args:
        topology: The TPU topology string.
        accelerator_type: The accelerator string.
        resources_per_unit: Optional manual override for resources per unit. If
            unspecified, the number of TPU chips in a host is assumed.
        num_slices: The number of TPU slices.

    Returns:
        A tuple containing:
        - num_workers: Total workers required.
        - unit_resources: The resource dictionary for a single worker.
    """
    accelerator_version = get_tpu_version_from_type(accelerator_type)

    chips_per_host = get_chips_per_host(topology, accelerator_version)
    total_chips_per_slice = get_num_chips_from_topology(topology)

    total_chips_available = total_chips_per_slice * num_slices

    # Calculate the per-unit resources based on the TPU topology.
    final_resources = resources_per_unit.copy() if resources_per_unit else {}

    if "CPU" not in final_resources:
        final_resources["CPU"] = 1

    # If user didn't specify TPU, default to # of chips on 1 host.
    if "TPU" not in final_resources:
        final_resources["TPU"] = chips_per_host

    tpus_per_unit = final_resources["TPU"]

    # Validate TPU resource values.
    if tpus_per_unit <= 0:
        raise ValueError("TPU resources must be positive.")

    if total_chips_available % tpus_per_unit != 0:
        raise ValueError(
            f"Total chips ({total_chips_available}) not divisible by "
            f"TPUs requested per unit ({tpus_per_unit})."
        )

    if total_chips_per_slice % tpus_per_unit != 0:
        raise ValueError(
            f"The requested resources per bundle ({tpus_per_unit} TPU chips) do not "
            f"divide evenly into the chips available per slice ({total_chips_per_slice}). "
            "This configuration results in an uneven distribution of workers across slices, "
            "which is not supported."
        )

    num_workers = int(total_chips_available // tpus_per_unit)

    return num_workers, final_resources


@PublicAPI(stability="alpha")
def get_tpu_coordinator_env_vars(
    coordinator_address: str,
    num_slices: int,
    slice_id: int,
    coordinator_port: str = "8081",
) -> Dict[str, str]:
    """
    Returns the environment variables required for JAX multi-slice coordination.

    Args:
        coordinator_address: The IP address or hostname of the coordinator.
        num_slices: The total number of slices in the cluster.
        slice_id: The index of the current slice.
        coordinator_port: The port the coordinator is listening on.

    Returns:
        A dictionary mapping environment variable names to their values.
    """
    return {
        "MEGASCALE_COORDINATOR_ADDRESS": coordinator_address,
        "MEGASCALE_PORT": coordinator_port,
        "MEGASCALE_NUM_SLICES": str(num_slices),
        "MEGASCALE_SLICE_ID": str(slice_id),
    }


@PublicAPI(stability="alpha")
class SlicePlacementGroup:
    """
    A handle to a placement group reservation for a TPU slice.

    The following definitions are added for clarity:

    - Accelerator type: A string describing the accelerator type and version (e.g. TPU-V2, TPU-V6E).
    - Accelerator version: The accelerator generation only (e.g. v6e, v5p, v5litepod).
    - Pod type: The TPU accelerator version and the number of chips in a topology. (e.g. v6e-128, v5p-8).
    - Accelerator topology: The physical topology representing the structure (e.g. 2x2x2, 16x16).

        Args:
            topology: The TPU topology string (e.g. "2x2x2").
            accelerator_version: The TPU accelerator generation (e.g. "v6e", "v5p", "v4").
            resources_per_bundle: Optionally specify the resources to include in every worker bundle.
            strategy: PlacementGroup parameter. The strategy to create the placement group. Currently default to "SPREAD"

             - "PACK": Packs Bundles into as few nodes as possible.
             - "SPREAD": Places Bundles across distinct nodes as even as possible.
             - "STRICT_PACK": Packs Bundles into one node. The group is
               not allowed to span multiple nodes.
             - "STRICT_SPREAD": Packs Bundles across distinct nodes.

            lifetime: PlacementGroup parameter. Either `None`, which defaults to the placement group
                will fate share with its creator and will be deleted once its
                creator is dead, or "detached", which means the placement group
                will live as a global object independent of the creator.

            num_slices: Number of TPU slices in the SlicePlacementGroup. Defaults to 1 when unspecified.

        Examples:

        .. testcode:: python
            :skipif: True

            import ray
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
            from ray.util.tpu import SlicePlacementGroup

            slice_handle = SlicePlacementGroup(topology="4x4", accelerator_version="v6e")
            slice_pg = slice_handle.placement_group
            ray.get(slice_pg.ready(), timeout=10)

            @ray.remote(num_cpus=0, resources={'TPU': 4})
            def spmd_task(world, rank):
                print(f"Current TPU is rank {rank} of {world}")

            tasks = [
                spmd_task.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=slice_pg,
                    )
                ).remote(world=4, rank=i)
                for i in range(slice_handle.num_hosts)
            ]

    """

    def __init__(
        self,
        topology: str,
        accelerator_version: str,
        resources_per_bundle: Optional[Dict[str, float]] = None,
        # below are args related to PG
        strategy: str = "SPREAD",
        name: str = "",
        lifetime: Optional[str] = None,
        # default
        num_slices: int = 1,
    ):
        self._topology = topology.strip().lower()
        self._accelerator_version = accelerator_version.strip().lower()
        self._resources_per_bundle = resources_per_bundle or {}
        self._num_slices = num_slices

        # Calculate number of bundles and bundle resources for specified TPU topology.
        self._num_bundles, self._bundle_resources = get_tpu_worker_resources(
            topology=self._topology,
            accelerator_type=self._accelerator_version,
            resources_per_unit=resources_per_bundle,
            num_slices=self._num_slices,
        )

        self._chips_per_host = get_chips_per_host(
            self._topology, self._accelerator_version
        )

        total_chips = get_num_chips_from_topology(self._topology)
        hosts_per_slice = max(1, total_chips // self._chips_per_host)
        self._num_hosts = hosts_per_slice * self._num_slices

        self._head_pgs: List[PlacementGroup] = []
        self._bundle_label_selector: List[Dict[str, str]] = []
        self._validate_tpu_config()
        self._placement_group = None

        # Reserve a TPU slice of the provided accelerator version and topology.
        self._placement_group = self._reserve_slice(
            strategy,
            name,
            lifetime,
        )

    def _accelerator_version_check(self, accelerator_version: str):
        if accelerator_version not in VALID_TPU_TYPES:
            raise ValueError(
                f"Invalid accelerator version: {accelerator_version}. Must be one of: {VALID_TPU_TYPES}"
            )

    def _validate_tpu_config(self):
        # Should validate topology and generation values and return a
        # ValueError if invalid.
        self._accelerator_version_check(self.accelerator_version)
        if not TPUAcceleratorManager.is_valid_tpu_accelerator_topology(
            tpu_accelerator_version=self.accelerator_version,
            tpu_topology=self._topology,
        ):
            raise ValueError(
                f"Invalid accelerator topology: '{self._topology}' for "
                f"accelerator version: '{self.accelerator_version}'"
            )

    def _reserve_slice(
        self,
        strategy: str = "SPREAD",
        name: str = "",
        lifetime: Optional[str] = None,
    ) -> PlacementGroup:
        """Performs the two-step scheduling to reserve a TPU slice."""
        self._bundle_label_selector = []
        bundles = []
        bundles_per_slice = self._num_bundles // self._num_slices

        # Construct accelerator format for reserve_tpu_slice. e.g. From "v6e" to "TPU-V6E", "v5p" to "TPU-V5P".
        accelerator_type = "TPU-" + self.accelerator_version.upper()

        try:
            for _ in range(self.num_slices):
                reservation = reserve_tpu_slice(self._topology, accelerator_type)
                if not reservation:
                    raise RuntimeError(
                        f"Failed to reserve TPU slice. Requested {self.num_slices} "
                        f"slice(s) of topology '{self._topology}' with accelerator type "
                        f"'{accelerator_type}'. Ensure that sufficient TPU resources are "
                        "available in the cluster."
                    )

                # Store the head placement group for clean-up when un-reserving the slice.
                slice_name, head_pg = reservation
                self._head_pgs.append(head_pg)

                # Reserving a slice is done through constructing num_hosts bundles, each with a label selector for
                # the unique name of an available TPU slice.
                selector = {ray._raylet.RAY_NODE_TPU_SLICE_NAME_KEY: slice_name}
                self._bundle_label_selector.extend([selector] * bundles_per_slice)
                bundles += [
                    self._bundle_resources.copy() for _ in range(bundles_per_slice)
                ]

            pg = placement_group(
                bundles=bundles,
                strategy=strategy,
                name=name,
                lifetime=lifetime,
                bundle_label_selector=self._bundle_label_selector,
            )

            return pg
        except Exception:
            self.shutdown()
            raise

    @property
    def placement_group(self) -> PlacementGroup:
        """The underlying PlacementGroup object."""
        return self._placement_group

    @property
    def chips_per_host(self) -> int:
        """The number of chips per host for this TPU slice."""
        # This is the same value as resources per worker for TPU.
        return self._chips_per_host

    @property
    def num_hosts(self) -> int:
        """The total number of hosts in the SlicePlacementGroup."""
        return self._num_hosts

    @property
    def num_bundles(self) -> int:
        """The total number of bundles in the SlicePlacementGroup."""
        return self._num_bundles

    @property
    def topology(self) -> str:
        """The physical topology of the TPU slice."""
        return self._topology

    @property
    def accelerator_version(self) -> str:
        """The TPU accelerator type of the slice."""
        return self._accelerator_version

    @property
    def num_slices(self) -> int:
        """The number of TPU slices this SlicePlacementGroup spans."""
        return self._num_slices

    @property
    def head_placement_groups(self) -> List[PlacementGroup]:
        """The internal head PGs used to reserve the slices."""
        return self._head_pgs

    @property
    def bundle_label_selector(self) -> List[Dict[str, str]]:
        """The bundle label selector list for the worker PG."""
        return self._bundle_label_selector

    @property
    def bundle_resources(self) -> Dict[str, float]:
        """The resources that are assigned to each bundle."""
        return self._bundle_resources

    def shutdown(self):
        """Removes the worker placement group and all internal head PGs."""
        if self._placement_group:
            remove_placement_group(self._placement_group)
            self._placement_group = None
        for head_pg in self._head_pgs:
            remove_placement_group(head_pg)
        self._head_pgs = []


@PublicAPI(stability="alpha")
@client_mode_wrap
def slice_placement_group(
    topology: str,
    accelerator_version: str,
    resources_per_bundle: Optional[Dict[str, float]] = None,
    num_slices: int = 1,
    **kwargs,
) -> SlicePlacementGroup:
    """Asynchronously creates a PlacementGroup for a TPU slice.

    A slice placement group reserves num_slices TPU slice(s) and creates a placement
    group for scheduling tasks or actors.

    Args:
        topology: The desired TPU pod topology (e.g. "4x4", "2x8").
        accelerator_version: The TPU accelerator generation, (e.g. "v4", "v5p", "v6e").
        resources_per_bundle: Specify the number of resources to reserve per bundle.
            When unspecified, SlicePlacementGroup defaults to reserving 1 bundle per TPU host in
            a topology, with the bundle resources set to the number of TPU in a host.
            Ex: Specifying {"TPU": 1} for a 4x4 topology would result in 16 bundles, each with 1 TPU.
            If resources_per_bundle=None for the same topology, there would be 4 bundles with 4 TPU each.
        num_slices: The number of tpu slices within the placement group
        **kwargs: Additional arguments for the placement group, such as 'name', 'lifetime', or 'strategy'.

    Returns:
        The handle for the created SlicePlacementGroup.
    """

    return SlicePlacementGroup(
        topology=topology,
        accelerator_version=accelerator_version,
        resources_per_bundle=resources_per_bundle,
        num_slices=num_slices,
        **kwargs,
    )


@PublicAPI(stability="alpha")
def get_tpu_devices() -> List["jax.Device"]:
    """Returns a list of JAX TPU devices visible to the worker."""
    try:
        import jax

        return jax.devices("tpu")
    except (ImportError, RuntimeError):
        return []


@PublicAPI(stability="alpha")
def get_global_device_id_from_actor(actor: "ray.actor.ActorHandle") -> int:
    """
    Get the global JAX device ID of the TPU device assigned to the actor.

    This function executes a remote task on the specified actor to retrieve the
    ID of the first JAX TPU device available to that actor.

    Args:
        actor: The Ray actor handle to get the device ID from.

    Returns:
        The global JAX device ID.
    """

    @ray.remote(num_cpus=0)
    def get_id():
        try:
            import jax

            return jax.devices()[0].id
        except (ImportError, IndexError):
            return -1

    return ray.get(get_id.options(actor=actor).remote())


@PublicAPI(stability="alpha")
def get_local__from_global_id(global_id: int) -> Optional["jax.Device"]:
    """
    Get the local JAX device object from its global ID.

    This function is intended to be called on a worker to find the specific
    JAX device that corresponds to a global ID.

    Args:
        global_id: The global JAX device ID to look for.

    Returns:
        The `jax.Device` object if found, otherwise None.
    """
    try:
        import jax

        for device in jax.devices():
            if device.id == global_id:
                return device
        return None
    except ImportError:
        return None


# Global registry to prevent Python GC from releasing HBM mid-transfer
_PINNED_BUFFERS = {}
_TRANSPORT_CONTEXT = threading.local()


def is_tpu_transport_enabled():
    return getattr(_TRANSPORT_CONTEXT, "enabled", False)


def get_current_task_target_device_id():
    return getattr(_TRANSPORT_CONTEXT, "target_device_id", 0)


def register_send_buffer(transfer_uuid, array):
    """Step 1: Keep the reference alive for the ICI hardware stream."""
    _PINNED_BUFFERS[transfer_uuid] = array


def release_send_buffer(transfer_uuid):
    """Step 5: Hardware ACK received. Safe to let GC reclaim HBM."""
    if transfer_uuid in _PINNED_BUFFERS:
        del _PINNED_BUFFERS[transfer_uuid]
        logger.debug(f"Released TPU buffer for transfer {transfer_uuid}")


@PublicAPI(stability="alpha")
def get_jax_distributed_metadata() -> Optional[str]:
    """Gets JAX distributed metadata from environment variables."""
    return os.environ.get("MEGASCALE_COORDINATOR_ADDRESS")


@PublicAPI(stability="alpha")
def initialize_jax_distributed(num_processes, process_id, coordinator_address=None):
    """
    Sets up the JAX Control Plane.
    """
    try:
        import jax

        if jax.distributed.global_state is not None:
            return  # Safety check: JAX can only be initialized once per process

        # If the user didn't provide an address, we perform the discovery
        if coordinator_address is None:
            coordinator_address = get_jax_distributed_metadata()

        # The actual JAX call that starts the distributed state machine
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id,
            # initialization_timeout is key for large TPU slices
            initialization_timeout=300,
        )
        print(f"JAX Distributed Initialized: Process {process_id}/{num_processes}")
    except ImportError:
        logger.warning(
            "Could not initialize JAX distributed because jax is not installed."
        )


@PublicAPI(stability="alpha")
def redeem_tpu_transfer(metadata):
    """
    Step 4: Called by Actor B to finalize the ICI move.
    Both actors are now calling device_put simultaneously.
    """
    try:
        import jax

        # Identify the specific local chip intended for this data
        target_device = get_local_device_from_global_id(
            metadata.receiver_global_device_id
        )
        if target_device is None:
            raise ValueError(
                "redeem_tpu_transfer: target device not found on this host."
            )

        # For a point-to-point move, the sharding of the receiving array
        # is a single device sharding on the target device.
        sharding = jax.sharding.SingleDeviceSharding(target_device)

        # Handle the 'No Local Shards' case.
        # On the receiving side, we pass an empty list of arrays to
        # make_array_from_single_device_arrays.
        xs = []

        arr = jax.make_array_from_single_device_arrays(
            shape=metadata.shape, sharding=sharding, arrays=xs, dtype=metadata.dtype
        )

        # Trigger the collective handshake.
        return jax.device_put(arr, target_device)
    except ImportError:
        logger.warning("Could not redeem TPU transfer because jax is not installed.")
        return None
