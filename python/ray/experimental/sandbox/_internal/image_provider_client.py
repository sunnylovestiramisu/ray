"""Client for a node-local image provider plugin.

An image provider plugin is an out-of-process gRPC service
(``imageprovider.v1alpha1.ImageProviderPlugin``, see ``image_provider.proto``)
that serves on a Unix domain socket on the node. Given an image reference, it
lazily mounts the image, for example by streaming its layers with Riptide or
Nydus, and returns one merged, read-only root filesystem directory on the
host. A sandbox then boots from that directory instead of from an EROFS image
that Ray pulls and builds itself.

Every successful ``PrepareImage`` takes one lease on the image, and every
``ReleaseImage`` drops one. The functions here are stateless; the image
manager maps leases to sandboxes.

The client never logs registry credentials.

``image_provider.proto`` is vendored from the plugin's API
(``api/imageprovider/v1alpha1/image_provider.proto``), and its stubs are
generated the same way as ``http/_proto``'s, whose gencode runs on every
protobuf runtime Ray supports (3.20 and up)::

    pip install "grpcio-tools==1.62.3"
    cd python
    python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. \\
        ray/experimental/sandbox/_internal/image_provider.proto
"""

import json
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ray.experimental.sandbox._internal import docker_config
from ray.experimental.sandbox._internal.image_utils import parse_image_ref

try:
    import grpc

    from ray.experimental.sandbox._internal import (
        image_provider_pb2,
        image_provider_pb2_grpc,
    )
except ImportError:  # pragma: no cover - depends on the installed extras.
    grpc = None
    image_provider_pb2 = None
    image_provider_pb2_grpc = None

logger = logging.getLogger(__name__)

SOCKET_ENV = "RAY_SANDBOX_IMAGE_PROVIDER_SOCKET"
DEFAULT_SOCKET = "/run/ate/image-provider.sock"

# auto: stream when the plugin can, else pull and build an EROFS image.
# disabled: never call the plugin. required: fail rather than fall back.
STREAMING_MODE_ENV = "RAY_SANDBOX_IMAGE_STREAMING"
STREAMING_AUTO = "auto"
STREAMING_DISABLED = "disabled"
STREAMING_REQUIRED = "required"
_STREAMING_MODES = (STREAMING_AUTO, STREAMING_DISABLED, STREAMING_REQUIRED)

# Set to 0 to never ask the GCE metadata server for a registry token.
GCE_METADATA_AUTH_ENV = "RAY_SANDBOX_IMAGE_PROVIDER_GCE_METADATA_AUTH"
_GCE_TOKEN_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/"
    "service-accounts/default/token"
)
_GCE_METADATA_TIMEOUT_SECONDS = 1.0
# The username Google registries take an OAuth2 access token with.
_OAUTH2_ACCESS_TOKEN_USERNAME = "oauth2accesstoken"

CAN_PROVIDE_TIMEOUT_SECONDS = 2.0
RELEASE_TIMEOUT_SECONDS = 10.0


class ImageProviderError(RuntimeError):
    """An image provider plugin call failed."""


class ImageProviderNotSupportedError(ImageProviderError):
    """The plugin can't serve the image; the caller should fall back.

    Raised when the socket is missing, ``grpc`` is not installed, or
    ``PrepareImage`` returns ``FAILED_PRECONDITION`` or ``UNIMPLEMENTED``.
    """


@dataclass(frozen=True)
class RegistryAuth:
    """Registry credentials sent along with one image request.

    The secrets stay out of ``repr``, so logging the object leaks nothing.
    """

    server_address: str = ""
    username: str = ""
    password: str = field(default="", repr=False)
    identity_token: str = field(default="", repr=False)
    registry_token: str = field(default="", repr=False)

    def to_proto(self) -> Any:
        return image_provider_pb2.RegistryAuth(
            username=self.username,
            password=self.password,
            server_address=self.server_address,
            identity_token=self.identity_token,
            registry_token=self.registry_token,
        )


@dataclass(frozen=True)
class CanProvideResult:
    can_provide: bool
    provider_name: str
    serving: bool


@dataclass(frozen=True)
class PreparedImage:
    """A ``PrepareImageResponse`` whose layers are a merged rootfs."""

    rootfs_dir: str
    image_digest: str
    provider_name: str
    # The OCI v1 image config (Entrypoint, Cmd, Env, User, WorkingDir).
    config: Dict[str, Any]


def socket_path(path: Optional[str] = None) -> str:
    """The plugin's socket: ``path``, else ``$RAY_SANDBOX_IMAGE_PROVIDER_SOCKET``,
    else ``/run/ate/image-provider.sock``."""
    return path or os.environ.get(SOCKET_ENV) or DEFAULT_SOCKET


def streaming_mode() -> str:
    """The ``RAY_SANDBOX_IMAGE_STREAMING`` mode: auto, disabled or required."""
    mode = (os.environ.get(STREAMING_MODE_ENV) or STREAMING_AUTO).strip().lower()
    if mode not in _STREAMING_MODES:
        logger.warning(
            "Ignoring %s=%r; expected one of %s.",
            STREAMING_MODE_ENV,
            mode,
            ", ".join(_STREAMING_MODES),
        )
        return STREAMING_AUTO
    return mode


def available(path: Optional[str] = None) -> bool:
    """Whether a plugin socket exists and the gRPC stubs import."""
    return grpc is not None and os.path.exists(socket_path(path))


def _channel(path: Optional[str]):
    return grpc.insecure_channel(f"unix://{socket_path(path)}")


def _gce_metadata_token() -> Optional[str]:
    if os.environ.get(GCE_METADATA_AUTH_ENV, "1").strip() in ("0", "false", "no"):
        return None
    req = urllib.request.Request(_GCE_TOKEN_URL, headers={"Metadata-Flavor": "Google"})
    try:
        with urllib.request.urlopen(req, timeout=_GCE_METADATA_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8")).get("access_token")
    except Exception as err:
        logger.debug("No GCE metadata token for registry auth: %s", type(err).__name__)
        return None


def _is_google_registry(host: str) -> bool:
    return host.endswith(("docker.pkg.dev", "gcr.io"))


def resolve_registry_auth(image: str, timeout: float = 10.0) -> Optional[RegistryAuth]:
    """Best-effort registry credentials for ``image``, for the plugin to pull with.

    Uses the node's Docker config the way Ray's own pull does (credential
    helpers, then ``auths``). For Artifact Registry and GCR images without
    Docker config credentials, falls back to the node's GCE service account
    token from the metadata server, as ``oauth2accesstoken``. Never raises.

    Args:
        image: Registry image reference.
        timeout: Credential helper timeout in seconds.

    Returns:
        The credentials, or None to send none.
    """
    try:
        registry, _, _ = parse_image_ref(image)
    except Exception:
        return None
    try:
        creds = docker_config.registry_credentials(registry, timeout)
    except Exception as err:
        # The error names the config file or helper, never a secret.
        logger.debug("No Docker config credentials for %s: %s", registry, err)
        creds = None
    if creds is not None and (
        creds.username or creds.identity_token or creds.registry_token
    ):
        return RegistryAuth(
            server_address=registry,
            username=creds.username,
            password=creds.password,
            identity_token=creds.identity_token,
            registry_token=creds.registry_token,
        )
    if _is_google_registry(registry):
        token = _gce_metadata_token()
        if token:
            return RegistryAuth(
                server_address=registry,
                username=_OAUTH2_ACCESS_TOKEN_USERNAME,
                password=token,
            )
    return None


def can_provide(
    image: str,
    auth: Optional[RegistryAuth] = None,
    path: Optional[str] = None,
    timeout: float = CAN_PROVIDE_TIMEOUT_SECONDS,
) -> CanProvideResult:
    """Ask the plugin whether it can serve ``image`` as a merged rootfs.

    Raises:
        ImageProviderNotSupportedError: If there's no plugin socket or grpc.
        ImageProviderError: If the RPC fails.
    """
    if not available(path):
        raise ImageProviderNotSupportedError(
            f"No image provider plugin at {socket_path(path)}"
            + ("" if grpc is not None else " (grpc is not installed)")
        )
    request = image_provider_pb2.CanProvideRequest(
        image_ref=image, form=image_provider_pb2.LAYER_FORM_ROOTFS
    )
    if auth is not None:
        request.auth.CopyFrom(auth.to_proto())
    try:
        with _channel(path) as channel:
            stub = image_provider_pb2_grpc.ImageProviderPluginStub(channel)
            resp = stub.CanProvide(request, timeout=timeout)
    except grpc.RpcError as err:
        raise ImageProviderError(
            f"CanProvide failed for {image}: {err.code().name}: {err.details()}"
        ) from err
    return CanProvideResult(
        can_provide=bool(resp.can_provide),
        provider_name=resp.provider_name,
        serving=resp.state == image_provider_pb2.PLUGIN_STATE_SERVING,
    )


def _parse_config(raw: bytes, image: str) -> Dict[str, Any]:
    """The image config in ``.image_config.json``'s shape: ``{"config": {...}}``."""
    if not raw:
        return {"config": {}}
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as err:
        raise ImageProviderError(
            f"PrepareImage returned an unparsable config_json for {image}: {err}"
        ) from err
    if not isinstance(parsed, dict):
        raise ImageProviderError(
            f"PrepareImage returned a config_json for {image} that isn't an object"
        )
    # A full OCI image config nests the runtime config under "config".
    if isinstance(parsed.get("config"), dict):
        return parsed
    return {"config": parsed}


def prepare_image(
    image: str,
    auth: Optional[RegistryAuth] = None,
    path: Optional[str] = None,
    timeout: float = 120.0,
) -> PreparedImage:
    """Prepare ``image`` as a merged rootfs, taking one lease on it.

    Raises:
        ImageProviderNotSupportedError: If there's no plugin socket or grpc,
            or the plugin answers FAILED_PRECONDITION or UNIMPLEMENTED.
        ImageProviderError: If the RPC fails otherwise, or the response holds
            no rootfs.
    """
    if not available(path):
        raise ImageProviderNotSupportedError(
            f"No image provider plugin at {socket_path(path)}"
        )
    request = image_provider_pb2.PrepareImageRequest(
        image_ref=image, form=image_provider_pb2.LAYER_FORM_ROOTFS
    )
    if auth is not None:
        request.auth.CopyFrom(auth.to_proto())
    try:
        with _channel(path) as channel:
            stub = image_provider_pb2_grpc.ImageProviderPluginStub(channel)
            resp = stub.PrepareImage(request, timeout=timeout)
    except grpc.RpcError as err:
        code = err.code()
        if code in (
            grpc.StatusCode.FAILED_PRECONDITION,
            grpc.StatusCode.UNIMPLEMENTED,
        ):
            raise ImageProviderNotSupportedError(
                f"Image provider declined {image}: {code.name}: {err.details()}"
            ) from err
        raise ImageProviderError(
            f"PrepareImage failed for {image}: {code.name}: {err.details()}"
        ) from err

    layers = resp.WhichOneof("layers")
    if layers != "rootfs" or not resp.rootfs.dir:
        # The plugin took a lease; give it back before rejecting the answer.
        release_image(image, path=path)
        raise ImageProviderError(
            f"PrepareImage for {image} returned {layers or 'no layers'} "
            "instead of the requested rootfs"
        )
    try:
        config = _parse_config(resp.config_json, image)
    except ImageProviderError:
        release_image(image, path=path)
        raise
    return PreparedImage(
        # A single-layer image's rootfs may be a symlink to the layer's mount;
        # gVisor gets the directory itself.
        rootfs_dir=os.path.realpath(resp.rootfs.dir),
        image_digest=resp.image_digest,
        provider_name=resp.provider_name,
        config=config,
    )


def release_image(
    image: str,
    path: Optional[str] = None,
    timeout: float = RELEASE_TIMEOUT_SECONDS,
) -> bool:
    """Release one lease on ``image``. Best-effort: never raises.

    Returns:
        Whether the plugin acknowledged the release.
    """
    if not available(path):
        logger.warning(
            "Can't release image provider lease on %s: no plugin at %s",
            image,
            socket_path(path),
        )
        return False
    try:
        with _channel(path) as channel:
            stub = image_provider_pb2_grpc.ImageProviderPluginStub(channel)
            stub.ReleaseImage(
                image_provider_pb2.ReleaseImageRequest(image_ref=image),
                timeout=timeout,
            )
        return True
    except grpc.RpcError as err:
        logger.warning(
            "ReleaseImage failed for %s: %s: %s", image, err.code().name, err.details()
        )
        return False
