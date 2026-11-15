"""Tests for streaming sandbox images through an image provider plugin.

A fake ``imageprovider.v1alpha1.ImageProviderPlugin`` serves on a temporary
Unix socket in this process; the EROFS fallback is stubbed out, so nothing
here needs a registry, mkfs.erofs or runsc.
"""

import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent import futures

import pytest

grpc = pytest.importorskip("grpc")

from ray.experimental.sandbox import image_manager as image_manager_mod  # noqa: E402
from ray.experimental.sandbox._internal import (  # noqa: E402
    image_provider_client,
    image_provider_pb2 as pb,
    image_provider_pb2_grpc as pb_grpc,
    image_utils,
    overlayfs,
)
from ray.experimental.sandbox._internal.image_utils import (  # noqa: E402
    ROOTFS_IMAGE,
    STREAMED_MARKER,
)
from ray.experimental.sandbox.exceptions import SandboxCreationError  # noqa: E402
from ray.experimental.sandbox.image_manager import ImageManager  # noqa: E402

IMAGE = "registry.example.com/team/app:1.0"
IMAGE_CONFIG = {
    "Env": ["PATH=/usr/bin", "LANG=C.UTF-8"],
    "WorkingDir": "/app",
    "Entrypoint": ["/bin/app"],
    "User": "1000",
}


class FakePlugin(pb_grpc.ImageProviderPluginServicer):
    """An image provider plugin that hands out one fixed rootfs directory."""

    def __init__(self, rootfs_dir):
        self.rootfs_dir = rootfs_dir
        self.can_provide = True
        self.state = pb.PLUGIN_STATE_SERVING
        self.prepare_error = None
        self.reconcile_error = None
        self.wrong_form = False
        self.requests = []
        self.leases = Counter()

    def calls(self, method):
        return [req for name, req in self.requests if name == method]

    def CanProvide(self, request, context):
        self.requests.append(("CanProvide", request))
        return pb.CanProvideResponse(
            can_provide=self.can_provide, provider_name="fake", state=self.state
        )

    def PrepareImage(self, request, context):
        self.requests.append(("PrepareImage", request))
        if self.prepare_error is not None:
            context.abort(self.prepare_error, "fake plugin says no")
        self.leases[request.image_ref] += 1
        resp = pb.PrepareImageResponse(
            image_digest="sha256:" + "a" * 64,
            config_json=json.dumps(IMAGE_CONFIG).encode(),
            provider_name="fake",
        )
        if self.wrong_form:
            resp.layer_dirs.dirs.append(self.rootfs_dir)
        else:
            resp.rootfs.dir = self.rootfs_dir
        return resp

    def ReleaseImage(self, request, context):
        self.requests.append(("ReleaseImage", request))
        self.leases[request.image_ref] -= 1
        return pb.ReleaseImageResponse()

    def ReconcileImages(self, request, context):
        self.requests.append(("ReconcileImages", request))
        if self.reconcile_error is not None:
            context.abort(self.reconcile_error, "fake plugin can't reconcile")
        self.leases = Counter({a.image_ref: a.ref_count for a in request.active_images})
        return pb.ReconcileImagesResponse()

    def reconciled(self):
        """The leases each ReconcileImages call sent, by reference."""
        return [
            {a.image_ref: a.ref_count for a in req.active_images}
            for req in self.calls("ReconcileImages")
        ]


def _serve(fake, sock):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_ImageProviderPluginServicer_to_server(fake, server)
    server.add_insecure_port(f"unix://{sock}")
    server.start()
    return server


@pytest.fixture
def rootfs_dir(tmp_path):
    """A read-only-ish streamed rootfs with the usual mount points."""
    root = tmp_path / "plugin" / "rootfs"
    for d in ("bin", "etc", "proc", "sys", "dev", "tmp", "app"):
        (root / d).mkdir(parents=True)
    (root / "etc" / "hosts").write_text("")
    (root / "bin" / "sh").write_text("")
    return str(root)


@pytest.fixture
def plugin(rootfs_dir, monkeypatch):
    # Unix socket paths are limited to 108 bytes; pytest's tmp_path may not be.
    sock_dir = tempfile.mkdtemp(prefix="imgprov-", dir="/tmp")
    sock = os.path.join(sock_dir, "plugin.sock")
    fake = FakePlugin(rootfs_dir)
    server = _serve(fake, sock)
    monkeypatch.setenv(image_provider_client.SOCKET_ENV, sock)
    monkeypatch.setenv(image_provider_client.STREAMING_MODE_ENV, "auto")
    monkeypatch.setenv(image_provider_client.GCE_METADATA_AUTH_ENV, "0")
    try:
        yield fake
    finally:
        server.stop(None)
        shutil.rmtree(sock_dir, ignore_errors=True)


@pytest.fixture
def local_pulls(monkeypatch):
    """Stub out the local EROFS pull; returns the images it was asked for."""
    pulled = []

    def fake_pull(image, images_dir, timeout_seconds=120.0, instance_id=None):
        pulled.append((image, instance_id))
        target = os.path.join(images_dir, image_utils.image_cache_name(image))
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, ROOTFS_IMAGE), "wb") as f:
            f.write(b"erofs")
        with open(os.path.join(target, ".extracted"), "w") as f:
            f.write(image_utils.expected_extract_marker())
        if instance_id is not None:
            image_utils._mark_image_in_use(target, instance_id)
        return target

    monkeypatch.setattr(
        image_manager_mod, "pull_and_extract_container_image", fake_pull
    )
    return pulled


@pytest.fixture(autouse=True)
def _no_overlay_probe(monkeypatch):
    monkeypatch.setattr(overlayfs, "can_mount_rootfs_overlay", lambda: False)


def _base_spec():
    return {"process": {}, "root": {}, "mounts": [], "linux": {}}


def _read_meta(mgr):
    with open(os.path.join(mgr.get_streamed_image_dir(IMAGE), STREAMED_MARKER)) as f:
        return json.load(f)


def test_cold_pull_streams_the_image(tmp_path, plugin, local_pulls, rootfs_dir, caplog):
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    with caplog.at_level(logging.INFO, logger=image_manager_mod.__name__):
        image_dir = mgr.pull_image(IMAGE, instance_id="sbx-1")

    assert image_dir == mgr.get_streamed_image_dir(IMAGE)
    assert local_pulls == []
    assert not os.path.exists(os.path.join(image_dir, ROOTFS_IMAGE))
    # Both RPCs ask for a merged rootfs.
    assert [r.form for r in plugin.calls("CanProvide")] == [pb.LAYER_FORM_ROOTFS]
    assert [r.form for r in plugin.calls("PrepareImage")] == [pb.LAYER_FORM_ROOTFS]
    assert plugin.leases[IMAGE] == 1

    meta = _read_meta(mgr)
    assert meta["mode"] == "streamed"
    assert meta["provider"] == "fake"
    assert meta["rootfs_dir"] == os.path.realpath(rootfs_dir)
    assert meta["image_digest"] == "sha256:" + "a" * 64
    assert meta["prepare_ms"] >= 0
    # The config lands in .image_config.json's usual shape.
    assert mgr.get_image_config(IMAGE) == {"config": IMAGE_CONFIG}
    assert mgr.get_envs(IMAGE) == IMAGE_CONFIG["Env"]
    assert mgr.get_workdir(IMAGE) == "/app"
    assert mgr.is_image_extracted(IMAGE)

    info = mgr.describe_image(IMAGE)
    assert info["mode"] == "streamed"
    assert info["provider"] == "fake"
    assert info["digest"] == meta["image_digest"]
    assert info["rootfs"] == os.path.realpath(rootfs_dir)
    assert info["pins"] == ["sbx-1"]
    assert any(
        "image_pull mode=streamed provider=fake image=" + IMAGE in rec.getMessage()
        for rec in caplog.records
    )


def test_warm_pulls_take_no_extra_lease(tmp_path, plugin, local_pulls):
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    mgr.pull_image(IMAGE, instance_id="sbx-1")
    mgr.pull_image(IMAGE, instance_id="sbx-1")
    mgr.pull_image(IMAGE)
    mgr.create_oci_spec(image=IMAGE, base_spec=_base_spec(), readonly=False)
    # A fresh manager (another worker) sees the streamed entry too.
    ImageManager(images_dir=mgr.images_dir).create_oci_spec(
        image=IMAGE, base_spec=_base_spec(), readonly=False
    )
    assert len(plugin.calls("PrepareImage")) == 1
    assert plugin.leases[IMAGE] == 1


def test_each_sandbox_pin_holds_one_lease(tmp_path, plugin, local_pulls):
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    mgr.pull_image(IMAGE, instance_id="sbx-1")
    mgr.pull_image(IMAGE, instance_id="sbx-2")
    assert len(plugin.calls("PrepareImage")) == 2
    assert plugin.leases[IMAGE] == 2
    assert mgr.describe_image(IMAGE)["pins"] == ["sbx-1", "sbx-2"]

    mgr.release_image(IMAGE, "sbx-1")
    assert plugin.leases[IMAGE] == 1
    assert [r.image_ref for r in plugin.calls("ReleaseImage")] == [IMAGE]
    assert mgr.describe_image(IMAGE)["pins"] == ["sbx-2"]

    # Releasing twice, or an instance that never pulled, releases nothing.
    mgr.release_image(IMAGE, "sbx-1")
    mgr.release_image(IMAGE, "never-pulled")
    assert len(plugin.calls("ReleaseImage")) == 1

    mgr.release_image(IMAGE, "sbx-2")
    assert plugin.leases[IMAGE] == 0
    assert mgr.describe_image(IMAGE)["pins"] == []


def test_prepull_lease_is_claimed_by_the_first_sandbox(tmp_path, plugin, local_pulls):
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    mgr.pull_image(IMAGE)
    assert mgr.describe_image(IMAGE)["pins"] == [".unclaimed"]
    assert plugin.leases[IMAGE] == 1

    mgr.pull_image(IMAGE, instance_id="sbx-1")
    assert len(plugin.calls("PrepareImage")) == 1
    assert mgr.describe_image(IMAGE)["pins"] == ["sbx-1"]

    # A pin already holds a lease, so another pre-pull takes none.
    mgr.pull_image(IMAGE)
    assert len(plugin.calls("PrepareImage")) == 1

    mgr.release_image(IMAGE, "sbx-1")
    assert plugin.leases[IMAGE] == 0


def test_release_repeats_the_reference_the_lease_was_taken_with(
    tmp_path, plugin, local_pulls
):
    """Two spellings of one image share a cache entry; ReleaseImage must use
    the reference PrepareImage used."""
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    mgr.pull_image("python:3.12")
    mgr.pull_image("docker.io/library/python:3.12", instance_id="sbx-1")
    assert len(plugin.calls("PrepareImage")) == 1
    mgr.release_image("docker.io/library/python:3.12", "sbx-1")
    assert [r.image_ref for r in plugin.calls("ReleaseImage")] == ["python:3.12"]
    assert plugin.leases["python:3.12"] == 0


OTHER_IMAGE = "registry.example.com/team/other:2.0"
_HOLD_PIN = (
    "import fcntl, sys\n"
    "f = open(sys.argv[1])\n"
    "fcntl.flock(f, fcntl.LOCK_SH)\n"
    "print('locked', flush=True)\n"
    "sys.stdin.read()\n"
)


def _write_pin(images_dir, image, user, rootfs_dir=None):
    """A pin on a streamed image as another worker's pull writes it, and the
    entry's metadata if ``rootfs_dir`` is given."""
    streamed_dir = os.path.join(
        images_dir,
        image_utils.image_cache_name(image) + image_utils.STREAMED_DIR_SUFFIX,
    )
    users_dir = os.path.join(streamed_dir, image_utils._USERS_SUBDIR)
    os.makedirs(users_dir, exist_ok=True)
    if rootfs_dir is not None:
        with open(os.path.join(streamed_dir, STREAMED_MARKER), "w") as f:
            json.dump(
                {"mode": "streamed", "image_ref": image, "rootfs_dir": rootfs_dir}, f
            )
        with open(os.path.join(streamed_dir, ".image_config.json"), "w") as f:
            json.dump({"config": IMAGE_CONFIG}, f)
    pin = os.path.join(users_dir, user)
    with open(pin, "w") as f:
        f.write(image)
    return pin


def _hold_in_another_process(pin):
    """Hold a pin the way a live worker in another process (or Pod) does."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLD_PIN, pin],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout.readline().strip() == "locked"
    return proc


def test_first_streamed_pull_reconciles_the_live_pins(
    tmp_path, plugin, local_pulls, rootfs_dir
):
    """ReconcileImages goes out once, before the first lease, with the leases
    of every live pin on the node, the other workers' included, and without
    the pins of sandboxes whose workers died."""
    images_dir = str(tmp_path / "images")
    live_pin = _write_pin(images_dir, IMAGE, "sbx-other", rootfs_dir=rootfs_dir)
    dead_pin = _write_pin(images_dir, IMAGE, "sbx-dead")
    unclaimed_pin = _write_pin(images_dir, IMAGE, ".unclaimed")
    dead_other_pin = _write_pin(images_dir, OTHER_IMAGE, "sbx-dead-2")
    # A worker that held a pin and died dropped its lock.
    dead = _hold_in_another_process(dead_pin)
    dead.kill()
    dead.wait()
    other = _hold_in_another_process(live_pin)
    unclaimed_holder = _hold_in_another_process(unclaimed_pin)
    plugin.leases.update({IMAGE: 3, OTHER_IMAGE: 1})
    try:
        mgr = ImageManager(images_dir=images_dir)
        mgr.pull_image(IMAGE, instance_id="sbx-1")

        assert plugin.reconciled() == [{IMAGE: 2}]
        assert not os.path.exists(dead_pin)
        assert not os.path.exists(dead_other_pin)
        assert os.path.exists(live_pin)
        assert plugin.leases == Counter({IMAGE: 2})
        # The new sandbox claimed the unclaimed lease; no new lease was taken.
        assert plugin.calls("PrepareImage") == []

        # Later pulls, from this plugin socket, don't reconcile again.
        mgr.pull_image(OTHER_IMAGE, instance_id="sbx-2")
        mgr.release_image(IMAGE, "sbx-1")
        mgr.pull_image(IMAGE, instance_id="sbx-3")
        assert len(plugin.calls("ReconcileImages")) == 1
        assert plugin.leases == Counter({IMAGE: 2, OTHER_IMAGE: 1})
    finally:
        for proc in (other, unclaimed_holder):
            proc.kill()
            proc.wait()


def test_a_new_plugin_socket_is_reconciled_with_this_workers_pins(
    tmp_path, plugin, local_pulls, rootfs_dir, monkeypatch
):
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    mgr.pull_image(IMAGE, instance_id="sbx-1")
    mgr.pull_image(IMAGE)
    assert plugin.reconciled() == [{}]

    # The plugin restarted, on a new socket, and knows of no leases.
    sock_dir = tempfile.mkdtemp(prefix="imgprov-", dir="/tmp")
    restarted = FakePlugin(rootfs_dir)
    server = _serve(restarted, os.path.join(sock_dir, "plugin.sock"))
    monkeypatch.setenv(
        image_provider_client.SOCKET_ENV, os.path.join(sock_dir, "plugin.sock")
    )
    try:
        mgr.pull_image(OTHER_IMAGE, instance_id="sbx-2")
        # This worker's own pins hold their locks, so they're live.
        assert restarted.reconciled() == [{IMAGE: 1}]
        assert restarted.leases == Counter({IMAGE: 1, OTHER_IMAGE: 1})
    finally:
        server.stop(None)
        shutil.rmtree(sock_dir, ignore_errors=True)


def test_image_managers_on_a_reconciled_node_skip_reconcile(
    tmp_path, plugin, local_pulls, rootfs_dir, monkeypatch
):
    """ReconcileImages holds the leases lock exclusive, which stalls every
    streamed pull on the node, so the many ImageManagers on a node (one per
    thread or process) send it once per plugin socket."""
    images_dir = str(tmp_path / "images")
    # Keep the managers, and so the locks on their pins, alive.
    mgrs = [ImageManager(images_dir=images_dir) for _ in range(5)]
    for i, mgr in enumerate(mgrs):
        mgr.pull_image(IMAGE, instance_id=f"sbx-{i}")
    assert len(plugin.calls("ReconcileImages")) == 1

    # Another process on the node has no record of its own; it finds the
    # node already reconciled with this plugin and no dead pins.
    monkeypatch.setattr(image_manager_mod, "_reconciled_in_process", set())
    ImageManager(images_dir=images_dir).pull_image(OTHER_IMAGE, instance_id="sbx-x")
    assert len(plugin.calls("ReconcileImages")) == 1


def test_dead_pins_on_a_reconciled_node_are_reconciled_again(
    tmp_path, plugin, local_pulls, rootfs_dir, monkeypatch
):
    """A worker or Pod that restarts leaves pins of dead owners on a node that
    already reconciled with the plugin; the next process reconciles again."""
    images_dir = str(tmp_path / "images")
    mgr = ImageManager(images_dir=images_dir)
    mgr.pull_image(IMAGE, instance_id="sbx-1")
    assert len(plugin.calls("ReconcileImages")) == 1

    dead_pin = _write_pin(images_dir, OTHER_IMAGE, "sbx-dead")
    plugin.leases[OTHER_IMAGE] = 1
    dead = _hold_in_another_process(dead_pin)
    dead.kill()
    dead.wait()

    monkeypatch.setattr(image_manager_mod, "_reconciled_in_process", set())
    ImageManager(images_dir=images_dir).pull_image(IMAGE, instance_id="sbx-2")
    # sbx-1's pin is still held by ``mgr``, so it's live.
    assert plugin.reconciled()[-1] == {IMAGE: 1}
    assert not os.path.exists(dead_pin)
    assert plugin.leases == Counter({IMAGE: 2})


@pytest.mark.parametrize(
    "code,retried",
    [(grpc.StatusCode.UNIMPLEMENTED, False), (grpc.StatusCode.INTERNAL, True)],
)
def test_reconcile_errors_never_fail_the_pull(
    tmp_path, plugin, local_pulls, code, retried
):
    images_dir = str(tmp_path / "images")
    dead_pin = _write_pin(images_dir, OTHER_IMAGE, "sbx-dead")
    plugin.reconcile_error = code
    mgr = ImageManager(images_dir=images_dir)
    assert mgr.pull_image(IMAGE, instance_id="sbx-1") == mgr.get_streamed_image_dir(
        IMAGE
    )
    assert plugin.leases[IMAGE] == 1
    # The plugin may still hold the dead pin's lease, so the pin stays.
    assert os.path.exists(dead_pin)

    mgr.pull_image(IMAGE, instance_id="sbx-2")
    assert len(plugin.calls("ReconcileImages")) == (2 if retried else 1)
    assert plugin.leases[IMAGE] == 2


def test_reconcile_can_be_disabled(tmp_path, plugin, local_pulls, monkeypatch):
    monkeypatch.setenv(image_manager_mod.RECONCILE_ENV, "0")
    images_dir = str(tmp_path / "images")
    dead_pin = _write_pin(images_dir, OTHER_IMAGE, "sbx-dead")
    ImageManager(images_dir=images_dir).pull_image(IMAGE, instance_id="sbx-1")
    assert plugin.calls("ReconcileImages") == []
    assert os.path.exists(dead_pin)


def test_missing_socket_falls_back(tmp_path, local_pulls, monkeypatch):
    monkeypatch.setenv(image_provider_client.SOCKET_ENV, str(tmp_path / "nope.sock"))
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    image_dir = mgr.pull_image(IMAGE, instance_id="sbx-1")
    assert local_pulls == [(IMAGE, "sbx-1")]
    assert os.path.isfile(os.path.join(image_dir, ROOTFS_IMAGE))
    assert mgr.describe_image(IMAGE)["mode"] == "erofs"
    # No lock file or entry for a streamed image appears.
    assert not os.path.exists(mgr.get_streamed_image_dir(IMAGE) + ".lock")


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(lambda p: setattr(p, "can_provide", False), id="cannot"),
        pytest.param(
            lambda p: setattr(p, "state", pb.PLUGIN_STATE_DRAINING), id="draining"
        ),
        pytest.param(
            lambda p: setattr(p, "state", pb.PLUGIN_STATE_UNSPECIFIED),
            id="unspecified",
        ),
    ],
)
def test_can_provide_declines_fall_back(tmp_path, plugin, local_pulls, setup):
    setup(plugin)
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    mgr.pull_image(IMAGE, instance_id="sbx-1")
    assert local_pulls == [(IMAGE, "sbx-1")]
    assert plugin.calls("PrepareImage") == []
    assert mgr.image_mode(IMAGE) == "erofs"


@pytest.mark.parametrize(
    "code", [grpc.StatusCode.FAILED_PRECONDITION, grpc.StatusCode.UNIMPLEMENTED]
)
def test_prepare_declines_fall_back(tmp_path, plugin, local_pulls, code):
    plugin.prepare_error = code
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    mgr.pull_image(IMAGE, instance_id="sbx-1")
    assert local_pulls == [(IMAGE, "sbx-1")]
    assert mgr.image_mode(IMAGE) == "erofs"
    spec = mgr.create_oci_spec(image=IMAGE, base_spec=_base_spec(), readonly=False)
    assert spec["annotations"]["dev.gvisor.spec.rootfs.type"] == "erofs"
    # The sandbox's spec reuses its own pull; it doesn't ask the plugin again.
    assert len(plugin.calls("PrepareImage")) == 1
    # Releasing the EROFS pin releases no plugin lease.
    mgr.release_image(IMAGE, "sbx-1")
    assert plugin.calls("ReleaseImage") == []


def test_other_prepare_errors_fail_the_pull(tmp_path, plugin, local_pulls):
    plugin.prepare_error = grpc.StatusCode.INTERNAL
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    with pytest.raises(SandboxCreationError, match="INTERNAL"):
        mgr.pull_image(IMAGE, instance_id="sbx-1")
    assert local_pulls == []
    # The plugin already dropped the failed lease; Ray releases nothing.
    assert plugin.calls("ReleaseImage") == []


def test_a_response_without_a_rootfs_is_rejected(tmp_path, plugin, local_pulls):
    plugin.wrong_form = True
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    with pytest.raises(SandboxCreationError, match="layer_dirs"):
        mgr.pull_image(IMAGE, instance_id="sbx-1")
    # The lease the plugin took is given back.
    assert plugin.leases[IMAGE] == 0
    assert len(plugin.calls("ReleaseImage")) == 1


def test_required_mode_fails_instead_of_falling_back(
    tmp_path, plugin, local_pulls, monkeypatch
):
    monkeypatch.setenv(image_provider_client.STREAMING_MODE_ENV, "required")
    plugin.can_provide = False
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    with pytest.raises(SandboxCreationError, match="required"):
        mgr.pull_image(IMAGE, instance_id="sbx-1")
    assert local_pulls == []

    monkeypatch.setenv(image_provider_client.SOCKET_ENV, str(tmp_path / "nope.sock"))
    with pytest.raises(SandboxCreationError, match="required"):
        mgr.pull_image(IMAGE, instance_id="sbx-1")


def test_disabled_mode_never_calls_the_plugin(
    tmp_path, plugin, local_pulls, monkeypatch
):
    monkeypatch.setenv(image_provider_client.STREAMING_MODE_ENV, "disabled")
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    mgr.pull_image(IMAGE, instance_id="sbx-1")
    assert plugin.requests == []
    assert local_pulls == [(IMAGE, "sbx-1")]


def test_local_archives_are_never_streamed(tmp_path, plugin, local_pulls):
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    mgr.pull_image(str(tmp_path / "image.tar"), instance_id="sbx-1")
    assert plugin.requests == []


@pytest.mark.parametrize("disk", [None, "64Mi"])
def test_writable_streamed_spec_binds_the_rootfs(
    tmp_path, plugin, local_pulls, rootfs_dir, disk
):
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    mgr.pull_image(IMAGE, instance_id="sbx-1")
    spec = mgr.create_oci_spec(
        image=IMAGE,
        base_spec=_base_spec(),
        readonly=False,
        root_path=str(tmp_path / "bundle" / "rootfs"),
        disk=disk,
    )
    rootfs = os.path.realpath(rootfs_dir)
    expected = {
        "dev.gvisor.spec.rootfs.source": rootfs,
        "dev.gvisor.spec.rootfs.type": "bind",
        "dev.gvisor.spec.rootfs.overlay": "memory",
    }
    if disk:
        expected["dev.gvisor.spec.rootfs.options"] = f"size={64 * 1024 * 1024}"
    assert spec["annotations"] == expected
    assert spec["root"] == {"path": rootfs, "readonly": False}
    assert spec["process"]["env"] == IMAGE_CONFIG["Env"]


def test_erofs_spec_is_unchanged(tmp_path, local_pulls, monkeypatch):
    monkeypatch.setenv(image_provider_client.SOCKET_ENV, str(tmp_path / "nope.sock"))
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    image_dir = mgr.pull_image(IMAGE, instance_id="sbx-1")
    root_path = str(tmp_path / "bundle" / "rootfs")
    spec = mgr.create_oci_spec(
        image=IMAGE, base_spec=_base_spec(), readonly=False, root_path=root_path
    )
    assert spec["annotations"] == {
        "dev.gvisor.spec.rootfs.source": os.path.join(image_dir, ROOTFS_IMAGE),
        "dev.gvisor.spec.rootfs.type": "erofs",
        "dev.gvisor.spec.rootfs.overlay": "self",
    }
    assert spec["root"]["path"] == root_path


def test_readonly_streamed_spec_with_its_mount_points(
    tmp_path, plugin, local_pulls, rootfs_dir
):
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    mgr.pull_image(IMAGE, instance_id="sbx-1")
    hosts = tmp_path / "hosts"
    hosts.write_text("")
    spec = mgr.create_oci_spec(
        image=IMAGE, base_spec=_base_spec(), readonly=True, hosts_source=str(hosts)
    )
    rootfs = os.path.realpath(rootfs_dir)
    assert spec["annotations"] == {
        "dev.gvisor.spec.rootfs.source": rootfs,
        "dev.gvisor.spec.rootfs.type": "bind",
    }
    assert spec["root"] == {"path": rootfs, "readonly": True}


@pytest.mark.parametrize("can_mount", [True, False])
def test_readonly_streamed_spec_missing_a_mount_point(
    tmp_path, plugin, local_pulls, monkeypatch, can_mount
):
    """The streamed rootfs has no /etc/resolv.conf. An EROFS image always
    does (Ray seeds it), so only the streamed image needs an overlay for it."""
    monkeypatch.setattr(overlayfs, "can_mount_rootfs_overlay", lambda: can_mount)
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    mgr.pull_image(IMAGE, instance_id="sbx-1")
    resolv = tmp_path / "resolv.conf"
    resolv.write_text("nameserver 1.1.1.1\n")
    root_path = str(tmp_path / "bundle" / "rootfs")
    spec = mgr.create_oci_spec(
        image=IMAGE,
        base_spec=_base_spec(),
        readonly=True,
        resolv_conf_source=str(resolv),
        root_path=root_path,
    )
    rootfs_dir = mgr.get_streamed_rootfs_dir(IMAGE)
    if can_mount:
        # The backend boots it from a kernel overlay over the directory.
        assert overlayfs.needs_rootfs_overlay_for_image(spec, rootfs_dir)
        assert spec.get("annotations", {}) == {}
        assert spec["root"] == {"path": root_path, "readonly": True}
    else:
        # A private writable overlay in memory.
        assert spec["root"] == {"path": rootfs_dir, "readonly": False}
        assert spec["annotations"]["dev.gvisor.spec.rootfs.overlay"] == "memory"
        assert spec["annotations"]["dev.gvisor.spec.rootfs.type"] == "bind"


def test_overlay_over_a_streamed_rootfs_binds_the_directory(
    tmp_path, rootfs_dir, monkeypatch
):
    monkeypatch.setattr(
        overlayfs.ImageMountMode,
        "detect",
        classmethod(lambda cls, userns: overlayfs.ImageMountMode.FUSE),
    )
    overlay = overlayfs.prepare(
        image=rootfs_dir,
        mountpoint=str(tmp_path / "bundle" / "rootfs"),
        bundle_dir=str(tmp_path / "bundle"),
        userns=overlayfs.UserNamespaceType.HOST,
        image_mount_mode=overlayfs.ImageMountMode.DIRECTORY,
    )
    assert overlay.image == rootfs_dir
    assert overlay.image_mount_mode is overlayfs.ImageMountMode.DIRECTORY
    script = overlay.wrap(["true"])[-1]
    assert 'mount --rbind "$image" "$tmpfs/lower"' in script
    assert "erofs" not in script


def test_evicting_a_streamed_entry_leaves_the_plugin_rootfs(
    tmp_path, plugin, local_pulls, rootfs_dir
):
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    mgr.pull_image(IMAGE, instance_id="sbx-1")
    streamed_dir = mgr.get_streamed_image_dir(IMAGE)

    image_utils.evict_least_recently_used_images(mgr.images_dir, max_bytes=0)
    assert os.path.isdir(streamed_dir), "a pinned entry must stay"

    mgr.release_image(IMAGE, "sbx-1")
    image_utils.evict_least_recently_used_images(mgr.images_dir, max_bytes=0)
    assert not os.path.exists(streamed_dir)
    assert os.path.isfile(os.path.join(rootfs_dir, "bin", "sh"))
    assert plugin.leases[IMAGE] == 0

    # A later pull streams it again.
    mgr2 = ImageManager(images_dir=mgr.images_dir)
    mgr2.pull_image(IMAGE, instance_id="sbx-2")
    assert len(plugin.calls("PrepareImage")) == 2


def test_requests_carry_docker_config_credentials(
    tmp_path, plugin, local_pulls, monkeypatch
):
    config_dir = tmp_path / "docker"
    config_dir.mkdir()
    auth = base64.b64encode(b"robot:s3cret").decode()
    (config_dir / "config.json").write_text(
        json.dumps({"auths": {"registry.example.com": {"auth": auth}}})
    )
    monkeypatch.setenv("DOCKER_CONFIG", str(config_dir))
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    mgr.pull_image(IMAGE, instance_id="sbx-1")
    for req in plugin.calls("CanProvide") + plugin.calls("PrepareImage"):
        assert req.HasField("auth")
        assert req.auth.username == "robot"
        assert req.auth.password == "s3cret"
        assert req.auth.server_address == "registry.example.com"


def test_requests_without_credentials_send_no_auth(
    tmp_path, plugin, local_pulls, monkeypatch
):
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path / "no-docker-config"))
    mgr = ImageManager(images_dir=str(tmp_path / "images"))
    mgr.pull_image(IMAGE, instance_id="sbx-1")
    assert not plugin.calls("PrepareImage")[0].HasField("auth")


def test_google_registries_fall_back_to_the_metadata_token(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path / "no-docker-config"))
    monkeypatch.setattr(
        image_provider_client, "_gce_metadata_token", lambda: "fake-metadata-token"
    )
    auth = image_provider_client.resolve_registry_auth(
        "us-docker.pkg.dev/proj/repo/img@sha256:" + "b" * 64
    )
    assert auth.username == "_token"
    assert auth.password == "fake-metadata-token"
    assert auth.server_address == "us-docker.pkg.dev"
    assert "fake-metadata-token" not in repr(auth)
    assert image_provider_client.resolve_registry_auth("busybox:latest") is None


def test_google_registry_helper_usernames_are_normalized_to_token(
    monkeypatch, tmp_path
):
    for helper_user in ("_dcgcr_2_2_1_token", "oauth2accesstoken"):
        config_dir = tmp_path / helper_user
        config_dir.mkdir()
        encoded = base64.b64encode(f"{helper_user}:fake-helper-token".encode()).decode()
        (config_dir / "config.json").write_text(
            json.dumps({"auths": {"us-docker.pkg.dev": {"auth": encoded}}})
        )
        monkeypatch.setenv("DOCKER_CONFIG", str(config_dir))
        auth = image_provider_client.resolve_registry_auth(
            "us-docker.pkg.dev/proj/repo/img@sha256:" + "c" * 64
        )
        assert auth.username == "_token"
        assert auth.password == "fake-helper-token"
        assert auth.server_address == "us-docker.pkg.dev"


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))
