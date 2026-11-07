"""Shared pytest setup: test markers and the CI network guard.

Markers (registered in pyproject.toml):
  needs_data  needs gitignored local files (raw datasets, the processed Valencia run, build/).
              These tests also skip themselves when the files are missing.
  network     calls a real web service (Earth Search, Overpass, Anthropic).
  postgis     needs the PostGIS database: `docker compose up -d` locally, a service container
              in GitHub Actions.

CI runs `pytest -m "not needs_data and not network"`. On top of that, when the CI environment
variable is set (GitHub Actions sets CI=true), any test that is not marked `network` and still
tries to open an internet connection fails at once. That keeps CI fast and deterministic, and
catches a test that forgot to fake a web call. Connections to localhost (PostGIS) stay allowed.
"""

import os
import socket

import pytest

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}
_real_connect = socket.socket.connect


def _local_only_connect(sock, address):
    host = address[0] if isinstance(address, tuple) else address  # unix sockets: a path
    if sock.family == socket.AF_UNIX or host in LOCAL_HOSTS:
        return _real_connect(sock, address)
    raise RuntimeError(f"test tried to reach {address} in CI: fake it, or mark it `network`")


@pytest.fixture(autouse=True)
def _no_internet_in_ci(request, monkeypatch):
    if os.environ.get("CI") and request.node.get_closest_marker("network") is None:
        monkeypatch.setattr(socket.socket, "connect", _local_only_connect)
