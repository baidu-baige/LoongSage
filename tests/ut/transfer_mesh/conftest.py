"""Shared fixtures for transfer_mesh unit tests.

The ``cuda`` / ``multi_gpu`` markers are registered in pyproject.toml and the
matching skip decorators live in tests/conftest.py, so they are defined once for
the whole suite.
"""

import pytest


@pytest.fixture
def sample_rank_info_colocated():
    """8 senders + 8 receivers, each sender-receiver pair on the same GPU.

    Simulates a typical 8-GPU node where senders occupy ranks 0-7 and
    receivers occupy ranks 8-15, with rank i and rank i+8 sharing GPU i.

    Returns both the RankInfo list (for detect_topology / build_full_topology)
    and gathered dicts (for partition_receivers).
    """
    from coda.transfer_mesh.topology import RankInfo, Role

    infos = []
    gathered: list[dict[str, str | int]] = []
    ip = "10.0.0.1"
    for i in range(8):
        infos.append(RankInfo(rank=i, gpu_id=i, ip=ip, role=Role.SENDER))
        gathered.append({"rank": i, "role": "SENDER", "gpu_id": i, "ip": ip})
    for i in range(8):
        infos.append(RankInfo(rank=i + 8, gpu_id=i, ip=ip, role=Role.RECEIVER))
        gathered.append({"rank": i + 8, "role": "RECEIVER", "gpu_id": i, "ip": ip})
    return infos, gathered
