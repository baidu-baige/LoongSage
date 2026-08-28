from unittest.mock import MagicMock, patch
import pytest
from omegaconf import OmegaConf

from coda.resource_scheduler import ResourceScheduler


def _make_config(
    *,
    colocate: bool = True,
    trainer_gpus_per_node: int = 1,
    trainer_num_nodes: int = 1,
    rollout_gpus_per_node: int = 1,
    sglang_replicas: dict | None = None,
    opd: dict | None = None,
    log_level: str = "INFO",
):
    """Build a DictConfig matching the ResourceScheduler API."""
    cfg = {
        "log_level": log_level,
        "colocate": colocate,
        "trainer": {
            "num_gpus_per_node": trainer_gpus_per_node,
            "num_nodes": trainer_num_nodes,
        },
        "rollout": {
            "backend": "sglang",
            "num_gpus_per_node": rollout_gpus_per_node,
            "sglang_replicas": sglang_replicas if sglang_replicas is not None else {},
        },
    }
    if opd is not None:
        cfg["opd"] = opd
    return OmegaConf.create(cfg)


class TestInit:
    """Tests for ResourceScheduler.__init__ method."""

    @patch('coda.resource_scheduler.ResourceScheduler.create_placement_group')
    def test_init_default_values(self, mock_create_pg):
        """Test initialization with default values."""
        config = _make_config()
        scheduler = ResourceScheduler(config)

        assert scheduler.reorder_bundle_list == []
        assert scheduler.role_cursors == {}
        assert scheduler.colocate is True
        mock_create_pg.assert_called_once_with(1)

    @patch('coda.resource_scheduler.ResourceScheduler.create_placement_group')
    def test_init_with_collocate(self, mock_create_pg):
        """Test initialization with colocate=True."""
        config = _make_config(
            colocate=True,
            trainer_gpus_per_node=4,
            trainer_num_nodes=1,
        )
        scheduler = ResourceScheduler(config)

        assert scheduler.colocate is True
        mock_create_pg.assert_called_once_with(4)

    @patch('coda.resource_scheduler.ResourceScheduler.create_placement_group')
    def test_init_without_collocate(self, mock_create_pg):
        """Test initialization with colocate=False."""
        config = _make_config(
            colocate=False,
            trainer_gpus_per_node=2,
            trainer_num_nodes=2,
            rollout_gpus_per_node=4,
            sglang_replicas={"r0": {"num_nodes": 1}},
        )
        scheduler = ResourceScheduler(config)

        assert scheduler.colocate is False
        # 2 * 2 (trainer) + 1 * 4 (rollout replica) = 8 GPUs total
        mock_create_pg.assert_called_once_with(8)


class TestGetBundleIpAndGpu:
    """Tests for ResourceScheduler._get_bundle_ip_and_gpu method."""

    def _create_test_config(self):
        """Create a test config for ResourceScheduler."""
        return _make_config()

    @patch('coda.resource_scheduler.resource_scheduler.Probe')
    @patch('coda.resource_scheduler.resource_scheduler.ray')
    @patch('coda.resource_scheduler.resource_scheduler.PlacementGroupSchedulingStrategy')
    @patch('coda.resource_scheduler.resource_scheduler.ResourceScheduler.create_placement_group')
    def test_get_bundle_ip_and_gpu(self, mock_create_pg, mock_pg_strategy, mock_ray, mock_probe):
        """Test getting IP and GPU ID for a bundle."""
        config = self._create_test_config()
        scheduler = ResourceScheduler(config)
        mock_pg = MagicMock()

        # Mock the Probe class and its methods
        mock_probe_instance = MagicMock()
        mock_probe_instance.get_ip_and_gpu_id.remote.return_value = ("192.168.1.1", [0])

        mock_probe.options.return_value.remote.return_value = mock_probe_instance

        # Set up return values for ray.get
        mock_ray.get.return_value = ("192.168.1.1", [0])

        ip, gpu_id = scheduler._get_bundle_ip_and_gpu(mock_pg, 0)

        assert ip == "192.168.1.1"
        assert gpu_id == 0
        mock_pg_strategy.assert_called_once()

    @patch('coda.resource_scheduler.resource_scheduler.Probe')
    @patch('coda.resource_scheduler.resource_scheduler.ray')
    @patch('coda.resource_scheduler.resource_scheduler.PlacementGroupSchedulingStrategy')
    @patch('coda.resource_scheduler.resource_scheduler.ResourceScheduler.create_placement_group')
    def test_get_bundle_ip_and_gpu_no_cuda_visible(self, mock_create_pg, mock_pg_strategy, mock_ray, mock_probe):
        """Test getting GPU ID when CUDA_VISIBLE_DEVICES is not set."""
        config = self._create_test_config()
        scheduler = ResourceScheduler(config)
        mock_pg = MagicMock()

        # Create a mock Probe class
        mock_probe_instance = MagicMock()
        mock_probe_instance.get_ip_and_gpu_id.remote.return_value = ("192.168.1.2", [])

        # When Probe.options().remote() is called, return mock_probe_instance
        mock_probe.options.return_value.remote.return_value = mock_probe_instance

        # Set up return values for ray.get
        mock_ray.get.return_value = ("192.168.1.2", [])

        ip, gpu_id = scheduler._get_bundle_ip_and_gpu(mock_pg, 0)

        assert ip == "192.168.1.2"
        assert gpu_id == -1

    @patch('coda.resource_scheduler.resource_scheduler.Probe')
    @patch('coda.resource_scheduler.resource_scheduler.ray')
    @patch('coda.resource_scheduler.resource_scheduler.PlacementGroupSchedulingStrategy')
    @patch('coda.resource_scheduler.resource_scheduler.ResourceScheduler.create_placement_group')
    def test_get_bundle_ip_and_gpu_with_multiple_gpus(self, mock_create_pg, mock_pg_strategy, mock_ray, mock_probe):
        """Test getting GPU ID when CUDA_VISIBLE_DEVICES has multiple GPUs."""
        config = self._create_test_config()
        scheduler = ResourceScheduler(config)
        mock_pg = MagicMock()

        # Create a mock Probe class
        mock_probe_instance = MagicMock()
        mock_probe_instance.get_ip_and_gpu_id.remote.return_value = ("192.168.1.3", [1])

        # When Probe.options().remote() is called, return mock_probe_instance
        mock_probe.options.return_value.remote.return_value = mock_probe_instance

        # Set up return values for ray.get
        mock_ray.get.return_value = ("192.168.1.3", [1])

        ip, gpu_id = scheduler._get_bundle_ip_and_gpu(mock_pg, 0)

        assert ip == "192.168.1.3"
        assert gpu_id == 1


class TestCreatePlacementGroup:
    """Tests for ResourceScheduler.create_placement_group method."""

    def _create_test_config(self):
        """Create a test config for ResourceScheduler."""
        return _make_config(
            colocate=False,
            sglang_replicas={"r0": {"num_nodes": 1}},
        )

    def _scheduler_without_pg(self):
        """Build a ResourceScheduler whose __init__ does not create a real pg."""
        with patch('coda.resource_scheduler.resource_scheduler.ResourceScheduler.create_placement_group'):
            return ResourceScheduler(self._create_test_config())

    def test_create_placement_group_sorts_bundles_by_ip_then_gpu_id(self):
        """create_placement_group must order bundles by (ip, gpu_id).

        The bundle probe returns them out of order; the resulting
        reorder_bundle_list is what schedule() indexes into, so the ordering is
        the contract under test.
        """
        scheduler = self._scheduler_without_pg()

        mock_pg = MagicMock()
        probe_results = [
            ("192.168.1.2", 1),
            ("192.168.1.1", 1),
            ("192.168.1.2", 0),
            ("192.168.1.1", 0),
        ]

        with patch(
            'coda.resource_scheduler.resource_scheduler.placement_group',
            return_value=mock_pg,
        ), patch('coda.resource_scheduler.resource_scheduler.ray'), patch.object(
            ResourceScheduler, '_get_bundle_ip_and_gpu', side_effect=probe_results
        ):
            scheduler.create_placement_group(4)

        assert [(b["ip"], b["gpu_id"]) for b in scheduler.reorder_bundle_list] == [
            ("192.168.1.1", 0),
            ("192.168.1.1", 1),
            ("192.168.1.2", 0),
            ("192.168.1.2", 1),
        ]
        # p_idx must keep pointing at the original bundle position after sorting,
        # otherwise PlacementGroupSchedulingStrategy targets the wrong bundle.
        assert [b["p_idx"] for b in scheduler.reorder_bundle_list] == [3, 1, 2, 0]
        assert all(b["pg"] is mock_pg for b in scheduler.reorder_bundle_list)

    def test_create_placement_group_sorts_ip_numerically_not_lexicographically(self):
        """IP octets must sort numerically, so .9 comes before .84 and .110.

        Plain string sorting yields .110 < .84 < .9, which scrambles node ordering.
        Anything derived from that ordering breaks with it -- notably multi-node
        inference replicas, which pair adjacent nodes in reorder_bundle_list.
        """
        scheduler = self._scheduler_without_pg()

        probe_results = [
            ("10.0.0.110", 0),
            ("10.0.0.9", 0),
            ("10.0.0.84", 0),
        ]

        with patch(
            'coda.resource_scheduler.resource_scheduler.placement_group',
            return_value=MagicMock(),
        ), patch('coda.resource_scheduler.resource_scheduler.ray'), patch.object(
            ResourceScheduler, '_get_bundle_ip_and_gpu', side_effect=probe_results
        ):
            scheduler.create_placement_group(3)

        assert [b["ip"] for b in scheduler.reorder_bundle_list] == [
            "10.0.0.9",
            "10.0.0.84",
            "10.0.0.110",
        ]

    def test_create_placement_group_sorts_non_ipv4_addresses_last(self):
        """Non-IPv4 addresses must still sort deterministically instead of raising."""
        scheduler = self._scheduler_without_pg()

        probe_results = [
            ("some-hostname", 0),
            ("10.0.0.9", 0),
            ("fe80::1", 0),
        ]

        with patch(
            'coda.resource_scheduler.resource_scheduler.placement_group',
            return_value=MagicMock(),
        ), patch('coda.resource_scheduler.resource_scheduler.ray'), patch.object(
            ResourceScheduler, '_get_bundle_ip_and_gpu', side_effect=probe_results
        ):
            scheduler.create_placement_group(3)

        assert [b["ip"] for b in scheduler.reorder_bundle_list] == [
            "10.0.0.9",
            "fe80::1",
            "some-hostname",
        ]

    def test_create_placement_group_requests_one_gpu_bundle_each(self):
        """Each bundle must request exactly 1 CPU + 1 GPU with PACK strategy."""
        scheduler = self._scheduler_without_pg()

        with patch(
            'coda.resource_scheduler.resource_scheduler.placement_group',
            return_value=MagicMock(),
        ) as mock_placement_group, patch(
            'coda.resource_scheduler.resource_scheduler.ray'
        ), patch.object(
            ResourceScheduler,
            '_get_bundle_ip_and_gpu',
            return_value=("192.168.1.1", 0),
        ):
            scheduler.create_placement_group(3)

        mock_placement_group.assert_called_once_with(
            [{"CPU": 1, "GPU": 1}] * 3, strategy="PACK"
        )


class TestSchedule:
    """Tests for ResourceScheduler.schedule method."""

    def _create_test_config(self):
        """Create a test config for ResourceScheduler."""
        return _make_config(
            colocate=False,
            sglang_replicas={"r0": {"num_nodes": 1}},
        )

    def setup_method(self):
        """Set up common test fixtures."""
        with patch('coda.resource_scheduler.resource_scheduler.ResourceScheduler.create_placement_group'):
            config = self._create_test_config()
            self.scheduler = ResourceScheduler(config)
        # Manually set up the reorder_bundle_list to avoid calling create_placement_group
        mock_pg = MagicMock()
        self.scheduler.reorder_bundle_list = [
            {"pg": mock_pg, "p_idx": 0, "ip": "192.168.1.1", "gpu_id": 0},
            {"pg": mock_pg, "p_idx": 1, "ip": "192.168.1.1", "gpu_id": 1},
            {"pg": mock_pg, "p_idx": 2, "ip": "192.168.1.2", "gpu_id": 0},
        ]

    @patch('coda.resource_scheduler.resource_scheduler.PlacementGroupSchedulingStrategy')
    def test_schedule_single_actor_default_mode(self, mock_pg_strategy):
        """Test scheduling a single actor with default (non-collocate) mode."""
        # Create mock actor class
        mock_actor = MagicMock()
        mock_actor.__name__ = "TestActor"
        mock_actor.options.return_value = mock_actor

        mock_pg_strategy.return_value = "mock_strategy"

        prepared_actor, bundle_index = self.scheduler.schedule(mock_actor, num_bundles=1)

        assert prepared_actor is mock_actor
        assert bundle_index == 0

        # Verify cursor was incremented
        assert self.scheduler.role_cursors["_global"] == 1

        # Verify actor.options was called
        mock_actor.options.assert_called_once()
        call_kwargs = mock_actor.options.call_args[1]
        assert call_kwargs["num_cpus"] == 0.1
        assert call_kwargs["num_gpus"] == 0.1
        assert call_kwargs["scheduling_strategy"] == "mock_strategy"

        # Verify PlacementGroupSchedulingStrategy was called with correct args
        mock_pg_strategy.assert_called_once()
        call_kwargs = mock_pg_strategy.call_args[1]
        assert call_kwargs["placement_group"] == self.scheduler.reorder_bundle_list[0]["pg"]
        assert call_kwargs["placement_group_bundle_index"] == 0

    @patch('coda.resource_scheduler.resource_scheduler.PlacementGroupSchedulingStrategy')
    def test_schedule_multiple_actors(self, mock_pg_strategy):
        """Test scheduling multiple actors (num_bundles > 1)."""
        mock_actor = MagicMock()
        mock_actor.__name__ = "WorkerActor"
        mock_actor.options.return_value = mock_actor

        mock_pg_strategy.return_value = "mock_strategy"

        _actor, bundle_index = self.scheduler.schedule(mock_actor, num_bundles=2)

        # Both bundles come from one contiguous allocation starting at the cursor
        assert bundle_index == 0
        # Verify cursor was incremented by 2
        assert self.scheduler.role_cursors["_global"] == 2

    @patch('coda.resource_scheduler.resource_scheduler.PlacementGroupSchedulingStrategy')
    def test_schedule_collocate_mode(self, mock_pg_strategy):
        """Test scheduling in collocate mode (cursor resets)."""
        self.scheduler.colocate = True
        mock_actor1 = MagicMock()
        mock_actor1.__name__ = "Actor1"
        mock_actor1.options.return_value = mock_actor1

        mock_actor2 = MagicMock()
        mock_actor2.__name__ = "Actor2"
        mock_actor2.options.return_value = mock_actor2

        mock_pg_strategy.return_value = "mock_strategy"

        # Schedule first actor
        _a1, idx1 = self.scheduler.schedule(mock_actor1, num_bundles=1)

        # Schedule second actor - should start from cursor=0 again
        _a2, idx2 = self.scheduler.schedule(mock_actor2, num_bundles=1)

        # Each role has its own cursor in collocate mode, so both land on bundle 0
        assert (idx1, idx2) == (0, 0)
        assert self.scheduler.role_cursors["Actor1"] == 1
        assert self.scheduler.role_cursors["Actor2"] == 1

    @patch('coda.resource_scheduler.resource_scheduler.PlacementGroupSchedulingStrategy')
    def test_schedule_non_collocate_mode_continues_from_cursor(self, mock_pg_strategy):
        """Test scheduling in non-collocate mode (cursor continues)."""
        mock_actor1 = MagicMock()
        mock_actor1.__name__ = "Actor1"
        mock_actor1.options.return_value = mock_actor1

        mock_actor2 = MagicMock()
        mock_actor2.__name__ = "Actor2"
        mock_actor2.options.return_value = mock_actor2

        mock_pg_strategy.return_value = "mock_strategy"

        # Schedule first actor with 1 bundle
        _a1, idx1 = self.scheduler.schedule(mock_actor1, num_bundles=1)
        assert idx1 == 0
        assert self.scheduler.role_cursors["_global"] == 1

        # Schedule second actor - should start from cursor=1
        _a2, idx2 = self.scheduler.schedule(mock_actor2, num_bundles=1)
        assert idx2 == 1
        assert self.scheduler.role_cursors["_global"] == 2

    @patch('coda.resource_scheduler.resource_scheduler.PlacementGroupSchedulingStrategy')
    def test_schedule_existing_role(self, mock_pg_strategy):
        """Test scheduling the same actor type multiple times."""
        mock_actor = MagicMock()
        mock_actor.__name__ = "Worker"
        mock_actor.options.return_value = mock_actor

        mock_pg_strategy.return_value = "mock_strategy"

        # Schedule first instance
        _a1, idx1 = self.scheduler.schedule(mock_actor, num_bundles=1)
        assert idx1 == 0
        assert self.scheduler.role_cursors["_global"] == 1

        # Schedule second instance
        _a2, idx2 = self.scheduler.schedule(mock_actor, num_bundles=1)
        assert idx2 == 1
        assert self.scheduler.role_cursors["_global"] == 2

    def test_schedule_no_available_bundles(self):
        """Test scheduling when no bundles are available."""
        self.scheduler.reorder_bundle_list = []

        mock_actor = MagicMock()
        mock_actor.__name__ = "TestActor"

        with pytest.raises(RuntimeError, match="No available bundles to allocate for role 'TestActor'"):
            self.scheduler.schedule(mock_actor, num_bundles=1)

    @patch('coda.resource_scheduler.resource_scheduler.PlacementGroupSchedulingStrategy')
    def test_schedule_exhausts_available_bundles(self, mock_pg_strategy):
        """Test that allocating more than available bundles raises error."""
        mock_actor = MagicMock()
        mock_actor.__name__ = "TestActor"
        mock_actor.options.return_value = mock_actor

        mock_pg_strategy.return_value = "mock_strategy"

        # Try to allocate 4 bundles when only 3 are available
        with pytest.raises(RuntimeError, match="No available bundles to allocate"):
            self.scheduler.schedule(mock_actor, num_bundles=4)

    @patch('coda.resource_scheduler.resource_scheduler.PlacementGroupSchedulingStrategy')
    def test_schedule_uses_primary_bundle(self, mock_pg_strategy):
        """Test that scheduling uses the first allocated bundle as primary."""
        mock_actor = MagicMock()
        mock_actor.__name__ = "TestActor"
        mock_actor.options.return_value = mock_actor

        mock_pg_strategy.return_value = MagicMock()

        # Schedule 2 bundles
        _actor, bundle_index = self.scheduler.schedule(mock_actor, num_bundles=2)

        # Verify PlacementGroupSchedulingStrategy uses bundle 0 (first allocated)
        call_kwargs = mock_pg_strategy.call_args[1]
        assert call_kwargs["placement_group_bundle_index"] == 0
        assert bundle_index == 0


class TestIntegration:
    """Integration tests for ResourceScheduler."""

    def _create_test_config(self, num_gpus=3, colocate=False):
        """Create a test config for ResourceScheduler."""
        return _make_config(
            colocate=colocate,
            sglang_replicas={"r0": {"num_nodes": 1}},
        )

    @patch('coda.resource_scheduler.resource_scheduler.ray')
    @patch('coda.resource_scheduler.resource_scheduler.placement_group')
    @patch('coda.resource_scheduler.resource_scheduler.PlacementGroupSchedulingStrategy')
    @patch('coda.resource_scheduler.resource_scheduler.ResourceScheduler.create_placement_group')
    def test_full_workflow(self, mock_create_pg, mock_pg_strategy, mock_pg_func, mock_ray):
        """Test full workflow: init, create PG, schedule actors."""
        # Setup
        mock_pg = MagicMock()
        mock_pg_func.return_value = mock_pg
        mock_pg_strategy.return_value = "mock_strategy"

        # Initialize scheduler
        config = self._create_test_config()
        scheduler = ResourceScheduler(config)

        # Manually set up reorder_bundle_list for the test
        scheduler.reorder_bundle_list = [
            {"pg": mock_pg, "p_idx": 0, "ip": "192.168.1.1", "gpu_id": 0},
            {"pg": mock_pg, "p_idx": 1, "ip": "192.168.1.1", "gpu_id": 1},
            {"pg": mock_pg, "p_idx": 2, "ip": "192.168.1.2", "gpu_id": 0},
        ]

        # Verify initialization created placement group
        assert len(scheduler.reorder_bundle_list) == 3

        # Schedule different actor types
        actor1 = MagicMock()
        actor1.__name__ = "Master"
        actor1.options.return_value = actor1

        actor2 = MagicMock()
        actor2.__name__ = "Worker"
        actor2.options.return_value = actor2

        scheduler.schedule(actor1, num_bundles=1)
        scheduler.schedule(actor2, num_bundles=2)

        # Verify state
        assert scheduler.role_cursors["_global"] == 3

    @patch('coda.resource_scheduler.resource_scheduler.ray')
    @patch('coda.resource_scheduler.resource_scheduler.placement_group')
    @patch('coda.resource_scheduler.resource_scheduler.PlacementGroupSchedulingStrategy')
    @patch('coda.resource_scheduler.resource_scheduler.ResourceScheduler.create_placement_group')
    def test_collocate_workflow(self, mock_create_pg, mock_pg_strategy, mock_pg_func, mock_ray):
        """Test workflow in collocate mode."""
        # Setup
        mock_pg = MagicMock()
        mock_pg_func.return_value = mock_pg
        mock_pg_strategy.return_value = "mock_strategy"

        # Initialize with colocate
        config = self._create_test_config(num_gpus=2, colocate=True)
        scheduler = ResourceScheduler(config)

        # Manually set up reorder_bundle_list for the test
        scheduler.reorder_bundle_list = [
            {"pg": mock_pg, "p_idx": 0, "ip": "192.168.1.1", "gpu_id": 0},
            {"pg": mock_pg, "p_idx": 1, "ip": "192.168.1.1", "gpu_id": 1},
        ]

        # Schedule multiple actor types on same bundles
        actor1 = MagicMock()
        actor1.__name__ = "RoleA"
        actor1.options.return_value = actor1

        actor2 = MagicMock()
        actor2.__name__ = "RoleB"
        actor2.options.return_value = actor2

        scheduler.schedule(actor1, num_bundles=1)
        scheduler.schedule(actor2, num_bundles=1)

        # Each role has its own cursor in collocate mode
        assert scheduler.role_cursors["RoleA"] == 1
        assert scheduler.role_cursors["RoleB"] == 1


class TestProbeGetFreePort:
    """Tests for Probe.get_free_port.

    ``Probe`` is decorated with ``@ray.remote``, so the plain Python class is
    reached through ``__ray_actor_class__``; that lets the real method run
    in-process with ``socket.socket`` patched, no Ray runtime needed.
    """

    @staticmethod
    def _probe():
        from coda.resource_scheduler.resource_scheduler import Probe

        return Probe.__ray_actor_class__()

    @staticmethod
    def _socket_patch(bind_side_effect):
        """Patch the socket used by get_free_port; bind() drives the outcome."""
        mock_socket = MagicMock()
        mock_socket.bind.side_effect = bind_side_effect
        ctx = patch("coda.resource_scheduler.resource_scheduler.socket.socket")
        return ctx, mock_socket

    def test_returns_requested_number_of_consecutive_ports(self):
        """All binds succeed -> port_num consecutive ports starting at randint."""
        ctx, mock_socket = self._socket_patch(None)
        with ctx as mock_socket_class, patch(
            "coda.resource_scheduler.resource_scheduler.random.randint",
            return_value=21000,
        ) as mock_randint:
            mock_socket_class.return_value.__enter__.return_value = mock_socket
            ports = self._probe().get_free_port(
                port_num=3, min_port=20000, max_port=30000
            )

        assert ports == [21000, 21001, 21002]
        # start_port must leave room for port_num ports.
        mock_randint.assert_called_with(20000, 30000 - 3 + 1)

    def test_retries_a_new_start_port_when_a_port_is_taken(self):
        """A busy port aborts the run and a fresh start_port is drawn."""
        # First candidate 21000 is busy; second candidate 22000 is free.
        ctx, mock_socket = self._socket_patch(
            [OSError("in use"), None, None]
        )
        with ctx as mock_socket_class, patch(
            "coda.resource_scheduler.resource_scheduler.random.randint",
            side_effect=[21000, 22000],
        ):
            mock_socket_class.return_value.__enter__.return_value = mock_socket
            ports = self._probe().get_free_port(
                port_num=2, min_port=20000, max_port=30000
            )

        assert ports == [22000, 22001]

    def test_raises_after_max_tries_when_every_port_is_busy(self):
        """Exhausting max_tries must raise, and the message must report both."""
        ctx, mock_socket = self._socket_patch(OSError("in use"))
        with ctx as mock_socket_class, patch(
            "coda.resource_scheduler.resource_scheduler.random.randint",
            return_value=20000,
        ):
            mock_socket_class.return_value.__enter__.return_value = mock_socket
            with pytest.raises(
                RuntimeError,
                match=r"Unable to find 1 consecutive free port\(s\) in range "
                r"\[20000-20001\] after 5 attempts",
            ):
                self._probe().get_free_port(
                    port_num=1, min_port=20000, max_port=20001, max_tries=5
                )

        assert mock_socket.bind.call_count == 5


class TestGetGlooMasterAddress:
    """Tests for ResourceScheduler.get_gloo_master_address method."""

    def _create_test_config(self):
        """Create a test config for ResourceScheduler."""
        return _make_config()

    def setup_method(self):
        """Set up common test fixtures."""
        with patch('coda.resource_scheduler.resource_scheduler.ResourceScheduler.create_placement_group'):
            config = self._create_test_config()
            self.scheduler = ResourceScheduler(config)

        # Manually set up the reorder_bundle_list
        mock_pg = MagicMock()
        self.scheduler.reorder_bundle_list = [
            {"pg": mock_pg, "p_idx": 0, "ip": "192.168.1.1", "gpu_id": 0},
            {"pg": mock_pg, "p_idx": 1, "ip": "192.168.1.2", "gpu_id": 0},
        ]

    @patch('coda.resource_scheduler.resource_scheduler.Probe')
    @patch('coda.resource_scheduler.resource_scheduler.ray')
    def test_get_gloo_master_address_default_bundle(self, mock_ray, mock_probe):
        """Test getting gloo master address with default bundle index."""
        # Mock the probe flow
        mock_ports_ref = MagicMock()
        mock_probe_instance = MagicMock()
        mock_probe_instance.get_free_port.remote.return_value = mock_ports_ref
        mock_probe.options.return_value.remote.return_value = mock_probe_instance
        # The code always requests 3 consecutive ports for gloo init
        mock_ray.wait.return_value = ([mock_ports_ref], [])
        mock_ray.get.return_value = [25000, 25001, 25002]

        ip, port = self.scheduler.get_gloo_master_address()

        assert ip == "192.168.1.1"
        assert port == 25000
        # Verify kill was called
        assert mock_ray.kill.called

    @patch('coda.resource_scheduler.resource_scheduler.Probe')
    @patch('coda.resource_scheduler.resource_scheduler.ray')
    def test_get_gloo_master_address_specific_bundle(self, mock_ray, mock_probe):
        """Test getting gloo master address with specific bundle index."""
        mock_ports_ref = MagicMock()
        mock_probe_instance = MagicMock()
        mock_probe_instance.get_free_port.remote.return_value = mock_ports_ref
        mock_probe.options.return_value.remote.return_value = mock_probe_instance
        mock_ray.wait.return_value = ([mock_ports_ref], [])
        mock_ray.get.return_value = [30000, 30001, 30002]

        ip, port = self.scheduler.get_gloo_master_address(bundle_idx=1)

        assert ip == "192.168.1.2"
        assert port == 30000

    @patch('coda.resource_scheduler.resource_scheduler.Probe')
    @patch('coda.resource_scheduler.resource_scheduler.ray')
    def test_get_gloo_master_address_forwards_custom_port_range(self, mock_ray, mock_probe):
        """The custom range must reach Probe.get_free_port, not just the return value.

        ``port`` comes straight from the mocked ray.get, so asserting a range on
        it proves nothing; the contract is that min_port/max_port are forwarded.
        """
        mock_ports_ref = MagicMock()
        mock_probe_instance = MagicMock()
        mock_probe.options.return_value.remote.return_value = mock_probe_instance
        mock_probe_instance.get_free_port.remote.return_value = mock_ports_ref
        mock_ray.wait.return_value = ([mock_ports_ref], [])
        mock_ray.get.return_value = [28000, 28001, 28002]

        ip, port = self.scheduler.get_gloo_master_address(min_port=20000, max_port=30000)

        assert ip == "192.168.1.1"
        assert port == 28000
        # transfer_mesh gloo init needs 3 consecutive ports out of the given range.
        mock_probe_instance.get_free_port.remote.assert_called_once_with(3, 20000, 30000)

    def test_get_gloo_master_address_empty_bundle_list(self):
        """Test that get_gloo_master_address raises when bundle list is empty."""
        self.scheduler.reorder_bundle_list = []

        with pytest.raises(RuntimeError, match="Placement group not initialized"):
            self.scheduler.get_gloo_master_address()

    def test_get_gloo_master_address_invalid_bundle_idx(self):
        """Test that get_gloo_master_address raises when bundle index is out of range."""
        with pytest.raises(RuntimeError, match="Bundle index 10 out of range"):
            self.scheduler.get_gloo_master_address(bundle_idx=10)

    @patch('coda.resource_scheduler.resource_scheduler.Probe')
    @patch('coda.resource_scheduler.resource_scheduler.ray')
    @patch('coda.resource_scheduler.resource_scheduler.logging')
    def test_get_gloo_master_address_probe_failure(self, mock_logging, mock_ray, mock_probe):
        """Test that get_gloo_master_address retries on probe failure."""
        mock_ports_ref = MagicMock()
        mock_probe_instance = MagicMock()
        mock_probe_instance.get_free_port.remote.return_value = mock_ports_ref
        mock_probe.options.return_value.remote.return_value = mock_probe_instance
        mock_ray.wait.return_value = ([mock_ports_ref], [])
        # First two attempts fail, third succeeds
        mock_ray.get.side_effect = [
            Exception("Connection failed"),
            Exception("Connection failed"),
            [28000, 28001, 28002]
        ]

        ip, port = self.scheduler.get_gloo_master_address()

        assert ip == "192.168.1.1"
        assert port == 28000
        # Verify kill was called for each attempt
        assert mock_ray.kill.call_count == 3

    @patch('coda.resource_scheduler.resource_scheduler.Probe')
    @patch('coda.resource_scheduler.resource_scheduler.ray')
    @patch('coda.resource_scheduler.resource_scheduler.logging')
    def test_get_gloo_master_address_all_attempts_fail(self, mock_logging, mock_ray, mock_probe):
        """Test that get_gloo_master_address raises after all retries fail."""
        mock_ray.get.side_effect = Exception("Connection failed")
        mock_probe_instance = MagicMock()
        mock_probe.options.return_value.remote.return_value = mock_probe_instance

        with pytest.raises(RuntimeError, match="Failed to get gloo master address"):
            self.scheduler.get_gloo_master_address()
