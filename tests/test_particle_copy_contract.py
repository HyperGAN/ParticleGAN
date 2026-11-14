"""Independent copies reject invalid row owners before publishing coordinates."""
from copy import deepcopy

import pytest
import torch
from test_e22_row_contract import _rows
from test_feature_portability import make, owned_defaults  # noqa: F401

from particlegan.birth_death import ParticleBirthDeath


def copy_owner(backend):
    if backend == "knn":
        rows = _rows()
        controller = ParticleBirthDeath(rows, seed=0)
        return (rows.table, rows.averaged_table, rows.optimizer, controller,
                lambda child, parent: controller._move(child, parent))
    trainer = make(backend="feature_cells")
    controller = trainer.birth_death
    return (trainer.prior.z, trainer.ema_prior.z, trainer.opt_g, controller,
            lambda child, parent: controller._move(trainer, child, parent))


@pytest.mark.parametrize("backend", ["knn", "feature_cells"])
@pytest.mark.parametrize("fault", ["sparse_moment", "bad_history", "duplicate_child", "unpaired_rows"])
def test_invalid_copy_rejects_before_any_owner_or_stream_changes(backend, fault):
    table, average, optimizer, controller, move = copy_owner(backend)
    # Real Adam initialization, followed by a fault in an owner that used to be
    # read only after fast and averaged coordinates had already been changed.
    table.square().sum().backward()
    optimizer.step()
    if fault == "sparse_moment":
        optimizer.state[table]["exp_avg_sq"] = optimizer.state[table]["exp_avg_sq"].to_sparse()
    if fault == "bad_history":
        optimizer.latent_history = torch.zeros(1, table.shape[1], dtype=table.dtype)
    child, parent = torch.tensor([0, 1]), torch.tensor([8, 9])
    if fault == "duplicate_child":
        child = torch.tensor([0, 0])
    if fault == "unpaired_rows":
        parent = parent[:1]
    before = (table.detach().clone(), average.clone(), deepcopy(optimizer.state[table]),
              deepcopy(getattr(optimizer, "latent_history", None)), controller.stream.get_state().clone())
    lineage = getattr(controller, "lineage", None)
    neighbors = None if lineage is None else lineage.neighbors.clone()

    with pytest.raises(ValueError, match="particle row copy"):
        move(child, parent)

    assert torch.equal(table, before[0])
    assert torch.equal(average, before[1])
    for key, value in optimizer.state[table].items():
        if isinstance(value, torch.Tensor):
            assert value.layout == before[2][key].layout
            assert torch.equal(value.to_dense(), before[2][key].to_dense())
        else:
            assert value == before[2][key]
    history = getattr(optimizer, "latent_history", None)
    if before[3] is None:
        assert history is None
    else:
        assert torch.equal(history, before[3])
    assert torch.equal(controller.stream.get_state(), before[4])
    if lineage is not None:
        assert torch.equal(lineage.neighbors, neighbors)
        assert controller._copy_parent_rows is None


def test_feature_copy_preserves_overlapping_donors_geometry_history_and_draw_order():
    trainer = make(backend="feature_cells")
    table, average, optimizer, controller = (trainer.prior.z, trainer.ema_prior.z,
                                            trainer.opt_g, trainer.birth_death)
    table.square().sum().backward()
    optimizer.step()
    optimizer.latent_history = table.detach().clone() + 20
    # A non-table-shaped custom state remains global. Mixed precision row
    # histories remain supported: they need the table's device, not its dtype.
    optimizer.state[table]["custom_global"] = torch.tensor([7., 8., 9.])
    optimizer.state[table]["master_history"] = table.detach().double().clone() + 5
    with torch.no_grad():
        average.copy_((2 * table).cos() + 3)
    child, parent = torch.tensor([0, 1, 7]), torch.tensor([1, 0, 14])
    before, before_average = table.detach().clone(), average.clone()
    before_state, before_history = deepcopy(optimizer.state[table]), optimizer.latent_history.clone()
    stream = torch.Generator().set_state(controller.stream.get_state())
    noise = torch.randn(before[parent].shape, dtype=table.dtype, generator=stream)
    geometry = deepcopy(controller.latent_geometry)
    fast_delta = geometry.displacement(before[parent], trainer.prior,
                                       trainer.controller.latent_bandwidth, noise, rows=parent)
    average_delta = geometry.displacement(before_average[parent], trainer.ema_prior,
                                          trainer.controller.latent_bandwidth, noise, rows=parent)
    ambient = torch.get_rng_state().clone()

    controller._move(trainer, child, parent)

    assert torch.equal(table[child], before[parent] + fast_delta)
    assert torch.equal(average[child], before_average[parent] + average_delta)
    untouched = torch.ones(len(table), dtype=torch.bool)
    untouched[child] = False
    assert torch.equal(table[untouched], before[untouched])
    assert torch.equal(average[untouched], before_average[untouched])
    for key, value in optimizer.state[table].items():
        if value.shape == table.shape:
            assert torch.equal(value[child], before_state[key][parent])
            assert torch.equal(value[untouched], before_state[key][untouched])
        else:
            assert torch.equal(value, before_state[key])
    assert torch.equal(optimizer.latent_history[child], before_history[parent])
    assert torch.equal(optimizer.latent_history[untouched], before_history[untouched])
    assert torch.equal(controller.stream.get_state(), stream.get_state())
    assert torch.equal(torch.get_rng_state(), ambient)
    controller.lineage.validate(controller.lineage.neighbors)
