"""Record complete model-toy states without changing observation or RNG state."""
from pathlib import Path

import torch

from .contracts import file_hash
from .state import state_digest


def checkpoint_observer(fixture, directory, receipts):
    """Wrap the existing observer; reject stateful evaluation and overwrites.

    State is captured after each genuine observation, including media-only
    observations. The caller's receipt owns the index; nothing is added to the
    fixture's checkpoint or random streams.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    observe = fixture.observe

    def recorded(*args, **kwargs):
        before = state_digest(fixture.state_dict())
        result = observe(*args, **kwargs)
        state = fixture.state_dict()
        if state_digest(state) != before:
            raise ValueError("model toy evaluation changed training state before checkpoint")
        path = directory / f"step-{fixture.completed_steps:06d}.pt"
        with path.open("xb") as stream:
            torch.save(state, stream)
        receipts.append(dict(step=fixture.completed_steps, path=str(path.resolve()),
                             sha256=file_hash(path), state_sha256=before))
        return result

    return recorded
