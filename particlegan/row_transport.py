"""Preparation for independent particle copies; routed splits own another law."""
import torch


@torch.no_grad()
def prepare_copy_history(table, optimizer, children, parents):
    """Validate row-local copy owners and snapshot their source incarnations.

    Call before publishing either fast or averaged coordinates or drawing copy
    jitter. Only table-shaped state belongs to this copy law; scalar/global
    optimizer state stays shared. This preparation is not an evaluation or
    optimizer-step rollback mechanism.
    """
    for rows in (children, parents):
        if (not isinstance(rows, torch.Tensor) or rows.ndim != 1 or rows.dtype != torch.long
                or rows.layout != torch.strided or rows.device != table.device
                or bool(((rows < 0) | (rows >= len(table))).any())):
            raise ValueError("particle row copy requires valid table row indices")
    if children.shape != parents.shape or len(torch.unique(children)) != len(children):
        raise ValueError("particle row copy requires distinct paired child rows")
    tensors = [value for value in optimizer.state.get(table, {}).values()
               if isinstance(value, torch.Tensor) and value.shape == table.shape]
    history = getattr(optimizer, "latent_history", None)
    if history is not None:
        if (not isinstance(history, torch.Tensor) or history.shape != table.shape
                or history.dtype != table.dtype):
            raise ValueError("particle row copy requires table-shaped latent history")
        tensors.append(history)
    if any(value.layout != torch.strided or value.device != table.device for value in tensors):
        raise ValueError("particle row copy requires dense row history on the table device")
    return [(value, value[parents].clone()) for value in tensors]
