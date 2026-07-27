"""Parameter-group helpers shared by the optimization solvers."""

from collections.abc import Iterable
import math

import torch


def freeze_parameters(
    parameters: Iterable[torch.nn.Parameter],
) -> None:
    """Disable autograd and clear any accumulated gradient."""
    for parameter in parameters:
        parameter.requires_grad_(False)
        parameter.grad = None


def active_parameters(
    parameters: Iterable[torch.nn.Parameter],
) -> list[torch.nn.Parameter]:
    """Return parameters that remain active for optimization."""
    return [parameter for parameter in parameters if parameter.requires_grad]


def clip_grad_groups(
    groups: Iterable[Iterable[torch.nn.Parameter]],
    max_norm: float,
) -> list[torch.Tensor]:
    """Clip each logical group containing active gradients independently."""
    norms = []
    for group in groups:
        parameters = [
            parameter
            for parameter in group
            if parameter.requires_grad and parameter.grad is not None
        ]
        if parameters:
            norms.append(torch.nn.utils.clip_grad_norm_(parameters, max_norm))
    return norms


def cosine_multiplier(
    step: int,
    total_steps: int,
    final_ratio: float = 0.1,
) -> float:
    """Cosine multiplier with exact capped start and end semantics."""
    if not math.isfinite(total_steps) or total_steps < 1:
        raise ValueError("total_steps must be finite and at least 1")
    if not math.isfinite(final_ratio) or not 0.0 <= final_ratio <= 1.0:
        raise ValueError("final_ratio must be finite and in [0, 1]")
    capped = min(max(step, 0), total_steps)
    phase = capped / total_steps
    return final_ratio + (1.0 - final_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * phase)
    )


def _consumed_cosine_multiplier(
    update_index: int,
    update_count: int,
    final_ratio: float = 0.1,
) -> float:
    """Map actual optimizer updates onto both cosine endpoints."""
    if update_count == 1:
        return cosine_multiplier(1, 1, final_ratio)
    return cosine_multiplier(update_index, update_count - 1, final_ratio)
