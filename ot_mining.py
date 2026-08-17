"""Numerically stable dense balanced OT utilities for the O1 experiment.

The solver deliberately has no dependency on the training/memory-bank code.  It
operates on a cost matrix and returns a transport plan plus diagnostics, which
makes the small O1 experiment reproducible before a sparse UOT miner is added.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


_EPS = 1e-12


@dataclass
class BalancedSinkhornResult:
    """Result of :func:`balanced_sinkhorn_log`."""

    plan: torch.Tensor
    row_marginal: torch.Tensor
    col_marginal: torch.Tensor
    iterations: int
    converged: bool
    max_row_error: float
    max_col_error: float
    objective: float
    final_objective_relative_delta: float
    final_log_scaling_delta: float


def _validate_problem(
    cost: torch.Tensor,
    a: Optional[torch.Tensor],
    b: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(cost, torch.Tensor) or cost.ndim != 2:
        raise ValueError("cost must be a rank-2 torch.Tensor")
    if cost.numel() == 0:
        raise ValueError("cost must be non-empty")
    if not torch.is_floating_point(cost):
        cost = cost.float()
    else:
        cost = cost.to(dtype=torch.float32)
    if not torch.isfinite(cost).all():
        raise ValueError("cost contains NaN or Inf")

    n_rows, n_cols = cost.shape
    if a is None:
        a = torch.full((n_rows,), 1.0 / n_rows, dtype=cost.dtype, device=cost.device)
    else:
        a = torch.as_tensor(a, dtype=cost.dtype, device=cost.device).flatten()
    if b is None:
        b = torch.full((n_cols,), 1.0 / n_cols, dtype=cost.dtype, device=cost.device)
    else:
        b = torch.as_tensor(b, dtype=cost.dtype, device=cost.device).flatten()
    if a.numel() != n_rows or b.numel() != n_cols:
        raise ValueError("marginals must match cost dimensions")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("marginals contain NaN or Inf")
    if (a <= 0).any() or (b <= 0).any():
        raise ValueError("marginals must be strictly positive")
    total_a = a.sum()
    total_b = b.sum()
    if not torch.isclose(total_a, total_b, atol=1e-6, rtol=1e-6):
        raise ValueError("row and column marginals must have the same total mass")
    # Normalizing both sides makes the returned plan a probability plan while
    # preserving the requested relative masses.  This also handles integer-like
    # input marginals without silently changing their ratio.
    total = (total_a + total_b) / 2
    return cost, a / total, b / total


def transport_objective(plan: torch.Tensor, cost: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Return <P,C> + epsilon * sum(P * (log(P)-1))."""
    if plan.shape != cost.shape:
        raise ValueError("plan and cost must have the same shape")
    positive = plan > 0
    entropy_term = torch.where(
        positive,
        plan * (torch.log(plan.clamp_min(_EPS)) - 1.0),
        torch.zeros_like(plan),
    ).sum()
    return (plan * cost).sum() + float(epsilon) * entropy_term


def balanced_sinkhorn_log(
    cost: torch.Tensor,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    epsilon: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-3,
    patience: int = 5,
) -> BalancedSinkhornResult:
    """Solve balanced entropic OT using log-domain dual updates.

    ``cost`` is kept in FP32 intentionally.  Convergence requires the marginal
    residual, relative objective change, and log-scaling change to stay below
    ``tol`` for ``patience`` consecutive iterations.  The implementation is
    dense by design and targets the 512/1,000-sample O1 correctness study.
    """
    if epsilon <= 0 or not torch.isfinite(torch.tensor(float(epsilon))):
        raise ValueError("epsilon must be finite and positive")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if tol <= 0 or patience < 1:
        raise ValueError("tol must be positive and patience must be at least 1")

    cost, a, b = _validate_problem(cost, a, b)
    log_a = torch.log(a)
    log_b = torch.log(b)
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    stable_steps = 0
    converged = False
    objective_relative_delta = float("inf")
    log_scaling_delta = float("inf")
    previous_objective: Optional[float] = None
    iterations = 0

    for iteration in range(1, max_iter + 1):
        old_f, old_g = f, g
        f = float(epsilon) * (log_a - torch.logsumexp((g[None, :] - cost) / epsilon, dim=1))
        g = float(epsilon) * (log_b - torch.logsumexp((f[:, None] - cost) / epsilon, dim=0))

        # Fix the otherwise arbitrary additive gauge.  Keeping potentials in a
        # similar numeric range improves stability for long runs.
        shift = f.mean()
        f = f - shift
        g = g + shift

        log_plan = (f[:, None] + g[None, :] - cost) / float(epsilon)
        plan = torch.exp(log_plan)
        row_error = (plan.sum(dim=1) - a).abs().max().item()
        col_error = (plan.sum(dim=0) - b).abs().max().item()
        log_scaling_delta = max(
            (f - old_f).abs().max().item(),
            (g - old_g).abs().max().item(),
        ) / float(epsilon)
        objective = (
            (plan * cost).sum()
            + float(epsilon) * (plan * (log_plan - 1.0)).sum()
        ).item()
        if previous_objective is not None:
            objective_relative_delta = abs(objective - previous_objective) / max(
                abs(previous_objective), _EPS
            )
        previous_objective = objective
        iterations = iteration
        if not torch.isfinite(plan).all():
            raise FloatingPointError("Sinkhorn produced NaN or Inf")
        if max(row_error, col_error, objective_relative_delta, log_scaling_delta) < tol:
            stable_steps += 1
            if stable_steps >= patience:
                converged = True
                break
        else:
            stable_steps = 0

    # Recompute from the final dual variables rather than returning a stale
    # intermediate plan from before the last update.
    plan = torch.exp((f[:, None] + g[None, :] - cost) / float(epsilon))
    if not torch.isfinite(plan).all() or (plan < 0).any():
        raise FloatingPointError("Sinkhorn produced an invalid transport plan")
    row_marginal = plan.sum(dim=1)
    col_marginal = plan.sum(dim=0)
    max_row_error = (row_marginal - a).abs().max().item()
    max_col_error = (col_marginal - b).abs().max().item()
    objective = transport_objective(plan, cost, epsilon).item()
    return BalancedSinkhornResult(
        plan=plan,
        row_marginal=row_marginal,
        col_marginal=col_marginal,
        iterations=iterations,
        converged=converged,
        max_row_error=max_row_error,
        max_col_error=max_col_error,
        objective=objective,
        final_objective_relative_delta=objective_relative_delta,
        final_log_scaling_delta=log_scaling_delta,
    )


def pot_sinkhorn_reference(
    cost: torch.Tensor,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    epsilon: float = 0.05,
    max_iter: int = 2_000,
    tol: float = 1e-9,
) -> torch.Tensor:
    """Solve the same balanced problem with POT's float64 log Sinkhorn.

    POT is an optional experiment dependency (``uv sync --extra ot``).  This
    function is a numerical reference for O1 and is not used by training or the
    memory-bank path.
    """
    try:
        import ot
    except ImportError as error:
        raise RuntimeError(
            "POT is not installed; run `uv sync --extra ot` inside the project virtual environment"
        ) from error
    if epsilon <= 0 or max_iter < 1 or tol <= 0:
        raise ValueError("epsilon, max_iter, and tol must be positive")
    cost, a, b = _validate_problem(cost, a, b)
    reference = ot.sinkhorn(
        a.detach().cpu().double().numpy(),
        b.detach().cpu().double().numpy(),
        cost.detach().cpu().double().numpy(),
        reg=float(epsilon),
        method="sinkhorn_log",
        numItermax=max_iter,
        stopThr=tol,
        warn=False,
    )
    return torch.from_numpy(reference).to(device=cost.device, dtype=torch.float64)


def gt_transport_mass(
    plan: torch.Tensor,
    gt_mask: torch.Tensor,
    source_marginal: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    """Return raw and source-normalized GT transport mass.

    ``gt_mask`` is diagnostic metadata only.  The normalized value averages the
    fraction of each source row's marginal assigned to its true targets, making
    it comparable across methods that use different row masses.
    """
    if plan.ndim != 2 or gt_mask.shape != plan.shape:
        raise ValueError("gt_mask must have the same rank-2 shape as plan")
    mask = gt_mask.to(device=plan.device, dtype=torch.bool)
    raw = plan.masked_select(mask).sum()
    if source_marginal is None:
        source_marginal = plan.sum(dim=1)
    source_marginal = source_marginal.to(device=plan.device, dtype=plan.dtype)
    per_source = plan.masked_fill(~mask, 0).sum(dim=1) / source_marginal.clamp_min(_EPS)
    return raw.item(), per_source.mean().item()


def marginal_errors(plan: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    """Compute maximum absolute row and column marginal errors."""
    return (
        (plan.sum(dim=1) - a.to(plan)).abs().max().item(),
        (plan.sum(dim=0) - b.to(plan)).abs().max().item(),
    )
