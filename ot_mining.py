"""Numerically stable OT utilities for the O1 and O2 experiments.

The solvers deliberately have no dependency on the training or memory-bank
code. O1 uses dense balanced OT, while O2 uses sparse unbalanced OT over a
bidirectional top-k candidate graph.
"""

from dataclasses import dataclass
import math
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


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


@dataclass
class SparseTransportGraph:
    row_index: torch.Tensor
    col_index: torch.Tensor
    cost: torch.Tensor
    row_count: int
    col_count: int


@dataclass
class SparseUOTResult:
    plan: torch.Tensor
    row_marginal: torch.Tensor
    col_marginal: torch.Tensor
    iterations: int
    converged: bool
    objective: float
    final_objective_relative_delta: float
    final_log_scaling_delta: float


@dataclass
class O2MiningResult:
    i2t_indices: torch.Tensor
    i2t_scores: torch.Tensor
    i2t_confidence: torch.Tensor
    t2i_indices: torch.Tensor
    t2i_scores: torch.Tensor
    t2i_confidence: torch.Tensor
    diagnostics: Dict[str, float]


def _segment_logsumexp(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    maxima = torch.full((size,), -torch.inf, dtype=values.dtype, device=values.device)
    maxima.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
    if not torch.isfinite(maxima).all():
        raise ValueError("each transport node must have at least one candidate edge")
    shifted = torch.exp(values - maxima[index])
    totals = torch.zeros(size, dtype=values.dtype, device=values.device)
    totals.scatter_add_(0, index, shifted)
    return maxima + torch.log(totals.clamp_min(_EPS))


def _generalized_kl(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return (
        values * torch.log((values + _EPS) / (reference + _EPS))
        - values
        + reference
    ).sum()


def _sparse_uot_objective(
    plan: torch.Tensor,
    cost: torch.Tensor,
    prior: torch.Tensor,
    row_marginal: torch.Tensor,
    row_target: torch.Tensor,
    col_marginal: torch.Tensor,
    col_target: torch.Tensor,
    epsilon: float,
    rho: float,
) -> torch.Tensor:
    outside_support_mass = (1.0 - prior.sum()).clamp_min(0.0)
    transport_kl = _generalized_kl(plan, prior) + outside_support_mass
    return (
        (plan * cost).sum()
        + float(epsilon) * transport_kl
        + float(rho) * _generalized_kl(row_marginal, row_target)
        + float(rho) * _generalized_kl(col_marginal, col_target)
    )


def _validate_sparse_graph(graph: SparseTransportGraph) -> SparseTransportGraph:
    row_index = torch.as_tensor(graph.row_index, dtype=torch.long)
    col_index = torch.as_tensor(graph.col_index, dtype=torch.long, device=row_index.device)
    cost = torch.as_tensor(graph.cost, dtype=torch.float32, device=row_index.device)
    if row_index.ndim != 1 or col_index.ndim != 1 or cost.ndim != 1:
        raise ValueError("sparse graph tensors must be rank-1")
    if not (row_index.numel() == col_index.numel() == cost.numel()) or cost.numel() == 0:
        raise ValueError("sparse graph tensors must be non-empty and have equal lengths")
    if graph.row_count < 1 or graph.col_count < 1:
        raise ValueError("transport graph dimensions must be positive")
    if row_index.min() < 0 or row_index.max() >= graph.row_count:
        raise ValueError("row index is outside the transport graph")
    if col_index.min() < 0 or col_index.max() >= graph.col_count:
        raise ValueError("column index is outside the transport graph")
    if not torch.isfinite(cost).all():
        raise ValueError("transport costs contain NaN or Inf")
    row_degree = torch.zeros(graph.row_count, dtype=torch.long, device=row_index.device)
    col_degree = torch.zeros(graph.col_count, dtype=torch.long, device=row_index.device)
    row_degree.scatter_add_(0, row_index, torch.ones_like(row_index))
    col_degree.scatter_add_(0, col_index, torch.ones_like(col_index))
    if (row_degree == 0).any() or (col_degree == 0).any():
        raise ValueError("each transport node must have at least one candidate edge")
    return SparseTransportGraph(
        row_index=row_index,
        col_index=col_index,
        cost=cost,
        row_count=int(graph.row_count),
        col_count=int(graph.col_count),
    )


def sparse_unbalanced_sinkhorn(
    graph: SparseTransportGraph,
    epsilon: float = 0.05,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-3,
    patience: int = 5,
) -> SparseUOTResult:
    """Solve entropy-regularized unbalanced OT on a sparse bipartite graph."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive")
    if max_iter < 1 or not math.isfinite(tol) or tol <= 0 or patience < 1:
        raise ValueError("invalid UOT stopping parameters")
    graph = _validate_sparse_graph(graph)
    row_index = graph.row_index
    col_index = graph.col_index
    cost = graph.cost
    row_marginal_target = torch.full(
        (graph.row_count,), 1.0 / graph.row_count, dtype=cost.dtype, device=cost.device
    )
    col_marginal_target = torch.full(
        (graph.col_count,), 1.0 / graph.col_count, dtype=cost.dtype, device=cost.device
    )
    log_row_target = torch.log(row_marginal_target)
    log_col_target = torch.log(col_marginal_target)
    log_prior = log_row_target[row_index] + log_col_target[col_index]
    log_kernel = log_prior - cost / float(epsilon)
    prior = torch.exp(log_prior)
    exponent = float(rho) / float(rho + epsilon)
    log_u = torch.zeros_like(row_marginal_target)
    log_v = torch.zeros_like(col_marginal_target)
    stable_steps = 0
    converged = False
    final_delta = float("inf")
    objective_relative_delta = float("inf")
    previous_objective: Optional[float] = None
    iterations = 0

    for iteration in range(1, max_iter + 1):
        previous_log_u = log_u
        previous_log_v = log_v
        log_kv = _segment_logsumexp(
            log_kernel + log_v[col_index], row_index, graph.row_count
        )
        log_u = exponent * (log_row_target - log_kv)
        log_ktu = _segment_logsumexp(
            log_kernel + log_u[row_index], col_index, graph.col_count
        )
        log_v = exponent * (log_col_target - log_ktu)
        final_delta = max(
            (log_u - previous_log_u).abs().max().item(),
            (log_v - previous_log_v).abs().max().item(),
        )
        iterations = iteration
        if not torch.isfinite(log_u).all() or not torch.isfinite(log_v).all():
            raise FloatingPointError("unbalanced Sinkhorn produced NaN or Inf scalings")
        iteration_plan = torch.exp(
            log_kernel + log_u[row_index] + log_v[col_index]
        )
        iteration_row_marginal = torch.zeros_like(row_marginal_target)
        iteration_col_marginal = torch.zeros_like(col_marginal_target)
        iteration_row_marginal.scatter_add_(0, row_index, iteration_plan)
        iteration_col_marginal.scatter_add_(0, col_index, iteration_plan)
        iteration_objective = _sparse_uot_objective(
            iteration_plan,
            cost,
            prior,
            iteration_row_marginal,
            row_marginal_target,
            iteration_col_marginal,
            col_marginal_target,
            epsilon,
            rho,
        ).item()
        if previous_objective is not None:
            objective_relative_delta = abs(
                iteration_objective - previous_objective
            ) / max(abs(previous_objective), _EPS)
        previous_objective = iteration_objective
        if max(final_delta, objective_relative_delta) < tol:
            stable_steps += 1
            if stable_steps >= patience:
                converged = True
                break
        else:
            stable_steps = 0

    log_plan = log_kernel + log_u[row_index] + log_v[col_index]
    plan = torch.exp(log_plan)
    if not torch.isfinite(plan).all() or (plan < 0).any():
        raise FloatingPointError("unbalanced Sinkhorn produced an invalid transport plan")
    row_marginal = torch.zeros_like(row_marginal_target)
    col_marginal = torch.zeros_like(col_marginal_target)
    row_marginal.scatter_add_(0, row_index, plan)
    col_marginal.scatter_add_(0, col_index, plan)
    objective = _sparse_uot_objective(
        plan,
        cost,
        prior,
        row_marginal,
        row_marginal_target,
        col_marginal,
        col_marginal_target,
        epsilon,
        rho,
    ).item()
    return SparseUOTResult(
        plan=plan,
        row_marginal=row_marginal,
        col_marginal=col_marginal,
        iterations=iterations,
        converged=converged,
        objective=objective,
        final_objective_relative_delta=objective_relative_delta,
        final_log_scaling_delta=final_delta,
    )


def _edge_similarities(
    image_embeddings: torch.Tensor,
    caption_embeddings: torch.Tensor,
    row_index: torch.Tensor,
    col_index: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    similarities = torch.empty(row_index.numel(), device=row_index.device)
    for start in range(0, row_index.numel(), chunk_size):
        end = min(start + chunk_size, row_index.numel())
        similarities[start:end] = (
            image_embeddings[row_index[start:end]] * caption_embeddings[col_index[start:end]]
        ).sum(dim=1)
    return similarities


def build_bidirectional_topk_graph(
    image_embeddings: torch.Tensor,
    caption_embeddings: torch.Tensor,
    candidate_k: int = 32,
    block_size: int = 1024,
    edge_chunk_size: int = 65536,
    device: Optional[torch.device] = None,
) -> SparseTransportGraph:
    """Build the union of I2T and T2I top-k cosine-similarity edges."""
    if image_embeddings.ndim != 2 or caption_embeddings.ndim != 2:
        raise ValueError("image and caption embeddings must be rank-2")
    if image_embeddings.shape[1] != caption_embeddings.shape[1]:
        raise ValueError("image and caption embedding dimensions must match")
    if image_embeddings.shape[0] == 0 or caption_embeddings.shape[0] == 0:
        raise ValueError("image and caption embeddings must be non-empty")
    if candidate_k < 1 or block_size < 1 or edge_chunk_size < 1:
        raise ValueError("candidate and block sizes must be positive")
    if device is None:
        device = image_embeddings.device
    image_embeddings = F.normalize(
        image_embeddings.to(device=device, dtype=torch.float32), dim=1
    )
    caption_embeddings = F.normalize(
        caption_embeddings.to(device=device, dtype=torch.float32), dim=1
    )
    if (
        not torch.isfinite(image_embeddings).all()
        or not torch.isfinite(caption_embeddings).all()
    ):
        raise ValueError("image and caption embeddings must contain only finite values")
    row_count = image_embeddings.shape[0]
    col_count = caption_embeddings.shape[0]
    i2t_k = min(candidate_k, col_count)
    t2i_k = min(candidate_k, row_count)
    row_parts = []
    col_parts = []

    for start in range(0, row_count, block_size):
        end = min(start + block_size, row_count)
        similarities = image_embeddings[start:end].mm(caption_embeddings.t())
        indices = similarities.topk(i2t_k, dim=1).indices
        rows = torch.arange(start, end, device=device).unsqueeze(1).expand_as(indices)
        row_parts.append(rows.reshape(-1))
        col_parts.append(indices.reshape(-1))
        del similarities

    for start in range(0, col_count, block_size):
        end = min(start + block_size, col_count)
        similarities = caption_embeddings[start:end].mm(image_embeddings.t())
        indices = similarities.topk(t2i_k, dim=1).indices
        columns = torch.arange(start, end, device=device).unsqueeze(1).expand_as(indices)
        row_parts.append(indices.reshape(-1))
        col_parts.append(columns.reshape(-1))
        del similarities

    row_index = torch.cat(row_parts)
    col_index = torch.cat(col_parts)
    edge_keys = row_index * col_count + col_index
    edge_keys = torch.unique(edge_keys, sorted=True)
    row_index = torch.div(edge_keys, col_count, rounding_mode="floor")
    col_index = edge_keys.remainder(col_count)
    similarities = _edge_similarities(
        image_embeddings,
        caption_embeddings,
        row_index,
        col_index,
        edge_chunk_size,
    )
    return SparseTransportGraph(
        row_index=row_index,
        col_index=col_index,
        cost=1.0 - similarities,
        row_count=row_count,
        col_count=col_count,
    )


def _segment_argmax(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    maxima = torch.full((size,), -torch.inf, dtype=values.dtype, device=values.device)
    maxima.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
    positions = torch.arange(values.numel(), device=values.device, dtype=torch.long)
    sentinel = torch.full_like(positions, values.numel())
    candidate_positions = torch.where(values == maxima[index], positions, sentinel)
    selected = torch.full((size,), values.numel(), device=values.device, dtype=torch.long)
    selected.scatter_reduce_(0, index, candidate_positions, reduce="amin", include_self=True)
    if (selected == values.numel()).any():
        raise ValueError("each transport node must have a selected edge")
    return selected


def _source_statistics(
    plan: torch.Tensor,
    source_index: torch.Tensor,
    source_count: int,
    target_index: torch.Tensor,
    target_mass: float,
    confidence_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_mass = torch.zeros(source_count, dtype=plan.dtype, device=plan.device)
    source_mass.scatter_add_(0, source_index, plan)
    conditional = plan / source_mass[source_index].clamp_min(_EPS)
    degree = torch.zeros(source_count, dtype=plan.dtype, device=plan.device)
    degree.scatter_add_(0, source_index, torch.ones_like(plan))
    entropy = torch.zeros(source_count, dtype=plan.dtype, device=plan.device)
    entropy.scatter_add_(
        0,
        source_index,
        -conditional * torch.log(conditional.clamp_min(_EPS)),
    )
    concentration = torch.ones_like(source_mass)
    multi_edge = degree > 1
    concentration[multi_edge] = 1.0 - entropy[multi_edge] / torch.log(degree[multi_edge])
    concentration = concentration.clamp(0.0, 1.0)
    relative_mass = (source_mass / float(target_mass)).clamp(0.0, 1.0)
    if confidence_mode == "row_mass":
        confidence = relative_mass
    elif confidence_mode == "concentration":
        confidence = concentration
    elif confidence_mode == "mass_concentration":
        confidence = relative_mass * concentration
    else:
        raise ValueError(f"unsupported OT confidence mode: {confidence_mode}")
    selected_edges = _segment_argmax(conditional, source_index, source_count)
    return (
        target_index[selected_edges],
        conditional[selected_edges],
        confidence.clamp(0.0, 1.0),
    )


@torch.no_grad()
def mine_o2_pairs(
    image_embeddings: torch.Tensor,
    caption_embeddings: torch.Tensor,
    candidate_k: int = 32,
    epsilon: float = 0.05,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-3,
    block_size: int = 1024,
    confidence_mode: str = "mass_concentration",
    device: Optional[torch.device] = None,
) -> O2MiningResult:
    """Mine one OT pseudo-pair and one continuous confidence per source node."""
    if device is None:
        device = image_embeddings.device
    device = torch.device(device)
    peak_gpu_memory_mb = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started_at = time.perf_counter()
    graph = build_bidirectional_topk_graph(
        image_embeddings=image_embeddings,
        caption_embeddings=caption_embeddings,
        candidate_k=candidate_k,
        block_size=block_size,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    graph_seconds = time.perf_counter() - started_at
    solver_started_at = time.perf_counter()
    transport = sparse_unbalanced_sinkhorn(
        graph,
        epsilon=epsilon,
        rho=rho,
        max_iter=max_iter,
        tol=tol,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    solver_seconds = time.perf_counter() - solver_started_at
    i2t_indices, i2t_scores, i2t_confidence = _source_statistics(
        plan=transport.plan,
        source_index=graph.row_index,
        source_count=graph.row_count,
        target_index=graph.col_index,
        target_mass=1.0 / graph.row_count,
        confidence_mode=confidence_mode,
    )
    t2i_indices, t2i_scores, t2i_confidence = _source_statistics(
        plan=transport.plan,
        source_index=graph.col_index,
        source_count=graph.col_count,
        target_index=graph.row_index,
        target_mass=1.0 / graph.col_count,
        confidence_mode=confidence_mode,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    diagnostics = {
        "candidate_edges": float(graph.cost.numel()),
        "candidate_k": float(candidate_k),
        "iterations": float(transport.iterations),
        "converged": float(transport.converged),
        "objective": float(transport.objective),
        "final_objective_relative_delta": float(
            transport.final_objective_relative_delta
        ),
        "final_log_scaling_delta": float(transport.final_log_scaling_delta),
        "graph_seconds": float(graph_seconds),
        "solver_seconds": float(solver_seconds),
        "peak_gpu_memory_mb": float(peak_gpu_memory_mb),
        "i2t_mean_confidence": float(i2t_confidence.mean().item()),
        "t2i_mean_confidence": float(t2i_confidence.mean().item()),
        "i2t_mean_top1_mass": float(i2t_scores.mean().item()),
        "t2i_mean_top1_mass": float(t2i_scores.mean().item()),
    }
    return O2MiningResult(
        i2t_indices=i2t_indices.detach().cpu(),
        i2t_scores=i2t_scores.detach().cpu(),
        i2t_confidence=i2t_confidence.detach().cpu(),
        t2i_indices=t2i_indices.detach().cpu(),
        t2i_scores=t2i_scores.detach().cpu(),
        t2i_confidence=t2i_confidence.detach().cpu(),
        diagnostics=diagnostics,
    )
