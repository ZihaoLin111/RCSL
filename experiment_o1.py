"""Run O1: dense balanced Sinkhorn on a deterministic unpaired subset.

Example (on a machine with the repository's data/checkpoint assets)::

    python experiment_o1.py --checkpoint runs/.../checkpoint_24.pth.tar \
        --data-path ./data --vocab-path ./vocab --subset-size 512 \
        --output-dir runs/o1

The script does not modify training or the existing MNN memory bank.  Synthetic
ground truth is used only for post-hoc diagnostics.
"""

import argparse
import json
import os
import platform
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

import data
from model import SVSE
from ot_mining import balanced_sinkhorn_log, gt_transport_mass, pot_sinkhorn_reference, transport_objective
from vocab import deserialize_vocab, deserialize_vocab_glove


def _checkpoint_options(checkpoint: Dict[str, Any], args: argparse.Namespace) -> Any:
    opt = checkpoint.get("opt")
    if opt is None:
        raise KeyError("checkpoint does not contain the training options under 'opt'")
    for name, value in {
        "data_path": args.data_path,
        "vocab_path": args.vocab_path,
        "data_name": args.data_name or getattr(opt, "data_name", "f30k_precomp"),
        "stage": "learning",
        "caption_enhance": False,
        "img_enhance": False,
        "seed": args.seed if args.seed is not None else getattr(opt, "seed", 42),
        "paired_length": args.paired_length if args.paired_length is not None else getattr(opt, "paired_length", -1),
    }.items():
        setattr(opt, name, value)
    return opt


def _load_model(checkpoint: Dict[str, Any], opt: Any, vocab: Any, device: torch.device) -> SVSE:
    word2idx = vocab.word2idx if getattr(opt, "init_txt", "uniform") == "glove" else None
    opt.vocab_size = len(vocab)
    model = SVSE(opt, word2idx)
    image_state, text_state = checkpoint["model"]
    has_parallel_prefix = any(key.startswith("module.") for key in image_state)
    if has_parallel_prefix != any(key.startswith("module.") for key in text_state):
        raise ValueError("image and text checkpoint states disagree on DataParallel wrapping")
    if has_parallel_prefix:
        model.make_data_parallel()
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.val_start()
    return model


def _select_unpaired_groups(dataset: Any, subset_size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if subset_size < 1:
        raise ValueError("subset_size must be positive")
    if dataset.im_div < 1 or dataset.length % dataset.im_div != 0:
        raise ValueError("caption count must be divisible by dataset.im_div")
    image_count = dataset.img_length
    if dataset.old_length != image_count * dataset.im_div:
        raise ValueError("O1 requires a fixed number of captions per image group")
    image_ids = np.arange(image_count)
    unpaired_groups = image_ids[dataset.shuffle_inx != image_ids]
    if subset_size > len(unpaired_groups):
        raise ValueError(
            f"requested {subset_size} unpaired image groups, but only {len(unpaired_groups)} are available"
        )
    rng = np.random.RandomState(seed)
    selected_image_ids = np.sort(rng.permutation(unpaired_groups)[:subset_size]).astype(np.int64)
    caption_ids = np.concatenate(
        [np.arange(image_id * dataset.im_div, (image_id + 1) * dataset.im_div) for image_id in selected_image_ids]
    ).astype(np.int64)
    return selected_image_ids, caption_ids


def _encode_subset(
    dataset: Any,
    model: SVSE,
    image_ids: np.ndarray,
    caption_ids: np.ndarray,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image_set = data.Img_dataset(dataset.images[image_ids])
    image_loader = torch.utils.data.DataLoader(
        image_set, batch_size=batch_size, shuffle=False, collate_fn=data.collate_fn_img, num_workers=0
    )
    caption_strings = [dataset.captions[int(i)] for i in caption_ids]
    caption_set = data.Cap_dataset(caption_strings, dataset.vocab)
    caption_loader = torch.utils.data.DataLoader(
        caption_set, batch_size=batch_size, shuffle=False, collate_fn=data.collate_fn_cap, num_workers=0
    )
    image_embeddings: List[torch.Tensor] = []
    caption_embeddings: List[torch.Tensor] = []
    model_device = next(model.img_enc.parameters()).device
    with torch.no_grad():
        for images, lengths, _ in image_loader:
            image_embeddings.append(
                model.img_enc(images.to(model_device), lengths.to(model_device)).detach().float().cpu()
            )
        for captions, lengths, _ in caption_loader:
            caption_embeddings.append(
                model.txt_enc(captions.to(model_device), lengths.to(model_device)).detach().float().cpu()
            )
    return torch.cat(image_embeddings), torch.cat(caption_embeddings)


def _permutation_error(cost: torch.Tensor, result: Any, epsilon: float, max_iter: int, tol: float, seed: int) -> float:
    generator = torch.Generator(device=cost.device).manual_seed(seed)
    permutation = torch.randperm(cost.shape[1], generator=generator, device=cost.device)
    permuted = balanced_sinkhorn_log(cost[:, permutation], epsilon=epsilon, max_iter=max_iter, tol=tol)
    inverse = torch.argsort(permutation)
    return (result.plan - permuted.plan[:, inverse]).abs().max().item()


def _repeat_error(cost: torch.Tensor, result: Any, epsilon: float, max_iter: int, tol: float) -> float:
    repeated = balanced_sinkhorn_log(cost, epsilon=epsilon, max_iter=max_iter, tol=tol)
    return (result.plan - repeated.plan).abs().max().item()


def _reference_diagnostics(
    backend: str,
    cost: torch.Tensor,
    result: Any,
    epsilon: float,
) -> Dict[str, Any]:
    if backend == "none":
        return {"backend": "none", "available": False}
    try:
        reference = pot_sinkhorn_reference(cost, epsilon=epsilon)
    except RuntimeError:
        if backend == "pot":
            raise
        return {"backend": "pot", "available": False}
    pytorch_plan = result.plan.double()
    reference_row_target = torch.full(
        (reference.shape[0],), 1.0 / reference.shape[0], dtype=reference.dtype, device=reference.device
    )
    reference_col_target = torch.full(
        (reference.shape[1],), 1.0 / reference.shape[1], dtype=reference.dtype, device=reference.device
    )
    reference_objective = transport_objective(reference, cost.double(), epsilon).item()
    objective_relative_error = abs(result.objective - reference_objective) / max(
        abs(reference_objective), 1e-12
    )
    return {
        "backend": "pot",
        "available": True,
        "plan_l1_error": (pytorch_plan - reference).abs().sum().item(),
        "plan_max_error": (pytorch_plan - reference).abs().max().item(),
        "objective": reference_objective,
        "objective_relative_error": objective_relative_error,
        "max_row_error": (reference.sum(dim=1) - reference_row_target).abs().max().item(),
        "max_col_error": (reference.sum(dim=0) - reference_col_target).abs().max().item(),
    }


def run_o1(args: argparse.Namespace) -> Dict[str, Any]:
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    opt = _checkpoint_options(checkpoint, args)
    vocab_path = os.path.join(opt.vocab_path, f"{opt.data_name}_vocab.json")
    if getattr(opt, "init_txt", "uniform") == "glove":
        vocab = deserialize_vocab_glove(vocab_path)
    else:
        vocab = deserialize_vocab(vocab_path)
    dataset = data.PrecompDataset_gru(opt, os.path.join(opt.data_path, opt.data_name), "train", vocab)
    image_ids, caption_ids = _select_unpaired_groups(dataset, args.subset_size, args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_model(checkpoint, opt, vocab, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    total_start = time.perf_counter()
    image_embeddings, caption_embeddings = _encode_subset(dataset, model, image_ids, caption_ids, args.batch_size)
    image_embeddings = torch.nn.functional.normalize(image_embeddings.float(), dim=1).to(device)
    caption_embeddings = torch.nn.functional.normalize(caption_embeddings.float(), dim=1).to(device)
    similarity = image_embeddings @ caption_embeddings.t()
    cost = 1.0 - similarity
    result = balanced_sinkhorn_log(cost, epsilon=args.epsilon, max_iter=args.max_iter, tol=args.tol)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    core_elapsed = time.perf_counter() - total_start

    caption_group_ids = caption_ids // dataset.im_div
    gt_mask = torch.from_numpy(image_ids[:, None] == caption_group_ids[None, :]).to(result.plan.device)
    raw_gt_mass, normalized_gt_mass = gt_transport_mass(result.plan, gt_mask)
    permutation_error = _permutation_error(cost, result, args.epsilon, args.max_iter, args.tol, args.seed)
    repeat_error = _repeat_error(cost, result, args.epsilon, args.max_iter, args.tol)
    reference = _reference_diagnostics(args.reference_backend, cost, result, args.epsilon)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_elapsed = time.perf_counter() - total_start
    checks = {
        "converged": result.converged,
        "finite_cost": bool(torch.isfinite(cost).all()),
        "finite_plan": bool(torch.isfinite(result.plan).all()),
        "nonnegative_plan": bool((result.plan >= 0).all()),
        "row_marginal_error_below_1e-3": result.max_row_error < 1e-3,
        "col_marginal_error_below_1e-3": result.max_col_error < 1e-3,
        "deterministic_repeat": repeat_error == 0.0,
        "caption_permutation_equivariant": permutation_error < 1e-5,
    }
    if reference["available"]:
        checks["pot_plan_l1_error_below_1e-2"] = reference["plan_l1_error"] < 1e-2
        checks["pot_objective_relative_error_below_1e-3"] = reference["objective_relative_error"] < 1e-3
    summary: Dict[str, Any] = {
        "experiment": "O1",
        "passed": all(checks.values()),
        "checks": checks,
        "checkpoint": os.path.abspath(args.checkpoint),
        "data_name": opt.data_name,
        "seed": args.seed,
        "subset_size": int(args.subset_size),
        "caption_count": int(caption_ids.size),
        "im_div": int(dataset.im_div),
        "epsilon": args.epsilon,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "device": str(device),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "embedding_dim": int(image_embeddings.shape[1]),
        "core_elapsed_seconds": core_elapsed,
        "total_elapsed_seconds": total_elapsed,
        "similarity_min": similarity.min().item(),
        "similarity_max": similarity.max().item(),
        "cost_finite": bool(torch.isfinite(cost).all()),
        "plan_finite": bool(torch.isfinite(result.plan).all()),
        "plan_nonnegative": bool((result.plan >= 0).all()),
        "iterations": result.iterations,
        "converged": result.converged,
        "max_row_error": result.max_row_error,
        "max_col_error": result.max_col_error,
        "objective": result.objective,
        "final_objective_relative_delta": result.final_objective_relative_delta,
        "final_log_scaling_delta": result.final_log_scaling_delta,
        "raw_gt_transport_mass": raw_gt_mass,
        "normalized_gt_transport_mass": normalized_gt_mass,
        "caption_permutation_max_error": permutation_error,
        "deterministic_repeat_max_error": repeat_error,
        "reference": reference,
        "unpaired_image_ids": image_ids.tolist(),
        "corrupted_partner_image_ids": dataset.shuffle_inx[image_ids].astype(np.int64).tolist(),
        "caption_ids": caption_ids.tolist(),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "o1_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    if args.save_plan:
        np.savez_compressed(
            os.path.join(args.output_dir, "o1_transport.npz"),
            plan=result.plan.cpu().numpy(),
            similarity=similarity.cpu().numpy(),
            cost=cost.cpu().numpy(),
            gt_mask=gt_mask.cpu().numpy(),
            image_ids=image_ids,
            caption_ids=caption_ids,
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default="./data")
    parser.add_argument("--data-name", default=None)
    parser.add_argument("--vocab-path", default="./vocab")
    parser.add_argument("--subset-size", type=int, default=512)
    parser.add_argument("--paired-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--device", default=None, help="cpu, cuda, or a torch device string")
    parser.add_argument("--output-dir", default="./runs/o1")
    parser.add_argument("--save-plan", action="store_true")
    parser.add_argument(
        "--reference-backend",
        choices=("auto", "pot", "none"),
        default="auto",
        help="Use POT when available, require POT, or disable reference validation",
    )
    return parser


if __name__ == "__main__":
    result = run_o1(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=True))
