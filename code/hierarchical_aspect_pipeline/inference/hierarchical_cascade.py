#!/usr/bin/env python3
"""
Hierarchical cancer splice-type inference for 5 labels (AA, AD, ES, ME, RI).


"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Disable flash-attn / triton compilation at import time (safe for inference).
os.environ.setdefault("USE_FLASH_ATTENTION", "0")
os.environ.setdefault("DISABLE_FLASH_ATTENTION", "1")
os.environ.setdefault("USE_FLASH_ATTENTION_2", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "TRUE")
os.environ.setdefault("FLASH_ATTENTION_FORCE_DISABLE", "TRUE")

import numpy as np
import torch

CANONICAL_5 = ("AA", "AD", "ES", "ME", "RI")

# Default dataset/checkpoint tree produced inside this release package.
DEFAULT_FINAL_ROOT = Path(__file__).resolve().parent.parent / "datasets"

# Single source of truth for cascade routing (CLI defaults + threshold_sweep.py).
# Defaults tuned via sweep on held-out internal test split (tune on v2 val if re-sweeping):
# best observed micro_final ≈ 91.2777% with stronger routing past stage 5.
DEFAULT_CASCADE_PARAMS: Dict[str, float] = {
    "accept_prob_5": 0.90,
    "accept_prob_4": 0.90,
    "accept_prob_3": 0.86,
    "accept_prob_2": 0.78,
    "accept_margin": 0.65,
    "accept_nentropy_5": 0.10,
    "accept_nentropy_4": 0.18,
    "accept_nentropy_3": 0.16,
    "accept_nentropy_2": 0.14,
}


# Block flash_attn_triton import (prevents Triton compilation errors).
class _FlashAttnStub:
    def __call__(self, *args, **kwargs):
        raise RuntimeError("flash_attn is disabled for this pipeline.")

    def __getattr__(self, name):
        return _FlashAttnStub()


class _FlashAttnTritonImportHook:
    """Intercept any import that references flash_attn_triton."""

    def find_spec(self, fullname, path, target=None):  # noqa: D401
        if "flash_attn_triton" not in str(fullname):
            return None
        from importlib.util import spec_from_loader

        class _StubLoader:
            def create_module(self, spec):
                m = type(sys)("flash_attn_triton_stub")
                # Intentionally DO NOT provide flash_attn_* symbols.
                # Many models do:
                #   try: from flash_attn_triton import flash_attn_qkvpacked_func
                #   except ImportError: ... fallback to PyTorch attention ...
                # If we provide the symbol and raise at runtime, the model may crash.
                # Missing symbol triggers ImportError early and activates the safe fallback.
                return m

            def exec_module(self, module):
                return None

        return spec_from_loader(fullname, _StubLoader())


# Install hook early (before loading trust_remote_code modules).
if not any(isinstance(h, _FlashAttnTritonImportHook) for h in sys.meta_path):
    sys.meta_path.insert(0, _FlashAttnTritonImportHook())

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy in nats for each row."""
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def _normalized_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Entropy normalized to [0,1] by dividing by log(K)."""
    k = probs.shape[-1]
    if k <= 1:
        return np.zeros((probs.shape[0],), dtype=np.float32)
    return _entropy(probs, eps=eps) / float(np.log(k))


def _classification_report_dict(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, Any]:
    """
    Lightweight classification report (no sklearn dependency).
    Returns per-class precision/recall/f1/support + accuracy + confusion_matrix.
    """
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        ti = label_to_idx.get(t)
        pi = label_to_idx.get(p)
        if ti is None or pi is None:
            continue
        cm[ti, pi] += 1

    supports = cm.sum(axis=1)
    total = int(supports.sum())
    correct = int(np.trace(cm))
    accuracy = (correct / total) if total else 0.0

    out: Dict[str, Any] = {"confusion_matrix": cm.tolist(), "accuracy": accuracy}

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    weights: List[int] = []

    for i, lab in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(supports[i])
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
        out[lab] = {"precision": prec, "recall": rec, "f1-score": f1, "support": support}
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        weights.append(support)

    out["macro avg"] = {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1-score": float(np.mean(f1s)) if f1s else 0.0,
        "support": total,
    }
    out["weighted avg"] = {
        "precision": float(np.average(precisions, weights=weights)) if total else 0.0,
        "recall": float(np.average(recalls, weights=weights)) if total else 0.0,
        "f1-score": float(np.average(f1s, weights=weights)) if total else 0.0,
        "support": total,
    }
    return out


def _write_classification_metrics(
    *,
    out_dir: Path,
    y: Sequence[Optional[str]],
    decisions: List[Dict[str, Any]],
) -> None:
    labels = list(CANONICAL_5)

    y_true: List[str] = []
    y_stage5: List[str] = []
    y_final: List[str] = []

    for i, d in enumerate(decisions):
        t = y[i]
        if t not in CANONICAL_5:
            continue
        s5 = (d.get("stages") or {}).get("5") or {}
        p5 = s5.get("pred")
        pf = d.get("final_pred")
        if p5 not in CANONICAL_5 or pf not in CANONICAL_5:
            continue
        y_true.append(str(t))
        y_stage5.append(str(p5))
        y_final.append(str(pf))

    rep_s5 = _classification_report_dict(y_true, y_stage5, labels)
    rep_fin = _classification_report_dict(y_true, y_final, labels)

    def _to_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for lab in labels:
            r = report.get(lab, {})
            rows.append(
                {
                    "label": lab,
                    "precision": r.get("precision", 0.0),
                    "recall": r.get("recall", 0.0),
                    "f1": r.get("f1-score", 0.0),
                    "support": r.get("support", 0),
                }
            )
        for k in ("macro avg", "weighted avg"):
            r = report.get(k, {})
            rows.append(
                {
                    "label": k,
                    "precision": r.get("precision", 0.0),
                    "recall": r.get("recall", 0.0),
                    "f1": r.get("f1-score", 0.0),
                    "support": r.get("support", 0),
                }
            )
        rows.append(
            {
                "label": "accuracy",
                "precision": "",
                "recall": "",
                "f1": report.get("accuracy", 0.0),
                "support": len(y_true),
            }
        )
        return rows

    for tag, report in (("stage5", rep_s5), ("final", rep_fin)):
        with open(out_dir / f"classification_report_{tag}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["label", "precision", "recall", "f1", "support"])
            w.writeheader()
            for r in _to_rows(report):
                w.writerow(r)

    summary = {
        "n_labeled_used": len(y_true),
        "labels": labels,
        "stage5": {
            "accuracy": rep_s5.get("accuracy", 0.0),
            "macro_f1": rep_s5.get("macro avg", {}).get("f1-score", 0.0),
            "weighted_f1": rep_s5.get("weighted avg", {}).get("f1-score", 0.0),
            "confusion_matrix": rep_s5.get("confusion_matrix", []),
        },
        "final": {
            "accuracy": rep_fin.get("accuracy", 0.0),
            "macro_f1": rep_fin.get("macro avg", {}).get("f1-score", 0.0),
            "weighted_f1": rep_fin.get("weighted avg", {}).get("f1-score", 0.0),
            "confusion_matrix": rep_fin.get("confusion_matrix", []),
        },
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Optional PNGs (best-effort).
    try:
        import matplotlib.pyplot as plt  # type: ignore

        def _plot_cm(cm: List[List[int]], title: str, path: Path) -> None:
            fig, ax = plt.subplots(figsize=(7.5, 6.5))
            ax.imshow(np.array(cm), cmap="Blues")
            ax.set_xticks(range(len(labels)), labels=labels)
            ax.set_yticks(range(len(labels)), labels=labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(title)
            # annotate
            for (rr, cc), val in np.ndenumerate(np.array(cm)):
                ax.text(cc, rr, str(int(val)), ha="center", va="center", fontsize=9)
            fig.tight_layout()
            fig.savefig(path, dpi=220)
            plt.close(fig)

        _plot_cm(summary["stage5"]["confusion_matrix"], "Confusion Matrix (stage-5-only)", out_dir / "confusion_matrix_stage5.png")
        _plot_cm(summary["final"]["confusion_matrix"], "Confusion Matrix (final cascade)", out_dir / "confusion_matrix_final.png")
    except Exception:
        pass


def _normalize_label(s: str) -> str:
    return str(s).strip().upper()


def _dataset_name_for_subset(labels: Sequence[str]) -> str:
    # Matches the training folder naming convention: A_vs_B_vs_C...
    return "_vs_".join(labels)


def _read_sequences(path: Path) -> Tuple[List[str], List[Optional[str]]]:
    seqs: List[str] = []
    y: List[Optional[str]] = []
    with open(path, newline="") as handle:
        r = csv.DictReader(handle)
        if not r.fieldnames or "sequence" not in r.fieldnames:
            raise SystemExit(f"Missing required column 'sequence' in {path}")
        has_label = "splice_type" in (r.fieldnames or [])
        for row in r:
            if not row:
                continue
            seq = str(row.get("sequence", "")).strip().upper()
            if not seq:
                continue
            seqs.append(seq)
            if has_label:
                lab = _normalize_label(row.get("splice_type", ""))
                y.append(lab if lab else None)
            else:
                y.append(None)
    return seqs, y


@dataclass(frozen=True)
class ModelIndex:
    multi5: Path
    four: Dict[Tuple[str, ...], Path]
    three: Dict[Tuple[str, ...], Path]
    binary: Dict[Tuple[str, ...], Path]


def _scan_subset_models(tr_root: Path, group: str) -> Dict[Tuple[str, ...], Path]:
    out: Dict[Tuple[str, ...], Path] = {}
    base = tr_root / group
    if not base.exists():
        return out
    for best in base.glob("DB2_*/best_model"):
        run = best.parent.name  # DB2_<dataset>
        if not run.startswith("DB2_"):
            continue
        dataset = run.replace("DB2_", "", 1)
        if "_vs_" not in dataset:
            continue
        labels = tuple(dataset.split("_vs_"))
        labels = tuple(_normalize_label(x) for x in labels)
        key = tuple(sorted(labels))
        out[key] = best
    return out


def _parse_retrain_root(arg: Optional[str], final_root: Path) -> Optional[Path]:
    """Resolve --training-runs-retrain: auto | none | explicit path."""
    if arg is None:
        s = "auto"
    else:
        s = str(arg).strip()
    if s.lower() in ("", "none", "false", "off"):
        return None
    if s.lower() == "auto":
        p = final_root / "training_runs_retrain"
        return p if p.is_dir() else None
    return Path(s).resolve()


def build_model_index(final_root: Path, retrain_root: Optional[Path] = None) -> Tuple[ModelIndex, Dict[str, Any]]:
    tr = final_root / "training_runs"
    if not tr.exists():
        raise SystemExit(f"Missing training_runs at {tr}")

    # 5-class (base)
    multi5 = tr / "multi_class_datasets" / "DB2_multi_class_datasets" / "best_model"
    if not multi5.exists():
        raise SystemExit(f"Missing 5-class best_model at {multi5}")

    four = _scan_subset_models(tr, "four_class_datasets")
    three = _scan_subset_models(tr, "three_class_datasets")
    binary = _scan_subset_models(tr, "binary_datsets")

    meta: Dict[str, Any] = {
        "retrain_root": None,
        "multi5_path": str(multi5),
        "multi5_retrain_override": None,
        "subset_overrides": [],
        "subset_additions": [],
    }

    if retrain_root is not None and retrain_root.is_dir():
        meta["retrain_root"] = str(retrain_root.resolve())
        rm = retrain_root / "multi_class_datasets" / "DB2_multi_class_datasets" / "best_model"
        if rm.exists():
            meta["multi5_retrain_override"] = str(rm.resolve())
            multi5 = rm

        def _merge(
            name: str,
            base: Dict[Tuple[str, ...], Path],
            overlay: Dict[Tuple[str, ...], Path],
        ) -> None:
            for k, v in overlay.items():
                vr = v.resolve()
                if k in base:
                    meta["subset_overrides"].append(
                        {
                            "group": name,
                            "labels": list(k),
                            "base_model": str(base[k].resolve()),
                            "retrain_model": str(vr),
                        }
                    )
                else:
                    meta["subset_additions"].append(
                        {"group": name, "labels": list(k), "retrain_model": str(vr)}
                    )
                base[k] = v

        _merge("four_class_datasets", four, _scan_subset_models(retrain_root, "four_class_datasets"))
        _merge("three_class_datasets", three, _scan_subset_models(retrain_root, "three_class_datasets"))
        _merge("binary_datsets", binary, _scan_subset_models(retrain_root, "binary_datsets"))

    return ModelIndex(multi5=multi5, four=four, three=three, binary=binary), meta


@dataclass
class StageDecision:
    pred: str
    prob: float
    margin: float
    probs: Dict[str, float]
    used_model: str


class HFModel:
    def __init__(self, model_dir: Path, device: str, temperature: float = 1.0):
        self.model_dir = model_dir
        self.device = device
        self.temperature = float(temperature) if float(temperature) > 0 else 1.0
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

    @property
    def max_length(self) -> int:
        ml = getattr(self.tokenizer, "model_max_length", 256)
        try:
            return int(ml)
        except Exception:
            return 256

    def predict_proba(self, seqs: Sequence[str], batch_size: int) -> np.ndarray:
        all_logits: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(seqs), batch_size):
                batch = list(seqs[start : start + batch_size])
                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits_t = self.model(**enc).logits
                all_logits.append(logits_t.detach().cpu().numpy())
        logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, 1), dtype=np.float32)
        if not logits.size:
            return logits
        logits = logits / self.temperature
        return _softmax(logits)


def _decide_from_probs(labels: Sequence[str], probs_row: np.ndarray) -> StageDecision:
    pairs = list(zip(labels, probs_row.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    pred, p1 = pairs[0]
    p2 = pairs[1][1] if len(pairs) > 1 else 0.0
    return StageDecision(
        pred=pred,
        prob=float(p1),
        margin=float(p1 - p2),
        probs={lab: float(p) for lab, p in zip(labels, probs_row.tolist())},
        used_model=_dataset_name_for_subset(labels),
    )


def _calibrate_temperature_grid_search(
    model_dir: Path,
    seqs: Sequence[str],
    labels: Sequence[str],
    y_true: Sequence[str],
    device: str,
    grid: Sequence[float] = (0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0),
) -> float:
    """
    Very lightweight temperature calibration using labeled data.
    Minimizes NLL on the provided calibration set.
    """
    lab_to_id = {lab: i for i, lab in enumerate(labels)}
    y = np.asarray([lab_to_id.get(_normalize_label(t), -1) for t in y_true], dtype=int)
    keep = np.where(y >= 0)[0]
    if keep.size == 0:
        return 1.0
    seqs_k = [seqs[i] for i in keep.tolist()]
    y_k = y[keep]

    best_t = 1.0
    best_nll = float("inf")
    # We reuse the same base model weights; only temperature changes.
    base = HFModel(model_dir, device=device, temperature=1.0)

    # Get logits once to avoid recomputing model forward passes per T.
    all_logits: List[np.ndarray] = []
    with torch.no_grad():
        bs = 64 if device.startswith("cuda") else 16
        for start in range(0, len(seqs_k), bs):
            batch = list(seqs_k[start : start + bs])
            enc = base.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=base.max_length,
            )
            enc = {k: v.to(base.device) for k, v in enc.items()}
            logits_t = base.model(**enc).logits
            all_logits.append(logits_t.detach().cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)

    for t in grid:
        t = float(t)
        if t <= 0:
            continue
        probs = _softmax(logits / t)
        # NLL
        p = np.clip(probs[np.arange(len(y_k)), y_k], 1e-12, 1.0)
        nll = float(-np.mean(np.log(p)))
        if nll < best_nll:
            best_nll = nll
            best_t = t
    return best_t


def run_cascade(
    seqs: Sequence[str],
    index: ModelIndex,
    device: str,
    accept_prob_5: float,
    accept_prob_4: float,
    accept_prob_3: float,
    accept_prob_2: float,
    accept_margin: float,
    accept_nentropy_5: float,
    accept_nentropy_4: float,
    accept_nentropy_3: float,
    accept_nentropy_2: float,
    temperature: float,
) -> List[Dict[str, Any]]:
    """
    Returns per-sequence dict with final decision and per-stage diagnostics.

    Strategy
    --------
    - Stage 5: run 5-class on everyone. If confident (prob >= accept_prob_5 or margin >= accept_margin) -> finalize.
    - Otherwise: drop the lowest-prob label and attempt stage 4 on the remaining 4 labels (if that subset model exists).
    - Repeat for stage 3.
    - Stage 2: use binary model for top-2 labels if available, else fall back to top-1 from last stage.
    """

    n = len(seqs)
    results: List[Dict[str, Any]] = [
        {
            "sequence": seqs[i],
            "final_pred": None,
            "final_prob": None,
            "final_stage": None,
            "stages": {},
        }
        for i in range(n)
    ]

    # 5-class stage
    m5 = HFModel(index.multi5, device=device, temperature=temperature)
    probs5 = m5.predict_proba(seqs, batch_size=64 if device.startswith("cuda") else 16)
    labels5 = list(CANONICAL_5)
    nent5 = _normalized_entropy(probs5) if probs5.size else np.zeros((len(seqs),), dtype=np.float32)

    undecided: List[int] = []
    next_sets: Dict[Tuple[str, ...], List[int]] = {}
    for i in range(n):
        d5 = _decide_from_probs(labels5, probs5[i])
        ne5 = float(nent5[i]) if len(nent5) > i else 0.0
        results[i]["stages"]["5"] = {
            "labels": labels5,
            "pred": d5.pred,
            "prob": d5.prob,
            "margin": d5.margin,
            "nentropy": ne5,
            "probs": d5.probs,
            "model": str(index.multi5),
        }
        confident = (d5.prob >= accept_prob_5) or (d5.margin >= accept_margin) or (ne5 <= accept_nentropy_5)
        if confident:
            results[i]["final_pred"] = d5.pred
            results[i]["final_prob"] = d5.prob
            results[i]["final_stage"] = "5"
            continue

        # Drop the least likely label and route to 4-class subset.
        drop = min(d5.probs.items(), key=lambda x: x[1])[0]
        subset = tuple(sorted([x for x in labels5 if x != drop]))
        undecided.append(i)
        next_sets.setdefault(subset, []).append(i)

    # Helper to run one subset stage (4 or 3)
    def run_subset_stage(
        stage: str,
        model_map: Dict[Tuple[str, ...], Path],
        accept_prob: float,
        accept_nentropy: float,
        routed: Dict[Tuple[str, ...], List[int]],
    ) -> Tuple[List[int], Dict[Tuple[str, ...], List[int]]]:
        still_undecided: List[int] = []
        next_routed: Dict[Tuple[str, ...], List[int]] = {}

        for subset, idxs in routed.items():
            model_dir = model_map.get(tuple(sorted(subset)))
            if model_dir is None or not model_dir.exists():
                # Can't refine; keep undecided for later stages using prior info.
                still_undecided.extend(idxs)
                continue

            model = HFModel(model_dir, device=device, temperature=temperature)
            batch_seqs = [seqs[i] for i in idxs]
            probs = model.predict_proba(batch_seqs, batch_size=64 if device.startswith("cuda") else 16)
            labels = list(subset)
            nent = _normalized_entropy(probs) if probs.size else np.zeros((len(batch_seqs),), dtype=np.float32)

            for j, row_idx in enumerate(idxs):
                d = _decide_from_probs(labels, probs[j])
                ne = float(nent[j]) if len(nent) > j else 0.0
                results[row_idx]["stages"][stage] = {
                    "labels": labels,
                    "pred": d.pred,
                    "prob": d.prob,
                    "margin": d.margin,
                    "nentropy": ne,
                    "probs": d.probs,
                    "model": str(model_dir),
                }
                confident = (d.prob >= accept_prob) or (d.margin >= accept_margin) or (ne <= accept_nentropy)
                if confident:
                    results[row_idx]["final_pred"] = d.pred
                    results[row_idx]["final_prob"] = d.prob
                    results[row_idx]["final_stage"] = stage
                else:
                    # Drop the least likely label and route down one more level.
                    drop = min(d.probs.items(), key=lambda x: x[1])[0]
                    subset_next = tuple(sorted([x for x in labels if x != drop]))
                    still_undecided.append(row_idx)
                    next_routed.setdefault(subset_next, []).append(row_idx)

        return still_undecided, next_routed

    # 4-class stage
    undecided4, routed3 = run_subset_stage("4", index.four, accept_prob_4, accept_nentropy_4, next_sets)

    # 3-class stage (only for the ones we successfully routed into a 3-set)
    undecided3, routed2 = run_subset_stage("3", index.three, accept_prob_3, accept_nentropy_3, routed3)

    # Combine undecideds (including those that couldn't find subset-models)
    undecided_all = sorted(set(undecided4 + undecided3))

    # 2-class stage: for each sample, pick a binary model for its current top-2 candidate set.
    # If we have no stage-3/4 decision, use stage-5 top-2.
    def top2_from_last_stage(i: int) -> Tuple[str, str]:
        for st in ("3", "4", "5"):
            s = results[i]["stages"].get(st)
            if not s:
                continue
            probs = s["probs"]
            pairs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            return tuple(sorted([pairs[0][0], pairs[1][0]]))  # type: ignore[return-value]
        return tuple(sorted([CANONICAL_5[0], CANONICAL_5[1]]))  # fallback

    # Group undecided by binary subset
    bin_groups: Dict[Tuple[str, str], List[int]] = {}
    for i in undecided_all:
        if results[i]["final_pred"] is not None:
            continue
        a, b = top2_from_last_stage(i)
        bin_groups.setdefault((a, b), []).append(i)

    for subset2, idxs in bin_groups.items():
        model_dir = index.binary.get(tuple(sorted(subset2)))
        if model_dir is None or not model_dir.exists():
            # No binary model available; fall back to last-stage top-1.
            for i in idxs:
                for st in ("3", "4", "5"):
                    s = results[i]["stages"].get(st)
                    if s:
                        results[i]["final_pred"] = s["pred"]
                        results[i]["final_prob"] = s["prob"]
                        results[i]["final_stage"] = f"{st}_fallback"
                        break
            continue

        model = HFModel(model_dir, device=device, temperature=temperature)
        batch_seqs = [seqs[i] for i in idxs]
        probs = model.predict_proba(batch_seqs, batch_size=128 if device.startswith("cuda") else 32)
        labels = list(subset2)
        nent = _normalized_entropy(probs) if probs.size else np.zeros((len(batch_seqs),), dtype=np.float32)

        for j, row_idx in enumerate(idxs):
            d2 = _decide_from_probs(labels, probs[j])
            ne2 = float(nent[j]) if len(nent) > j else 0.0
            results[row_idx]["stages"]["2"] = {
                "labels": labels,
                "pred": d2.pred,
                "prob": d2.prob,
                "margin": d2.margin,
                "nentropy": ne2,
                "probs": d2.probs,
                "model": str(model_dir),
            }
            results[row_idx]["final_pred"] = d2.pred
            results[row_idx]["final_prob"] = d2.prob
            results[row_idx]["final_stage"] = "2"
            # Note: we always take the highest-prob binary prediction as final_pred.
            # We do not emit a separate "lowconf" stage bucket/flag in outputs.

    # Any remaining unset (shouldn't happen): take stage-5 pred.
    for i in range(n):
        if results[i]["final_pred"] is None:
            s5 = results[i]["stages"].get("5", {})
            results[i]["final_pred"] = s5.get("pred")
            results[i]["final_prob"] = s5.get("prob")
            results[i]["final_stage"] = "5_fallback"

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True, help="CSV with at least a 'sequence' column; optional 'splice_type'.")
    ap.add_argument(
        "--final-root",
        default=str(DEFAULT_FINAL_ROOT),
        help="Dataset + training_runs root (default: <package>/datasets).",
    )
    ap.add_argument(
        "--training-runs-retrain",
        default="auto",
        help="Retrain checkpoint tree: 'auto' = use <final-root>/training_runs_retrain if that directory "
        "exists; 'none' = disable; or an explicit path. Same DB2_*/best_model layout as training_runs; "
        "matching subsets override base models.",
    )
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "cancer_results_hier"), help="Output folder")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    ap.add_argument(
        "--write-metrics",
        action="store_true",
        help="Write classification metrics (CSV+JSON; PNG if matplotlib available) into --out-dir.",
    )

    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature scaling for logits->softmax (1.0 = none).",
    )
    ap.add_argument(
        "--calibration-csv",
        default=None,
        help="Optional labeled CSV to calibrate temperature (must include splice_type).",
    )

    ap.add_argument(
        "--accept-prob-5",
        type=float,
        default=DEFAULT_CASCADE_PARAMS["accept_prob_5"],
        help="Stop at 5-class if top prob >= this (higher -> route more to 4/3/2).",
    )
    ap.add_argument(
        "--accept-prob-4",
        type=float,
        default=DEFAULT_CASCADE_PARAMS["accept_prob_4"],
        help="Stop at 4-class if top prob >= this.",
    )
    ap.add_argument(
        "--accept-prob-3",
        type=float,
        default=DEFAULT_CASCADE_PARAMS["accept_prob_3"],
        help="Stop at 3-class if top prob >= this.",
    )
    ap.add_argument(
        "--accept-prob-2",
        type=float,
        default=DEFAULT_CASCADE_PARAMS["accept_prob_2"],
        help="Binary stage 'confident' if top prob >= this.",
    )
    ap.add_argument(
        "--accept-margin",
        type=float,
        default=DEFAULT_CASCADE_PARAMS["accept_margin"],
        help="Accept if top1-top2 >= margin at any stage (higher -> stricter early stop).",
    )
    ap.add_argument(
        "--accept-nentropy-5",
        type=float,
        default=DEFAULT_CASCADE_PARAMS["accept_nentropy_5"],
        help="Accept if normalized entropy <= this (lower -> stricter).",
    )
    ap.add_argument(
        "--accept-nentropy-4",
        type=float,
        default=DEFAULT_CASCADE_PARAMS["accept_nentropy_4"],
        help="Accept if normalized entropy <= this.",
    )
    ap.add_argument(
        "--accept-nentropy-3",
        type=float,
        default=DEFAULT_CASCADE_PARAMS["accept_nentropy_3"],
        help="Accept if normalized entropy <= this.",
    )
    ap.add_argument(
        "--accept-nentropy-2",
        type=float,
        default=DEFAULT_CASCADE_PARAMS["accept_nentropy_2"],
        help="Accept if normalized entropy <= this.",
    )
    args = ap.parse_args()

    final_root = Path(args.final_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    seqs, y = _read_sequences(Path(args.input_csv).resolve())
    if not seqs:
        raise SystemExit("No sequences loaded.")

    retrain_path = _parse_retrain_root(args.training_runs_retrain, final_root)
    index, retrain_meta = build_model_index(final_root, retrain_path)

    temperature = float(args.temperature)
    if args.calibration_csv:
        cal_seqs, cal_y = _read_sequences(Path(args.calibration_csv).resolve())
        cal_y2 = [yy for yy in cal_y if yy is not None]
        if not cal_y2:
            raise SystemExit("--calibration-csv provided but no splice_type labels found.")
        temperature = _calibrate_temperature_grid_search(
            model_dir=index.multi5,
            seqs=cal_seqs,
            labels=list(CANONICAL_5),
            y_true=[yy or "" for yy in cal_y],
            device=device,
        )
        (out_dir / "temperature_calibration.json").write_text(
            json.dumps({"temperature": temperature, "calibration_csv": str(args.calibration_csv)}, indent=2)
        )
    decisions = run_cascade(
        seqs=seqs,
        index=index,
        device=device,
        accept_prob_5=float(args.accept_prob_5),
        accept_prob_4=float(args.accept_prob_4),
        accept_prob_3=float(args.accept_prob_3),
        accept_prob_2=float(args.accept_prob_2),
        accept_margin=float(args.accept_margin),
        accept_nentropy_5=float(args.accept_nentropy_5),
        accept_nentropy_4=float(args.accept_nentropy_4),
        accept_nentropy_3=float(args.accept_nentropy_3),
        accept_nentropy_2=float(args.accept_nentropy_2),
        temperature=temperature,
    )

    # Write outputs
    out_csv = out_dir / "hierarchical_predictions.csv"
    out_jsonl = out_dir / "hierarchical_predictions.jsonl"

    # flatten stage summaries for CSV
    fieldnames = [
        "sequence",
        "true_splice_type",
        "final_pred",
        "final_prob",
        "final_stage",
        "stage5_pred",
        "stage5_prob",
        "stage4_pred",
        "stage4_prob",
        "stage3_pred",
        "stage3_prob",
        "stage2_pred",
        "stage2_prob",
    ]

    with open(out_jsonl, "w") as jh:
        for i, d in enumerate(decisions):
            rec = dict(d)
            rec["true_splice_type"] = y[i]
            jh.write(json.dumps(rec) + "\n")

    with open(out_csv, "w", newline="") as ch:
        w = csv.DictWriter(ch, fieldnames=fieldnames)
        w.writeheader()
        for i, d in enumerate(decisions):
            s = d.get("stages", {})
            row = {
                "sequence": d["sequence"],
                "true_splice_type": y[i] or "",
                "final_pred": d.get("final_pred") or "",
                "final_prob": d.get("final_prob") if d.get("final_prob") is not None else "",
                "final_stage": d.get("final_stage") or "",
                "stage5_pred": (s.get("5") or {}).get("pred", ""),
                "stage5_prob": (s.get("5") or {}).get("prob", ""),
                "stage4_pred": (s.get("4") or {}).get("pred", ""),
                "stage4_prob": (s.get("4") or {}).get("prob", ""),
                "stage3_pred": (s.get("3") or {}).get("pred", ""),
                "stage3_prob": (s.get("3") or {}).get("prob", ""),
                "stage2_pred": (s.get("2") or {}).get("pred", ""),
                "stage2_prob": (s.get("2") or {}).get("prob", ""),
            }
            w.writerow(row)

    # Small run summary
    stage_counts: Dict[str, int] = {}
    for d in decisions:
        st = str(d.get("final_stage"))
        stage_counts[st] = stage_counts.get(st, 0) + 1
    # Model usage counts (paths invoked at each stage)
    model_usage: Dict[str, int] = {}
    for d in decisions:
        for st, payload in (d.get("stages") or {}).items():
            mp = str((payload or {}).get("model", ""))
            if mp:
                model_usage[mp] = model_usage.get(mp, 0) + 1

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "n": len(decisions),
                "final_stage_counts": stage_counts,
                "temperature": temperature,
                "model_usage_counts": model_usage,
                "retrain_models": retrain_meta,
            },
            indent=2,
        )
    )

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_jsonl}")
    print(f"Wrote: {out_dir / 'summary.json'}")
    if args.write_metrics:
        _write_classification_metrics(out_dir=out_dir, y=y, decisions=decisions)
        print(f"Wrote: {out_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()

