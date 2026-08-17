#!/usr/bin/env python3
"""
Compute classification metrics for hierarchical pipeline outputs.

Reads `hierarchical_predictions.csv` produced by `hierarchical_cascade.py` and writes:
- classification_report_{stage5,final}.csv
- confusion_matrix_{stage5,final}.png
- metrics_summary.json (micro/macro/weighted F1 + accuracy)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


LABELS_5 = ["AA", "AD", "ES", "ME", "RI"]


def _norm_label(x: object) -> str:
    s = str(x or "").strip()
    if not s:
        return ""
    return s.upper()


def _valid_pairs(df: pd.DataFrame, pred_col: str) -> Tuple[List[str], List[str]]:
    y_true = [_norm_label(v) for v in df["true_splice_type"].tolist()]
    y_pred = [_norm_label(v) for v in df[pred_col].tolist()]
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t in LABELS_5 and p in LABELS_5]
    if not pairs:
        return [], []
    yt, yp = zip(*pairs)
    return list(yt), list(yp)


def _write_report_and_cm(
    *,
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    out_dir: Path,
    tag: str,
    title: str,
) -> Dict[str, object]:
    report = classification_report(y_true, y_pred, labels=labels, target_names=labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_path = out_dir / f"classification_report_{tag}.csv"
    report_df.to_csv(report_path, index=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7.5, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    cm_path = out_dir / f"confusion_matrix_{tag}.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=220)
    plt.close()

    return {
        "report_csv": str(report_path),
        "confusion_matrix_png": str(cm_path),
        "confusion_matrix": cm.tolist(),
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
        "micro_f1": float(report.get("micro avg", {}).get("f1-score", report.get("accuracy", 0.0))),
        "support": int(report.get("macro avg", {}).get("support", len(y_true))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help="Path to hierarchical_predictions.csv (default: <run-dir>/hierarchical_predictions.csv)",
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory produced by hierarchical_cascade.py (contains hierarchical_predictions.csv).",
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: <run-dir>)")
    args = ap.parse_args()

    if args.predictions_csv is None and args.run_dir is None:
        raise SystemExit("Provide --run-dir or --predictions-csv")

    if args.predictions_csv is None:
        preds_csv = (args.run_dir / "hierarchical_predictions.csv").resolve()
    else:
        preds_csv = args.predictions_csv.resolve()
        if args.run_dir is None:
            args.run_dir = preds_csv.parent

    if not preds_csv.exists():
        raise SystemExit(f"Missing predictions CSV: {preds_csv}")

    out_dir = (args.out_dir or args.run_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(preds_csv)
    needed = {"true_splice_type", "stage5_pred", "final_pred"}
    missing = sorted(list(needed - set(df.columns)))
    if missing:
        raise SystemExit(f"Missing columns in {preds_csv.name}: {missing}")

    y_true_s5, y_pred_s5 = _valid_pairs(df, "stage5_pred")
    y_true_f, y_pred_f = _valid_pairs(df, "final_pred")

    out: Dict[str, object] = {
        "predictions_csv": str(preds_csv),
        "n_rows": int(df.shape[0]),
        "labels": LABELS_5,
    }

    if y_true_s5:
        out["stage5"] = _write_report_and_cm(
            y_true=y_true_s5,
            y_pred=y_pred_s5,
            labels=LABELS_5,
            out_dir=out_dir,
            tag="stage5",
            title="Confusion Matrix (stage-5-only)",
        )
    else:
        out["stage5"] = {"error": "No valid labeled pairs for stage5_pred."}

    if y_true_f:
        out["final"] = _write_report_and_cm(
            y_true=y_true_f,
            y_pred=y_pred_f,
            labels=LABELS_5,
            out_dir=out_dir,
            tag="final",
            title="Confusion Matrix (final cascade)",
        )
    else:
        out["final"] = {"error": "No valid labeled pairs for final_pred."}

    summary_path = out_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()

