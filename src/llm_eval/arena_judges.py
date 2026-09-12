from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

LABELS = np.array(["a", "b", "tie"])


def human_labels(frame: pd.DataFrame) -> np.ndarray:
    required = {"winner_model_a", "winner_model_b", "winner_tie"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing human-label columns: {sorted(missing)}")
    mat = frame[["winner_model_a", "winner_model_b", "winner_tie"]].to_numpy(dtype=float)
    if not np.allclose(mat.sum(axis=1), 1):
        raise ValueError("each human row must contain exactly one winner label")
    return LABELS[np.argmax(mat, axis=1)]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def read_judge_logits(path: str | Path) -> pd.DataFrame:
    """Read judge JSONL without depending on one fragile column name.

    Public judge files have one row per Chatbot Arena example and a 3-value
    score/logit vector. This loader discovers that vector defensively and keeps
    an id column when present.
    """
    records = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"empty judge file: {path}")

    df = pd.DataFrame(records)
    vector_col = None
    for col in df.columns:
        sample = next((v for v in df[col] if isinstance(v, (list, tuple)) and len(v) == 3), None)
        if sample is not None:
            vector_col = col
            break
    if vector_col is None:
        # Some exports store the three values as separate numeric columns.
        candidates = [
            ["a", "b", "tie"],
            ["logit_a", "logit_b", "logit_tie"],
            ["score_a", "score_b", "score_tie"],
        ]
        for cols in candidates:
            if set(cols).issubset(df.columns):
                values = df[cols].to_numpy(dtype=float)
                probs = _softmax(values)
                out = pd.DataFrame(probs, columns=["p_a", "p_b", "p_tie"])
                if "id" in df:
                    out.insert(0, "id", df["id"].to_numpy())
                return out
        raise ValueError(f"could not identify a 3-class judge score vector in {path}; columns={list(df.columns)}")

    values = np.vstack(df[vector_col].map(np.asarray).to_numpy()).astype(float)
    # The repository describes these as logits. Softmax is harmless if they are
    # unnormalized scores and gives comparable confidence values.
    probs = _softmax(values)
    out = pd.DataFrame(probs, columns=["p_a", "p_b", "p_tie"])
    if "id" in df:
        out.insert(0, "id", df["id"].to_numpy())
    return out


def canonicalize_reversed(probs: pd.DataFrame) -> pd.DataFrame:
    """Map predictions from a B/A presentation back to the original A/B frame."""
    out = probs.copy()
    out[["p_a", "p_b"]] = probs[["p_b", "p_a"]].to_numpy()
    return out


def _align(human: pd.DataFrame, original: pd.DataFrame, reversed_probs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "id" in human.columns and "id" in original.columns and "id" in reversed_probs.columns:
        base = human[["id", "winner_model_a", "winner_model_b", "winner_tie"]]
        o = base.merge(original, on="id", how="inner", validate="one_to_one")
        r = base.merge(reversed_probs, on="id", how="inner", validate="one_to_one")
        if len(o) != len(r):
            raise ValueError("original/reversed judge files align to different numbers of human examples")
        # enforce the same order
        r = r.set_index("id").loc[o["id"]].reset_index()
        return o, r, base
    n = min(len(human), len(original), len(reversed_probs))
    if n == 0:
        raise ValueError("no aligned judge examples")
    h = human.iloc[:n].reset_index(drop=True)
    o = pd.concat([h, original.iloc[:n].reset_index(drop=True)], axis=1)
    r = pd.concat([h, reversed_probs.iloc[:n].reset_index(drop=True)], axis=1)
    return o, r, h


def _pred_labels(probs: pd.DataFrame) -> np.ndarray:
    return LABELS[np.argmax(probs[["p_a", "p_b", "p_tie"]].to_numpy(), axis=1)]


def bootstrap_accuracy_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    n = len(correct)
    vals = [correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


@dataclass
class JudgeAudit:
    judge: str
    n: int
    original_accuracy: float
    reversed_accuracy: float
    symmetric_accuracy: float
    position_consistency: float
    symmetric_kappa: float
    automated_coverage: float
    automated_accuracy: float
    human_review_rate: float
    accuracy_ci_low: float
    accuracy_ci_high: float

    def to_dict(self) -> dict:
        return asdict(self)


def audit_judge(
    human: pd.DataFrame,
    original_probs: pd.DataFrame,
    reversed_probs: pd.DataFrame,
    judge_name: str,
    confidence_threshold: float = 0.60,
) -> JudgeAudit:
    orig, rev_raw, _ = _align(human, original_probs, reversed_probs)
    rev = canonicalize_reversed(rev_raw)
    y = human_labels(orig)

    o = orig[["p_a", "p_b", "p_tie"]].to_numpy(dtype=float)
    r = rev[["p_a", "p_b", "p_tie"]].to_numpy(dtype=float)
    sym = (o + r) / 2.0

    po = LABELS[np.argmax(o, axis=1)]
    pr = LABELS[np.argmax(r, axis=1)]
    ps = LABELS[np.argmax(sym, axis=1)]
    conf = sym.max(axis=1)
    stable = po == pr
    automate = stable & (conf >= confidence_threshold)

    acc_o = float((po == y).mean())
    acc_r = float((pr == y).mean())
    acc_s = float((ps == y).mean())
    ci_lo, ci_hi = bootstrap_accuracy_ci(y, ps)
    auto_acc = float((ps[automate] == y[automate]).mean()) if automate.any() else float("nan")

    return JudgeAudit(
        judge=judge_name,
        n=int(len(y)),
        original_accuracy=acc_o,
        reversed_accuracy=acc_r,
        symmetric_accuracy=acc_s,
        position_consistency=float(stable.mean()),
        symmetric_kappa=float(cohen_kappa_score(y, ps)),
        automated_coverage=float(automate.mean()),
        automated_accuracy=auto_acc,
        human_review_rate=float(1.0 - automate.mean()),
        accuracy_ci_low=ci_lo,
        accuracy_ci_high=ci_hi,
    )


def release_decision(audit: JudgeAudit, policy: dict) -> dict:
    checks = {
        "min_symmetric_accuracy": audit.symmetric_accuracy >= policy.get("min_symmetric_accuracy", 0.50),
        "min_position_consistency": audit.position_consistency >= policy.get("min_position_consistency", 0.80),
        "min_automated_accuracy": (not np.isnan(audit.automated_accuracy)) and audit.automated_accuracy >= policy.get("min_automated_accuracy", 0.60),
        "min_automated_coverage": audit.automated_coverage >= policy.get("min_automated_coverage", 0.20),
    }
    return {
        "judge": audit.judge,
        "status": "PASS" if all(checks.values()) else "HUMAN_REVIEW_REQUIRED",
        "checks": checks,
        "policy": policy,
    }
