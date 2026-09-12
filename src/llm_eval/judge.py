from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

KEYS = ["question_id", "model_a", "model_b", "turn"]


def _revert(winner: str) -> str:
    if winner == "model_a":
        return "model_b"
    if winner == "model_b":
        return "model_a"
    return winner


def canonicalize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    required = set(KEYS + ["winner"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    rows = []
    for r in df[KEYS + ["winner"]].itertuples(index=False):
        if r.model_a <= r.model_b:
            rows.append((r.question_id, r.model_a, r.model_b, r.turn, r.winner))
        else:
            rows.append((r.question_id, r.model_b, r.model_a, r.turn, _revert(r.winner)))
    return pd.DataFrame(rows, columns=KEYS + ["winner"])


def _majority_winner(group: pd.Series) -> str:
    counts = group.value_counts()
    if counts.empty:
        return "tie"
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return "tie"
    return str(counts.index[0])


def align_annotation_level(human: pd.DataFrame, gpt4: pd.DataFrame) -> pd.DataFrame:
    """Replicate the official MT-Bench idea: compare the one GPT-4 vote with every human vote for the same canonical pair."""
    h = canonicalize_pairs(human).rename(columns={"winner": "human_winner"})
    g = canonicalize_pairs(gpt4).drop_duplicates(KEYS).rename(columns={"winner": "gpt4_winner"})
    return h.merge(g, on=KEYS, how="inner")


def align_majority_level(human: pd.DataFrame, gpt4: pd.DataFrame) -> pd.DataFrame:
    h = canonicalize_pairs(human)
    h = h.groupby(KEYS, dropna=False)["winner"].agg(_majority_winner).reset_index(name="human_winner")
    g = canonicalize_pairs(gpt4).drop_duplicates(KEYS).rename(columns={"winner": "gpt4_winner"})
    return h.merge(g, on=KEYS, how="inner")


def agreement_metrics(aligned: pd.DataFrame) -> dict:
    if aligned.empty:
        return {"n": 0, "agreement": float("nan"), "kappa": float("nan")}
    y_h = aligned["human_winner"].astype(str)
    y_g = aligned["gpt4_winner"].astype(str)
    return {
        "n": int(len(aligned)),
        "agreement": float((y_h == y_g).mean()),
        "kappa": float(cohen_kappa_score(y_h, y_g)),
    }


def reliability_by_turn(aligned: pd.DataFrame) -> list[dict]:
    return [{"turn": int(turn), **agreement_metrics(g)} for turn, g in aligned.groupby("turn")]


def bootstrap_agreement_ci(aligned: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    if aligned.empty:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    eq = (aligned["human_winner"].astype(str).to_numpy() == aligned["gpt4_winner"].astype(str).to_numpy()).astype(float)
    vals = [eq[rng.integers(0, len(eq), len(eq))].mean() for _ in range(n_boot)]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def agreement_without_ties(aligned: pd.DataFrame) -> dict:
    d = aligned[(aligned["human_winner"] != "tie") & (aligned["gpt4_winner"] != "tie")]
    return agreement_metrics(d)


def release_gate(aligned: pd.DataFrame, min_agreement: float = 0.80, min_kappa: float = 0.60, min_n: int = 100) -> dict:
    overall = agreement_metrics(aligned)
    turn_rows = reliability_by_turn(aligned)
    failing = [r for r in turn_rows if r["n"] < min_n or r["agreement"] < min_agreement or r["kappa"] < min_kappa]
    return {
        "overall": overall,
        "thresholds": {"min_agreement": min_agreement, "min_kappa": min_kappa, "min_n": min_n},
        "automated_release_gate": len(failing) == 0 and overall["agreement"] >= min_agreement and overall["kappa"] >= min_kappa,
        "human_review_slices": failing,
    }
