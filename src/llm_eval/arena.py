from __future__ import annotations

import ast
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def add_winner_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "winner" in out.columns:
        return out
    required = {"winner_model_a", "winner_model_b", "winner_tie"}
    if not required.issubset(out.columns):
        raise ValueError("Arena data must contain winner or winner_model_a/b/tie columns")
    out["winner"] = np.select(
        [out["winner_model_a"].astype(int).eq(1), out["winner_model_b"].astype(int).eq(1)],
        ["model_a", "model_b"],
        default="tie",
    )
    return out


def _text_len(value) -> int:
    if isinstance(value, list):
        return sum(len(str(x)) for x in value)
    if pd.isna(value):
        return 0
    s = str(value)
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return sum(len(str(x)) for x in parsed)
    except Exception:
        pass
    return len(s)


def preference_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = add_winner_label(df)
    rows = []
    for model in sorted(set(d["model_a"]).union(d["model_b"])):
        a = d[d["model_a"] == model]
        b = d[d["model_b"] == model]
        wins = int((a["winner"] == "model_a").sum() + (b["winner"] == "model_b").sum())
        losses = int((a["winner"] == "model_b").sum() + (b["winner"] == "model_a").sum())
        ties = int((a["winner"] == "tie").sum() + (b["winner"] == "tie").sum())
        n = wins + losses + ties
        if n:
            rows.append({"model": model, "battles": n, "wins": wins, "losses": losses, "ties": ties, "tie_adjusted_win_rate": (wins + 0.5 * ties) / n})
    return pd.DataFrame(rows).sort_values(["tie_adjusted_win_rate", "battles"], ascending=[False, False]).reset_index(drop=True)


def bradley_terry_scores(df: pd.DataFrame, min_battles: int = 50) -> pd.DataFrame:
    d = add_winner_label(df)
    counts = pd.concat([d["model_a"], d["model_b"]]).value_counts()
    models = sorted(counts[counts >= min_battles].index.tolist())
    index = {m: i for i, m in enumerate(models)}
    X, y = [], []
    for row in d.itertuples(index=False):
        ma, mb, winner = row.model_a, row.model_b, row.winner
        if ma not in index or mb not in index or winner == "tie":
            continue
        x = np.zeros(len(models))
        x[index[ma]] = 1
        x[index[mb]] = -1
        X.append(x)
        y.append(1 if winner == "model_a" else 0)
        X.append(-x)
        y.append(0 if winner == "model_a" else 1)
    if not X:
        return pd.DataFrame(columns=["model", "bt_score"])
    clf = LogisticRegression(fit_intercept=False, penalty=None, max_iter=2000)
    clf.fit(np.asarray(X), np.asarray(y))
    scores = clf.coef_.ravel()
    scores = scores - scores.mean()
    return pd.DataFrame({"model": models, "bt_score": scores}).sort_values("bt_score", ascending=False).reset_index(drop=True)


def position_bias(df: pd.DataFrame) -> dict:
    d = add_winner_label(df)
    non_tie = d[d["winner"] != "tie"]
    if non_tie.empty:
        return {"n": 0, "model_a_win_rate": float("nan")}
    return {"n": int(len(non_tie)), "model_a_win_rate": float((non_tie["winner"] == "model_a").mean())}


def verbosity_bias(df: pd.DataFrame) -> dict:
    d = add_winner_label(df)
    if not {"response_a", "response_b"}.issubset(d.columns):
        return {"n": 0, "longer_response_win_rate": float("nan")}
    d = d[d["winner"] != "tie"].copy()
    d["len_a"] = d["response_a"].map(_text_len)
    d["len_b"] = d["response_b"].map(_text_len)
    d = d[d["len_a"] != d["len_b"]]
    if d.empty:
        return {"n": 0, "longer_response_win_rate": float("nan")}
    longer_won = ((d["len_a"] > d["len_b"]) & (d["winner"] == "model_a")) | ((d["len_b"] > d["len_a"]) & (d["winner"] == "model_b"))
    return {"n": int(len(d)), "longer_response_win_rate": float(longer_won.mean())}
