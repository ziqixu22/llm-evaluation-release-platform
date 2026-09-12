import pandas as pd
from llm_eval.arena import add_winner_label, preference_summary, position_bias


def sample():
    return pd.DataFrame({
        "model_a": ["m1","m1","m2","m3"],
        "model_b": ["m2","m3","m3","m1"],
        "winner_model_a": [1,0,1,0],
        "winner_model_b": [0,1,0,1],
        "winner_tie": [0,0,0,0],
    })


def test_winner_and_preference_summary():
    d = add_winner_label(sample())
    assert list(d["winner"]) == ["model_a","model_b","model_a","model_b"]
    s = preference_summary(d)
    assert set(s["model"]) == {"m1","m2","m3"}


def test_position_bias():
    b = position_bias(sample())
    assert b["n"] == 4
    assert b["model_a_win_rate"] == 0.5
