import pandas as pd
from llm_eval.judge import align_annotation_level, align_majority_level, agreement_metrics, release_gate


def test_pair_orientation_is_canonicalized():
    human = pd.DataFrame({
        "question_id": [1,1,1],
        "model_a": ["b","b","b"],
        "model_b": ["a","a","a"],
        "turn": [1,1,1],
        "winner": ["model_b","model_b","model_a"],
    })
    gpt = pd.DataFrame({
        "question_id": [1], "model_a": ["a"], "model_b": ["b"],
        "turn": [1], "winner": ["model_a"]
    })
    aligned = align_annotation_level(human, gpt)
    assert len(aligned) == 3
    assert agreement_metrics(aligned)["agreement"] == 2/3
    majority = align_majority_level(human, gpt)
    assert agreement_metrics(majority)["agreement"] == 1.0


def test_release_gate_passes_perfect_agreement():
    aligned = pd.DataFrame({
        "question_id": range(120), "model_a": ["a"]*120, "model_b": ["b"]*120,
        "turn": [1]*120, "human_winner": ["model_a"]*60 + ["model_b"]*60,
        "gpt4_winner": ["model_a"]*60 + ["model_b"]*60,
    })
    assert release_gate(aligned)["automated_release_gate"] is True
