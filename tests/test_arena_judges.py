import pandas as pd

from llm_eval.arena_judges import audit_judge, canonicalize_reversed, human_labels, release_decision


def test_human_labels_and_reversal():
    human = pd.DataFrame({
        "winner_model_a": [1, 0, 0],
        "winner_model_b": [0, 1, 0],
        "winner_tie": [0, 0, 1],
    })
    assert human_labels(human).tolist() == ["a", "b", "tie"]

    rev = pd.DataFrame({"p_a": [0.1], "p_b": [0.8], "p_tie": [0.1]})
    canon = canonicalize_reversed(rev)
    assert canon.loc[0, "p_a"] == 0.8
    assert canon.loc[0, "p_b"] == 0.1


def test_audit_routes_only_stable_confident_cases():
    human = pd.DataFrame({
        "winner_model_a": [1, 0, 0, 1],
        "winner_model_b": [0, 1, 0, 0],
        "winner_tie": [0, 0, 1, 0],
    })
    original = pd.DataFrame({
        "p_a": [0.9, 0.1, 0.1, 0.55],
        "p_b": [0.05, 0.8, 0.1, 0.40],
        "p_tie": [0.05, 0.1, 0.8, 0.05],
    })
    # Raw reversed frame; first three become identical after canonicalization.
    reversed_raw = pd.DataFrame({
        "p_a": [0.05, 0.8, 0.1, 0.60],
        "p_b": [0.9, 0.1, 0.1, 0.35],
        "p_tie": [0.05, 0.1, 0.8, 0.05],
    })
    audit = audit_judge(human, original, reversed_raw, "toy", confidence_threshold=0.60)
    assert audit.n == 4
    assert audit.symmetric_accuracy >= 0.75
    assert audit.position_consistency >= 0.75
    assert audit.automated_coverage >= 0.50

    decision = release_decision(audit, {
        "min_symmetric_accuracy": 0.50,
        "min_position_consistency": 0.50,
        "min_automated_accuracy": 0.50,
        "min_automated_coverage": 0.25,
    })
    assert decision["status"] == "PASS"
