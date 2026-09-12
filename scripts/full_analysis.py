from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from datasets import load_dataset

from llm_eval.arena import preference_summary, bradley_terry_scores, position_bias, verbosity_bias
from llm_eval.arena_judges import read_judge_logits, audit_judge, release_decision

OUT = Path("results")
OUT.mkdir(exist_ok=True)
TMP = Path(".tmp_judges")
TMP.mkdir(exist_ok=True)

JUDGES = [
    "Meta-Llama-3-8B-Instruct",
    "Mistral-7B-Instruct-v0.1",
    "Starling-LM-7B-alpha",
]
BASE = "https://huggingface.co/datasets/potsawee/chatbot-arena-llm-judges/resolve/main"


def pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{100*x:.2f}%"


def download_judge(name: str, reversed_: bool = False) -> Path:
    folder = "llm-judges-reversed" if reversed_ else "llm-judges"
    path = TMP / f"{name}{'-reversed' if reversed_ else ''}.jsonl"
    urlretrieve(f"{BASE}/{folder}/{name}.jsonl", path)
    return path


def main():
    policy = json.loads(Path("configs/release_policy.json").read_text())

    # Product outcome layer: real user pairwise preferences.
    arena = load_dataset("lmarena-ai/arena-human-preference-55k", split="train").to_pandas()
    pref = preference_summary(arena)
    bt = bradley_terry_scores(arena, min_battles=100)
    pos = position_bias(arena)
    verb = verbosity_bias(arena)
    pref.to_csv(OUT / "arena_preference_summary.csv", index=False)
    bt.to_csv(OUT / "arena_bradley_terry.csv", index=False)

    # Measurement layer: human-labelled Arena outcomes vs automated judges.
    human = load_dataset("potsawee/chatbot-arena-llm-judges", split="train").to_pandas()
    audits = []
    decisions = []
    for name in JUDGES:
        original = read_judge_logits(download_judge(name, False))
        reversed_probs = read_judge_logits(download_judge(name, True))
        audit = audit_judge(
            human,
            original,
            reversed_probs,
            judge_name=name,
            confidence_threshold=float(policy["confidence_threshold"]),
        )
        audits.append(audit.to_dict())
        decisions.append(release_decision(audit, policy))

    audit_df = pd.DataFrame(audits).sort_values("symmetric_accuracy", ascending=False)
    audit_df.to_csv(OUT / "judge_audit.csv", index=False)
    best = audit_df.iloc[0].to_dict()
    best_decision = next(d for d in decisions if d["judge"] == best["judge"])

    metrics = {
        "human_preference": {
            "rows": int(len(arena)),
            "unique_models": int(len(set(arena["model_a"]).union(arena["model_b"]))),
            "position_bias": pos,
            "verbosity_bias": verb,
            "top_models_by_tie_adjusted_win_rate": pref.head(10).to_dict(orient="records"),
            "top_models_by_bradley_terry": bt.head(10).to_dict(orient="records"),
        },
        "judge_validation": {
            "human_labeled_rows": int(len(human)),
            "judges": audits,
            "release_decisions": decisions,
            "selected_judge": best["judge"],
            "selected_release_status": best_decision["status"],
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    md = [
        "# Verified Full-Data Results",
        "",
        "## 1. Human preference as the product outcome",
        "",
        f"- LMArena battles analyzed: **{len(arena):,}**",
        f"- Unique models represented: **{metrics['human_preference']['unique_models']}**",
        f"- Non-tie model-A win rate: **{pct(pos['model_a_win_rate'])}**",
    ]
    if verb.get("n"):
        md.append(f"- Longer-response win rate among eligible non-tied pairs: **{pct(verb['longer_response_win_rate'])}**")

    md += [
        "",
        "## 2. Automated judge validation on human-labelled Arena outcomes",
        "",
        f"Human-labelled examples: **{len(human):,}**",
        "",
        "| Judge | Symmetric acc. | Position consistency | Auto coverage | Auto acc. | Human review | Release |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    decisions_by_name = {d["judge"]: d for d in decisions}
    for r in audit_df.to_dict(orient="records"):
        d = decisions_by_name[r["judge"]]
        md.append(
            f"| {r['judge']} | {pct(r['symmetric_accuracy'])} | {pct(r['position_consistency'])} | "
            f"{pct(r['automated_coverage'])} | {pct(r['automated_accuracy'])} | {pct(r['human_review_rate'])} | {d['status']} |"
        )

    md += [
        "",
        "## 3. Release decision",
        "",
        f"Selected candidate judge: **{best['judge']}**",
        f"Release status: **{best_decision['status']}**",
        "",
        "The platform does not treat a raw judge score as sufficient evidence for automation. It first checks whether the judge remains directionally stable when answer order is reversed, then averages original/reversed probabilities to reduce position sensitivity. Only stable, sufficiently confident cases are eligible for automation; the rest are routed to human review.",
        "",
        "## 4. Business takeaways",
        "",
        "1. Human preference is the product outcome; automated judges are measurement tools.",
        "2. Judge quality should be evaluated on both correctness and robustness to answer order.",
        "3. Averaging original and reversed predictions reduces sensitivity to presentation order.",
        "4. A hybrid system can trade automation coverage for higher reliability by escalating unstable or low-confidence cases.",
        "5. Release gates should make failure visible instead of silently approving a weak evaluator.",
        "",
        "## 5. What this project does not claim",
        "",
        "- No claim that one judge is universally best across every task domain.",
        "- No claim that benchmark agreement is equivalent to real production quality.",
        "- No fabricated business savings; the project demonstrates a reproducible evaluation and routing framework.",
    ]
    Path("RESULTS.md").write_text("\n".join(md))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
