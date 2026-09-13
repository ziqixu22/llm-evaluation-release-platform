# LLM Evaluation & Release Decision Platform

Industry-style AI evaluation project built around a practical product question:

> **When can an AI product team trust an automated LLM judge, and when should evaluation be escalated to humans?**

The project treats evaluation as a **measurement and decision system**, not as a single benchmark score.

## Business loop

```text
real human preference outcomes
        ↓
model-quality / preference analysis
        ↓
automated judge predictions
        ↓
A/B reversal robustness audit
        ↓
symmetric judge estimate
        ↓
confidence + consistency routing
        ↓
automate trusted cases / human-review uncertain cases
        ↓
release gate
```

## Public data

### LMArena Human Preference 55K
Used as a product-quality layer for real pairwise user preferences across many LLMs.

### Chatbot Arena LLM Judges
A public 49,938-example single-turn dataset with human A/B/tie labels plus saved logits from multiple open-source LLM judges. The repository also provides **reversed-order judge predictions**, where response A and response B are swapped specifically to measure and mitigate positional sensitivity.

## What the platform evaluates

### Product outcome
- pairwise human preference
- tie-adjusted win rates
- Bradley-Terry ranking
- position and response-length diagnostics

### Automated evaluator quality
For each judge:
- original-order accuracy vs human labels
- reversed-order accuracy after mapping back to the original A/B frame
- symmetric accuracy after averaging original and reversed probabilities
- position consistency
- Cohen's kappa
- bootstrap confidence interval

### Human-in-the-loop routing
A case is eligible for automated evaluation only when:
1. the judge gives the same directional decision before and after A/B reversal; and
2. the symmetric prediction clears a configurable confidence threshold.

Unstable or low-confidence cases are routed to **human review**.

The release layer therefore measures both:
- **quality** — how often the judge matches human preference;
- **coverage** — how much evaluation work can safely be automated under the policy.

## Why this is not a toy project

The project goes beyond "compare model accuracy":

- uses real human preference outcomes;
- validates multiple automated judges against human labels;
- explicitly tests position robustness with counterfactual A/B reversal;
- separates evaluator accuracy from evaluator stability;
- implements confidence-based human escalation;
- converts metrics into a configurable release policy;
- includes modular Python code, unit tests, CI, full-data GitHub Actions, persisted results, and a learning/interview guide.

## Repository structure

```text
.
├── README.md
├── RESULTS.md
├── configs/release_policy.json
├── scripts/full_analysis.py
├── src/llm_eval/
│   ├── arena.py
│   ├── arena_judges.py
│   └── judge.py
├── results/
│   ├── metrics.json
│   ├── judge_audit.csv
│   ├── arena_preference_summary.csv
│   └── arena_bradley_terry.csv
├── tests/
├── docs/learning_guide.md
└── .github/workflows/
    ├── ci.yml
    └── full-analysis.yml
```

## Reproducibility

```bash
pip install -e '.[dev]'
pytest -q
python scripts/full_analysis.py
```

The full-analysis workflow downloads the public data and judge outputs, writes `RESULTS.md` and `results/*`, and uploads them as a GitHub Actions artifact.

## Resume-ready framing

> Built an LLM evaluation and release-decision platform using real human preference labels and multiple automated judges; audited judge reliability under A/B response reversal, implemented symmetric scoring and confidence/consistency-based human-review routing, and converted evaluator quality and automation coverage into configurable release gates.

## Interview takeaway

The central lesson is:

> **An LLM judge is itself a model. Before using it to approve another model, validate its agreement with humans, robustness to presentation order, uncertainty, and failure-routing policy.**

No real-company cost savings or production traffic are claimed; the project demonstrates the evaluation methodology and decision architecture with public data.

## Learn it for interviews

Use the project-specific [Interview Guide](docs/INTERVIEW_GUIDE.md) for a 60-second walkthrough, key concepts, likely questions, reproducible study steps, and the honest boundary of the work.
