# LLM Evaluation & Release Decision Platform

Industry-style AI evaluation project for answering a practical product question:

> **Can an AI product team trust an automated evaluator enough to make model-release decisions, and where should human review remain in the loop?**

The project uses two real public evaluation datasets:

- **LMArena Human Preference 55K** — 55K+ real user comparisons across 70+ LLMs for product-quality / preference outcomes.
- **MT-Bench Human Judgments** — expert human pairwise judgments plus GPT-4 pairwise judgments for measuring LLM-as-a-Judge reliability.

The workflow is:

```text
real human preferences
        ↓
quality / ranking analysis
        ↓
position + verbosity bias audit
        ↓
expert-human vs GPT-4 judge agreement
        ↓
agreement / Cohen's κ / slice reliability
        ↓
release-gate policy
        ↓
automate safe slices + route uncertain slices to humans
```

## Why this is a business project

A company evaluating a new chatbot, copilot, or support assistant does not only need a benchmark score. It needs to know:

1. whether users actually prefer one model over another;
2. whether the evaluation itself is biased;
3. whether an automated judge tracks expert humans closely enough;
4. whether that reliability holds across relevant slices;
5. whether a model/prompt change should ship, be blocked, or require human review.

This repository converts those questions into a reproducible evaluation and release-decision system.

## Planned verified outputs

The full-data GitHub Actions workflow writes the final verified numbers to `RESULTS.md` and `results/*.json`.

## Repository structure

```text
.
├── README.md
├── RESULTS.md                  # generated from full public data
├── configs/
│   └── release_policy.json
├── scripts/
│   └── full_analysis.py
├── src/llm_eval/
│   ├── arena.py
│   └── judge.py
├── results/
├── tests/
├── docs/
│   └── learning_guide.md
├── .github/workflows/
│   ├── ci.yml
│   └── full-analysis.yml
└── pyproject.toml
```

## Methods

### Human preference / model quality
- pairwise user-preference outcomes
- tie-adjusted preference rates
- Bradley-Terry ranking
- bootstrap confidence intervals
- ranking / sample-size filters

### Evaluation-quality audit
- position-bias analysis
- response-length / verbosity-bias analysis
- human-majority aggregation
- GPT-4 judge vs expert-human agreement
- Cohen's kappa
- turn-level reliability slices

### Release decision
The release policy treats the evaluator as a measurement instrument. A judge can only automate a slice when agreement and reliability thresholds are met; otherwise the slice is routed to human review.

## Data provenance

- LMArena Human Preference 55K is released under Apache-2.0 and contains 55K+ real-world preference battles across 70+ LLMs.
- MT-Bench Human Judgments contains 3,355 expert human annotations and 2,400 GPT-4 pairwise judgments for six models on 80 MT-Bench questions.

Raw data is downloaded during the full-analysis workflow and is not committed to this repository.

## Resume intent

After the full workflow completes, resume bullets should use only verified metrics from `RESULTS.md`; no accuracy, agreement, or business-impact number is invented in advance.