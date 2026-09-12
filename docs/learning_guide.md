# Learning Guide — LLM Evaluation & Release Decision Platform

## 1. Business problem

A company wants to release a new chatbot/copilot model. The naive question is: "Which model has the best benchmark score?"

The real business question is broader:

- Do users actually prefer the new model?
- Can an automated LLM judge replace expensive human evaluation?
- Is the judge biased by response position or verbosity?
- Does judge reliability degrade on harder/multi-turn cases?
- Which slices can be auto-approved, and which need human review?

This project treats **evaluation itself as a measurement system**.

## 2. Why human preference is the outcome

For an interactive assistant, product success is ultimately tied to whether users prefer the response, not whether an internal evaluator produces a high score.

LMArena provides real pairwise user preferences across many production-grade models. Each row is a head-to-head comparison between two responses to the same prompt.

## 3. Pairwise preference metrics

For each model, count wins, losses, and ties across battles. A tie-adjusted win rate is:

`(wins + 0.5 * ties) / total battles`

This is easy to interpret but can be confounded because models do not face identical opponents.

## 4. Bradley-Terry ranking

The Bradley-Terry model estimates latent model strength from pairwise comparisons while accounting for opponent strength.

Conceptually:

`P(model i beats model j) = sigmoid(skill_i - skill_j)`

The project fits this as a pairwise logistic model after filtering models with too little data.

## 5. Position-bias audit

If model A wins much more than 50% merely because its answer appears first, the measurement process is biased.

We therefore compute the non-tie model-A win rate. A large deviation from 50% is a warning signal.

## 6. Verbosity-bias audit

LLM judges and humans may prefer longer answers even when length does not imply quality.

The project measures how often the longer response wins among comparisons where response lengths differ.

This is not proof of causal verbosity bias, but it is a useful measurement-risk diagnostic.

## 7. Why validate LLM-as-a-Judge?

Human evaluation is expensive and slow. Automated judges are attractive because they scale cheaply.

But before using a judge to approve model releases, we need to know how closely it tracks expert humans.

MT-Bench contains expert human preferences and GPT-4 pairwise judgments over the same model comparisons.

## 8. Human majority label

Multiple experts may evaluate the same pair. We aggregate human judgments into a majority label before comparing them with GPT-4.

If the human vote is tied, the aggregate label becomes a tie rather than pretending there is certainty.

## 9. Agreement

Exact agreement is:

`number of GPT-4 decisions matching human majority / aligned comparisons`

This is intuitive but does not account for agreement that could occur by chance.

## 10. Cohen's kappa

Cohen's kappa adjusts observed agreement for chance agreement.

Roughly:

- near 1: very strong agreement
- near 0: little beyond chance
- below 0: systematic disagreement

The project reports both agreement and kappa because a single metric is not enough.

## 11. Bootstrap confidence interval

The observed agreement is only an estimate from a finite sample.

We repeatedly resample aligned judgments and recompute agreement to obtain a 95% bootstrap confidence interval.

This answers: "How uncertain is our estimate of judge reliability?"

## 12. Slice reliability

Aggregate agreement can hide weak subgroups. The project therefore evaluates reliability by conversation turn.

If turn-2 reliability is materially worse than turn-1, multi-turn evaluation should remain human-reviewed even if overall agreement looks strong.

## 13. Release gate

The release gate is a business rule, not just a metric report.

Example thresholds:

- agreement >= 80%
- kappa >= 0.60
- enough sample size per slice

If every required slice passes, automated evaluation can be used as a release gate. Otherwise failing slices are routed to human review.

## 14. Business loop

The complete loop is:

1. Human preference defines product quality.
2. Pairwise analysis ranks models.
3. Bias audits test evaluation validity.
4. Human-vs-judge analysis estimates automation reliability.
5. Confidence intervals quantify uncertainty.
6. Slice analysis finds where automation breaks.
7. Release policy combines the evidence into PASS vs HUMAN REVIEW.
8. New model/prompt versions can be evaluated through the same regression framework.

## 15. Why this is not a toy project

It does not stop at "Model A scored 82%."

It includes:

- real user outcome data
- expert human labels
- automated judge validation
- statistical uncertainty
- bias diagnostics
- pairwise ranking
- slice analysis
- policy thresholds
- automated CI/full-data workflows
- a concrete release decision

## 16. 60-second interview structure

> I built an LLM evaluation and release-decision platform around a practical question: whether an AI team can trust an automated judge enough to gate model releases. I used LMArena human preferences as the product-quality outcome and estimated model quality with tie-adjusted preference rates and Bradley-Terry rankings, while auditing position and verbosity effects. Then I used MT-Bench expert labels to validate GPT-4 as an LLM judge, measuring exact agreement, Cohen's kappa, bootstrap confidence intervals, and reliability by conversation turn. Finally I converted those metrics into a release policy: high-confidence slices can be automated, while low-agreement or under-sampled slices are routed to human review. The key lesson is that an evaluator is itself a model and needs validation, uncertainty estimates, bias checks, and monitoring before it can make production decisions.

## 17. Questions to prepare for

- Why use human preference rather than benchmark accuracy as the product outcome?
- What problem does Bradley-Terry solve?
- Why can raw win rate be misleading?
- What is position bias?
- Does longer-response win rate prove verbosity bias?
- Why use Cohen's kappa in addition to agreement?
- Why bootstrap the confidence interval?
- Why aggregate human labels first?
- Why slice by turn?
- How would you extend this to safety/factuality/coding evaluations?
- How would you estimate the cost savings from evaluator automation?
- How would you monitor an evaluator over time?
