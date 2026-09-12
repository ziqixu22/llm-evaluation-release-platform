# Learning Guide — LLM Evaluation & Release Decision Platform

## 1. Business problem

A company is deciding whether an automated LLM judge can help evaluate a chatbot/copilot release.

The naive question is:

> Which judge has the highest accuracy?

The real product question is:

> Which evaluation decisions are reliable enough to automate, and which should be escalated to humans?

That means the project must evaluate **both correctness and robustness**.

## 2. Why human preference is the ground truth layer

For interactive AI products, the ultimate quality signal is whether users prefer one response over another.

The project uses real pairwise human preference outcomes with three possible labels:

- response A wins
- response B wins
- tie

This lets the evaluator be tested against the outcome it is supposed to approximate.

## 3. Product-quality layer

The LMArena preference dataset is used to understand model-level preference behavior.

For each model we can compute:

`tie-adjusted win rate = (wins + 0.5 × ties) / total battles`

This is interpretable, but opponents differ across models.

So the project also uses a Bradley-Terry model:

`P(i beats j) = sigmoid(skill_i - skill_j)`

Bradley-Terry estimates latent relative strength while accounting for opponent quality.

## 4. Why an LLM judge needs its own validation

Using another LLM as an evaluator is attractive because it is fast and scalable.

But the evaluator can fail because of:

- factual mistakes
- preference differences from humans
- position bias
- verbosity bias
- unstable decisions
- overconfidence

So an automated judge cannot be treated as an oracle.

## 5. Counterfactual A/B reversal

The Chatbot Arena LLM Judges dataset provides two predictions for the same example:

1. original presentation: answer A, then answer B
2. reversed presentation: answer B, then answer A

After the reversed prediction is mapped back to the original A/B frame, a robust judge should usually make the same decision.

Example:

```text
Original order
A = response X
B = response Y
Judge → A wins

Reversed order
A = response Y
B = response X
Judge → B wins
```

After mapping the reversed decision back to the original frame, both say **response X wins**.

That is position-consistent.

## 6. Position consistency

The project measures:

`position consistency = fraction of examples where original and reversed decisions agree after canonicalization`

This is different from accuracy.

A judge can be:

- accurate but unstable;
- stable but systematically wrong;
- both accurate and stable;
- neither.

That is why one metric is not enough.

## 7. Symmetric prediction

To reduce dependence on presentation order, the platform averages the probability vector from:

- original presentation
- canonicalized reversed presentation

For each example:

`symmetric_probability = (p_original + p_reversed_canonical) / 2`

The final symmetric label is the class with the highest averaged probability.

This is a simple form of counterfactual robustness averaging.

## 8. Accuracy and Cohen's kappa

Accuracy measures exact agreement with human labels.

Cohen's kappa additionally adjusts for chance agreement.

Because the task has three classes — A, B, tie — both are useful.

The project reports:

- original accuracy
- reversed accuracy
- symmetric accuracy
- symmetric Cohen's kappa

## 9. Bootstrap confidence interval

A measured accuracy is still a sample estimate.

The platform repeatedly resamples the evaluation examples and recalculates accuracy to obtain a bootstrap 95% confidence interval.

This communicates uncertainty instead of pretending one point estimate is exact.

## 10. Human-in-the-loop routing

The project does not force every evaluation into automation.

A case is eligible for automated evaluation only when:

1. original and reversed predictions are directionally consistent; and
2. the symmetric prediction exceeds the configured confidence threshold.

Otherwise:

`route → human review`

This creates two additional business metrics:

### Automation coverage

`automated cases / all cases`

### Automated-case accuracy

`human agreement among cases approved for automation`

These two metrics reveal the central business trade-off:

> Higher confidence thresholds may improve reliability but reduce automation coverage.

## 11. Release gate

The release gate combines several checks:

- minimum symmetric accuracy
- minimum position consistency
- minimum automated-case accuracy
- minimum automation coverage

If a candidate judge fails the policy, the system returns:

`HUMAN_REVIEW_REQUIRED`

instead of silently approving the evaluator.

The exact thresholds are configurable in `configs/release_policy.json` and should be treated as product-policy choices, not universal scientific constants.

## 12. Complete business loop

The project can be explained as:

```text
Human preferences
↓
What users value
↓
Automated LLM judge
↓
Does it match humans?
↓
Does it survive A/B reversal?
↓
How confident is the symmetric decision?
↓
Automate stable/high-confidence cases
↓
Escalate unstable/uncertain cases
↓
Release gate
```

## 13. Why this is not a toy project

A toy project might report one accuracy number.

This project includes:

- real human preference labels
- pairwise model-quality analysis
- multiple automated judges
- position-bias counterfactuals
- symmetric probability aggregation
- confidence intervals
- human-in-the-loop routing
- configurable release policy
- unit tests
- CI
- full-data workflow
- persisted result artifacts

## 14. 60-second interview version

> I built an LLM evaluation and release-decision platform around the question of when an automated judge can be trusted and when evaluation should remain human-in-the-loop. I used real human pairwise preferences as the outcome, then evaluated multiple LLM judges on the same A/B/tie task. A key reliability test was counterfactual answer-order reversal: I compared each judge's original prediction with its prediction after swapping response A and B, mapped both back to the same frame, and measured position consistency. I then averaged the original and reversed class probabilities to create a more order-robust symmetric prediction. Instead of automating every decision, the system only automates cases that are position-consistent and sufficiently confident, while routing the rest to human review. Finally, I convert evaluator accuracy, stability, automation coverage, and automated-case accuracy into a configurable release gate. The main lesson is that an LLM judge is itself a model and needs validation and failure routing before it can be trusted to evaluate another model.

## 15. Questions you should be able to answer

- Why use human preference labels?
- Why is raw judge accuracy insufficient?
- What is position bias?
- Why reverse A and B?
- How do you canonicalize the reversed prediction?
- Why average original and reversed probabilities?
- Accuracy vs Cohen's kappa?
- Why use a bootstrap confidence interval?
- What is automation coverage?
- Why not automate low-confidence cases?
- How would you choose the confidence threshold in a real company?
- What would happen if human-review capacity were limited?
- How would you extend this to safety, factuality, coding, or policy compliance evaluations?
- How would you monitor evaluator drift after model updates?

## 16. What not to claim

Do not claim:

- real production deployment;
- real company cost savings;
- universal superiority of one judge;
- that benchmark agreement guarantees production quality.

The defensible claim is that you built and validated a reproducible **LLM evaluation + human-review routing + release-policy framework** using public human-labeled data.