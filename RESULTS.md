# Verified Full-Data Results

## 1. Human preference as the product outcome

- LMArena battles analyzed: **57,477**
- Unique models represented: **64**
- Non-tie model-A win rate: **50.52%**
- Longer-response win rate among eligible non-tied pairs: **61.62%**

## 2. Automated judge validation on human-labelled Arena outcomes

Human-labelled examples: **49,938**

| Judge | Symmetric acc. | Position consistency | Auto coverage | Auto acc. | Human review | Release |
|---|---:|---:|---:|---:|---:|---|
| Starling-LM-7B-alpha | 47.44% | 66.37% | 61.00% | 53.48% | 39.00% | HUMAN_REVIEW_REQUIRED |
| Meta-Llama-3-8B-Instruct | 46.92% | 63.40% | 61.84% | 52.85% | 38.16% | HUMAN_REVIEW_REQUIRED |
| Mistral-7B-Instruct-v0.1 | 44.03% | 67.29% | 56.39% | 51.12% | 43.61% | HUMAN_REVIEW_REQUIRED |

## 3. Release decision

Selected candidate judge: **Starling-LM-7B-alpha**
Release status: **HUMAN_REVIEW_REQUIRED**

The platform does not treat a raw judge score as sufficient evidence for automation. It first checks whether the judge remains directionally stable when answer order is reversed, then averages original/reversed probabilities to reduce position sensitivity. Only stable, sufficiently confident cases are eligible for automation; the rest are routed to human review.

## 4. Business takeaways

1. Human preference is the product outcome; automated judges are measurement tools.
2. Judge quality should be evaluated on both correctness and robustness to answer order.
3. Averaging original and reversed predictions reduces sensitivity to presentation order.
4. A hybrid system can trade automation coverage for higher reliability by escalating unstable or low-confidence cases.
5. Release gates should make failure visible instead of silently approving a weak evaluator.

## 5. What this project does not claim

- No claim that one judge is universally best across every task domain.
- No claim that benchmark agreement is equivalent to real production quality.
- No fabricated business savings; the project demonstrates a reproducible evaluation and routing framework.
