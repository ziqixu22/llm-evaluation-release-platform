# Interview Guide — LLM Evaluation & Release Platform

## 60-second walkthrough
This project treats an LLM release as a decision problem. Rather than demoing a chatbot, it evaluates model quality against human-preference and stress-test benchmarks, tests judge reliability and disagreements, slices failures, compares cost and quality, and defines release gates. The deliverable is a defensible recommendation: release, route selectively, or hold.

## Know these ideas
- **Human preference vs LLM judge:** human labels are the reference; automated judges scale evaluation but must be validated against humans.
- **Failure slices:** aggregate scores can hide unsafe or unreliable subsets.
- **Release gate:** a pre-defined quality/safety/cost condition required before launch.
- **Routing:** send requests to different models or humans according to difficulty, risk, latency, or cost.
- **Distribution shift:** benchmark success does not guarantee production success.

## Likely questions
**How would you tell whether an LLM judge is trustworthy?**  
Measure its agreement with independently labeled human examples, inspect systematic disagreement slices, and avoid using it as the sole release authority where agreement is weak.

**What would make you block release?**  
A material regression on a critical safety or quality slice, unreliable evaluator agreement, or a cost/latency profile that breaks the product constraint.

## Reproduce and learn
1. Start with the evaluation contract and identify the release decision.
2. Trace one benchmark from raw example to scored output.
3. Inspect a judge/human disagreement example.
4. Explain one failure slice and its proposed mitigation.
5. State the specific gate that would change release to hold.

## Honest boundary
Offline benchmarks are evidence, not proof of real-world performance; production monitoring remains necessary.