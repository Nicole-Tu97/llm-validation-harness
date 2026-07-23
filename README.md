# LLM Validation Harness — challenging a grounded banking assistant

An independent, reproducible harness that validates an LLM assistant the way a second-line model-risk
team would: it *challenges* the model empirically and writes a validation report. The system under test is
a grounded banking Q&A assistant — it must answer **only** from a provided policy context and **abstain**
when the answer isn't there (the pattern behind LLM-powered digital-banking and customer-service assistants).

The point isn't the assistant; it's the **validation around it** — the checks, thresholds, and evidence you
need before trusting an LLM in production.

> Runs out of the box on a deterministic **mock** assistant (no API key, fully reproducible). Set
> `USE_LLM=1` with `ANTHROPIC_API_KEY` to point the same harness at a real Claude model.

## What it checks
Each dimension is a check with a metric and a pass/fail threshold:

- **Faithfulness / hallucination** — does it answer out-of-scope questions instead of abstaining?
- **Abstention (conceptual soundness)** — does it correctly say "not in the policy" when it should?
- **Reproducibility & stability** — same question, repeated calls: same answer?
- **Robustness** — does re-wording the question change the answer?
- **Calibration / uncertainty** — is self-reported confidence backed by accuracy? (ECE + curve)
- **Fairness** — persona-swapped questions with identical facts must get identical answers.
- **Benchmarking** — a `guarded` (champion) vs. `naive` (challenger) configuration.

## Results (champion = guarded)

| Dimension | Value | Threshold | |
|---|--:|---|:--:|
| Hallucination rate | 0.00 | ≤ 0.15 | ✅ |
| Answerable correctness | 0.94 | ≥ 0.85 | ✅ |
| Reproducibility (temp 0.7) | 0.62 | ≥ 0.95 | ❌ |
| Robustness (paraphrase) | 0.78 | ≥ 0.90 | ❌ |
| Fairness consistency | 1.00 | = 1.0 | ✅ |
| ECE (calibration error) | 0.16 | ≤ 0.10 | ❌ |

![validation summary](outputs/validation_summary.png)

The harness surfaces real, actionable issues even on the "good" configuration: answers aren't deterministic
across repeated calls, they're sensitive to paraphrasing, and confidence overstates accuracy. Benchmarking
makes the value of the controls concrete — the `naive` configuration hallucinates **75%** of out-of-scope
questions versus **0%** for `guarded`. That delta *is* the control.

![calibration](outputs/calibration.png)

Full write-up, example failures, and recommendations: [`outputs/validation_report.md`](outputs/validation_report.md).

## Run
```bash
pip install -r requirements.txt
python3 src/harness.py                       # deterministic mock (no API key)
USE_LLM=1 ANTHROPIC_API_KEY=... python3 src/harness.py   # validate a real Claude model
```

## Structure
```
src/data.py       self-contained eval set: answerable / unanswerable / fairness-pair questions
src/assistant.py  system under test — grounded banking assistant (mock default, real Claude optional)
src/harness.py    the validation battery, metrics, plots, and report
outputs/          metrics.json, validation_report.md, validation_summary.png, calibration.png
```

## Notes
Illustrative work sample on synthetic data, not a production system. The mock assistant is deliberately
built with realistic flaws (some hallucination, imperfect calibration, phrasing sensitivity) so the harness
has something to catch — the same code runs unchanged against a real model.
