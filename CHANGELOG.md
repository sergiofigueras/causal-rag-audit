# Changelog

All notable changes are documented in this file.

## 0.2.0 - 2026-09-14

- Adds a default `paper-v0.1` scoring profile that reproduces the manuscript's
  answer, abstention, citation-set, and partial-format parsing rules.
- Keeps scoreable answers when citation formatting is invalid and reports the
  format problem independently.
- Adds the opt-in `strict-exact` profile and custom Python answer/abstention judges.
- Reports per-world proof citation recall (PCR) and precision (PCP), including
  macro means and CI thresholds.
- Rejects reordered document IDs across paired worlds to prevent undeclared
  positional confounding.
- Adds paper-replay regression fixtures and conformance tests.

## 0.1.0 - 2026-09-14

- Introduces the versioned causal-audit dataset format.
- Adds strict structural and intervention validation.
- Adds callable, keyword-function, and JSON HTTP target adapters.
- Implements original, counterfactual, and necessary-evidence ablation runs.
- Reports observational support, paired responsiveness, abstention, citation validity, coverage CES, and strict CES.
- Adds JSON and Markdown reports, CI thresholds, examples, and cross-platform tests.
