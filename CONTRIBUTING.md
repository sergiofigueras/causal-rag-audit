# Contributing

Contributions are welcome through focused pull requests.

1. Create a Python 3.10 or newer virtual environment.
2. Install the package with `python -m pip install -e .`.
3. Run `make check` before opening a pull request.
4. Add standard-library `unittest` coverage for behavioral changes.
5. Keep the core dependency-free. New runtime dependencies require a documented reason.

Audit datasets must document their intervention unit, answer-separation check, proof sets, ablation rationale, nuisance controls, and limitations. Automatically generated cases require human review before they are used to make consequential claims.
