# Changelog

## v0.2 — Complex Computation Graphs and Operators

- **Goal:** Support branching graphs, shared variables, gradient accumulation and natural arithmetic expressions with first-order autodiff.
- **Artifacts:** `steps/step11.py`–`step24.py` (excluding theoretical Step15), `dezero/`, `tests/test_v02.py`, [v0.2 learning record](docs/v0.2.md) and [Step01–24 notes](docs/深度学习入门2自制框架/steps.md).
- **Acceptance:** All existing 22 step scripts run; the v0.1 five-test suite and v0.2 regression suite pass; `import dezero` works. Final Git tag is maintained separately.
- **Open issues:** Tensor broadcasting-gradient reduction and higher-order derivatives are intentionally deferred to v0.4 and v0.3 respectively. Current package name is `dezero`, not `learndl`.

## v0.1 — Basic Automatic Differentiation

- **Goal:** Understand computation graphs and the chain rule by implementing first-order automatic differentiation for simple composite functions.
- **Artifacts:** `steps/step01.py`–`step04.py`, `steps/step06.py`–`step10.py`, and the [v0.1 learning record](docs/v0.1.md).
- **Acceptance:** Nine step scripts ran successfully; all five `step10.py` unit tests passed, including analytical and numerical gradient checks for a composite function. Reproduce with `python -m unittest discover -s steps -p 'step10.py' -v` from the project root (verified 2026-10-17).
- **Open issues:** Shared variables, gradient accumulation, branching computation graphs, and general operator overloads are intentionally deferred to v0.2. A `Function` instance must not be reused across calls in the current design.

## v0.0 — Project Initialization

- **Goal:** Establish the LearnDL project purpose, boundaries, learning workflow, and versioning rules.
- **Artifacts:** `README.md`, `docs/`, initial repository configuration.
- **Acceptance:** Reviewed documentation, clear separation from LearnOCR/PyTorch model practice, consistent internal links.
- **Open issues:** None required for initialization.
