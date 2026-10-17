# Changelog

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
