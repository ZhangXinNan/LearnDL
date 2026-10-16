# LearnDL — Learning and Development Workflow

> Document version: V1.1  
> Status: Approved baseline (streamlined revision)  
> Revised: 2026-10-16  
> Related documents: [Project Goals and Core Requirements](goals.md) · [Versioning and Iteration Guidelines](versioning.md)

## 1. Purpose

This document defines **how to complete a small learning goal in each LearnDL iteration**. LearnDL is maintained by one person over the long term. The workflow exists to deepen understanding of algorithms, retain verification evidence, and preserve learning outcomes—not to create administrative overhead.

> **One small goal → Understand the principles → Choose a simple approach → Practice firsthand → Verify and analyze → Summarize.**

The default workflow has seven steps. **These are steps for thinking and checking, not seven required documents.** Steps may be adjusted or combined for theory, implementation, experimentation, or optimization goals. Return to earlier steps whenever problems arise.

## 2. Seven-Step Workflow

```text
1. Goal and acceptance criteria
            ↓
2. Explanation of principles
            ↓
3. Approach and implementation method
            ↓
4. Practice / implementation (as needed)
            ↓
5. Testing, verification, and analysis ── Not passed → Return to 2 / 3 / 4
            ↓
6. Visualization (optional)
            ↓
7. Summary and archiving
```

### 1. Goal and Acceptance Criteria

**Question: What is the one thing to accomplish, and what would count as done?**

- Define one main goal and, if helpful, what is out of scope. Split goals that are too large.
- Specify the problem, inputs and outputs, constraints, and expected outcome; choose suitable metrics or a comparison baseline.
- Write down usually 1–3 verifiable acceptance criteria in advance.

**Minimum output:** A one-sentence goal and its acceptance criteria.

### 2. Explanation of Principles

**Question: Why does this work, and how does the core computation happen?**

- Explain concepts, algorithmic mechanisms, key formulas, and variable or tensor dimensions when relevant.
- Describe main assumptions, applicability, and limitations; check key derivations independently.
- Fill mathematical gaps as needed, and consider numerical stability without endlessly expanding prerequisite study.

**Minimum output:** A concise explanation, derivation, or example that your future self can understand.

### 3. Approach and Implementation Method

**Question: What is the simplest, clearest way to achieve the goal?**

- Choose a language and tools suited to the goal. NumPy is useful for transparent calculations; PyTorch may serve as a numerical reference, but reproducing full models belongs to separate projects.
- Briefly describe the key steps, inputs and outputs, necessary dependencies, and verification method.
- Prefer minimal implementations, fewer abstractions, and fewer dependencies. Explore low-level details or performance optimization only when they serve the learning goal.

**Minimum output:** A few lines describing the approach and verification plan; no separate design document is required.

### 4. Practice / Implementation

**Question: Can I do the essential work myself?**

- **Algorithm goals:** Implement the essential computation path with clear, executable logic.
- **Theory goals:** Work through a derivation, example, or hand calculation; do not force a code implementation.
- **Experiment goals:** Reuse an existing implementation when appropriate rather than rewriting the same component.
- When errors, anomalies, or unexpected results appear, identify causes, fix them, and record important lessons. Avoid premature abstraction and complex architecture.

**Minimum output:** Code, a derivation, or an experiment record appropriate to the goal.

### 5. Testing, Verification, and Result Analysis

**Question: What evidence shows that the result is correct, and what does the experiment tell us?**

- Use evidence appropriate to the goal: hand calculations or known results, unit tests, reference implementations, gradient checks, boundary tests, or controlled experiments.
- Use reasonable numerical tolerances; check precision, stability, operation order, and related factors when necessary.
- For important experiments, record data sources, configurations, random seeds, execution environments, and results so they can reasonably be reproduced.
- For performance optimization, measure the baseline first, keep conditions comparable, and report metrics such as latency, throughput, or memory alongside correctness and accuracy checks.
- Analyze observations, possible causes, and limitations. If the acceptance criteria are not met, return to earlier steps and revise the work.

**Minimum output:** Verification method, key results, interpretation, and a pass/fail conclusion.

### 6. Visualization (Optional)

**Question: Would a chart explain the result better than numbers or words?**

Create curves, computation graphs, error distributions, model diagrams, or performance comparisons only when they help understanding. Label axes, units, and conditions. If a numerical table or short explanation is sufficient, **skip this step**. Result analysis belongs in Step 5 and must not be skipped just because no visualization is needed.

**Minimum output:** A chart and short explanation if useful; otherwise nothing is required.

### 7. Summary and Archiving

**Question: What have I actually learned, and what remains unclear?**

- Summarize the principles, implementation choices, verification evidence, key issues, and limitations in your own words.
- Optionally note the level of mastery (L1–L5 in `goals.md`). Record open questions without automatically expanding the current version's scope.
- Once acceptance criteria are met, record the result and release the version according to `versioning.md`. **Do not release a version that has not passed acceptance.**

**Minimum output:** A short summary, acceptance result, and links or paths to the artifacts.

## 3. Applying the Workflow to Different Goals

| Goal type | Main focus | May be omitted or simplified | Required evidence |
| --- | --- | --- | --- |
| Algorithm implementation | Principles, essential computation, correctness | Step 6: visualization | Implementation, tests or reference comparison, conclusion |
| Theoretical understanding | Explanation, derivation, concrete examples | Simplify Step 3; omit code in Step 4 and visualization in Step 6 | Independent explanation, checked derivation/example, conclusion |
| Comparative experiment | Hypothesis, variables, metrics, result analysis | Reimplementation may be skipped; Step 6 as needed | Reproducible conditions, data, interpretation |
| Inference optimization | Correctness, performance baseline, fair measurement | Unrelated derivations; Step 6 as needed | Before/after measurements, environment conditions, accuracy or correctness checks |

**Never omit a clear goal, appropriate verification evidence, or an interpretable conclusion.** Adapt the remaining steps to the work.

## 4. Minimal Record Format

By default, keep **one Markdown learning record per project version**. Store code, tests, or figures separately only when needed. Do not create seven files for the seven steps.

```markdown
# v0.X — Small Goal Title

## Goal and Acceptance
- Goal:
- Completion criteria:

## Principles and Approach
- Core principles / key formulas:
- Implementation or experiment method (language, libraries, steps, and rationale):

## Practice and Verification
- Code / derivation / experiment location:
- Verification method, results, and analysis:
- Visualization (only if useful):

## Summary
- What I learned:
- Limitations and open questions:
- Mastery level (optional):
- Acceptance: Passed / Not passed
```

Combine headings for very small goals; add formulas, tables, or links for more complex ones. Do not create content merely to fill out a template.

## 5. Minimum Checklist for Completing an Iteration

- [ ] The main goal and acceptance criteria are clear.
- [ ] I can explain the essential principles or approach myself.
- [ ] There is an implementation, derivation, or experiment result appropriate to the goal.
- [ ] Verification meets the acceptance criteria; important results can reasonably be reproduced, and issues or limitations are explained.
- [ ] A short summary and necessary artifact locations are recorded.
- [ ] After acceptance, the version is recorded and tagged in Git according to the versioning rules.

## 6. Principles for Solo Maintenance

- No mandatory Issues, pull requests, elaborate branching schemes, code reviews, or project-management platforms. Use them only when useful.
- Iteration, trial and error, and multiple commits are welcome. Prioritize **clarity, understandability, and ease of rerunning** code and experiments.
- No fixed programming language, framework, or complex directory structure is required. Extract shared code only when a real maintenance need emerges.
- Avoid long reports for every exercise, and link to existing reference material rather than copying it unnecessarily.
- If the goal is not yet met, keep improving the work. Do not lower verification standards just to release a version.

---

**Document responsibility:** This document explains “How do we carry out each learning exercise?” For the learning direction and quality baseline, see `goals.md`. For version numbers, Git tags, and release records, see `versioning.md`.
