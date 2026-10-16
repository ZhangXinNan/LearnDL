# LearnDL — Versioning and Iteration Guidelines

> Document version: V1.1  
> Status: Approved baseline (streamlined revision)  
> Revised: 2026-10-16  
> Related documents: [Project Goals and Core Requirements](goals.md) · [Learning and Development Workflow](workflow.md)

## 1. Purpose of Versions

LearnDL is a long-term personal learning project. A project version represents **one small goal that has been completed and verified**, not software maturity, days spent studying, number of commits, or level of expertise.

**Rule: Each time one independently verifiable basic feature or small learning goal is completed, advance the project version by `0.1`.** Do not use three-part version numbers (such as `v0.0.1` or `v0.1.1`), and do not impose Semantic Versioning rules that require major versions to indicate breaking changes.

## 2. Version Format and Rollover

Project versions use `vMAJOR.MINOR`, for example:

```text
v0.1 → v0.2 → … → v0.9 → v1.0 → v1.1 → … → v1.9 → v2.0
```

- **`v0.0` is the one-time initialization baseline exception**: it records an agreed project purpose, goals, workflow, and versioning rules. It is not an algorithm-learning achievement and does not consume a `0.1` learning increment.
- Release the first completed small goal as `v0.1`.
- Advance the version once for each subsequent completed small goal.
- Use only digits `0`–`9` for the minor number. After `v0.9`, roll over to `v1.0`; do not write `v0.10`.
- `v1.0` is simply the next iteration after rollover. **It does not mean graduation, production readiness, or that all topics have been learned.**
- Git tags always use a lowercase `v` prefix (for example, `v0.3`).

## 3. Size of an Iteration

Each version must have **one primary goal**, for example:

- Implement and verify a forward propagation function.
- Derive and validate a gradient computation method.
- Analyze the effect of a hyperparameter on experiment results.
- Measure the basic performance baseline of an inference implementation.

One goal may include the necessary explanation of principles, code, tests, visualizations, and summary. Do not bundle unrelated learning goals together. Split goals that are too large; accept any small, verifiable goal without requiring equal amounts of work in every version.

The following normally **do not warrant a separate version**: fixing bugs in the current goal, improving tests or notes, tidying code, changing configuration, or making attempts that have not yet produced an accepted result. If “identify and resolve a specific issue” is itself a clearly defined small learning goal, it may be released as its own version.

## 4. When to Release a Version

After completing a small goal, release it in this order:

1. Check the previously defined acceptance criteria and confirm that the result has sufficient evidence.
2. Complete the necessary learning summary, and save the code, derivations, tests, or experiment records.
3. Add an entry for the version to the root-level `CHANGELOG.md`.
4. Commit the relevant changes, then create the corresponding Git tag (for example, `v0.3`).

**Do not release a version before it passes acceptance.** You may make as many commits as needed while learning; commit count is unrelated to version numbers. If a significant issue is discovered in previously released work, document the correction and fix it in subsequent work rather than casually rewriting an existing Git tag.

## 5. Minimal Version Record

`CHANGELOG.md` only needs the following information for each version, normally with the newest version first:

```markdown
## v0.X — Small Goal Title

- **Goal:** The single small goal completed in this version.
- **Artifacts:** Locations of the code, derivation, experiment, or notes.
- **Acceptance:** Evidence used to confirm completion.
- **Open issues:** None / remaining issues.
```

There is no need to write a release announcement for every ordinary commit, correction, or intermediate learning step. Keep the full explanation, experiments, and summary in the version's learning record; `CHANGELOG.md` serves only as an index.

## 6. Project Versions vs. Documentation Versions

These two version types are **independent**:

| Type | Example | Meaning |
| --- | --- | --- |
| Project learning version | `v0.1`, `v0.2`, `v1.0` | A completed and accepted small learning goal, recorded with a Git tag. |
| Guideline document version | `goals.md` V1.1, `workflow.md` V1.1 | A revision of the document itself, not a measure of learning progress. |

Editing the wording of the goals, workflow, or versioning guidelines must not automatically advance the learning project version by `0.1`. Update a guideline document's own version when it changes substantively.

## 7. Long-Term Maintenance Principles

- Maintain the project as an individual, prioritizing simplicity, clarity, and traceability. No elaborate release process is required.
- Do not plan a large fixed sequence of versions upfront. Choose the next small goal based on what has already been learned.
- The highest completed Git tag is the primary indicator of the current project version. If no tag exists, refer to the most recent accepted entry in `CHANGELOG.md`.
- Follow [`workflow.md`](workflow.md) for **how each version is studied, implemented, verified, and summarized**. This document does not duplicate the seven-step workflow.

---

**In one sentence: One small goal, one acceptance decision, one `0.1` version increment.**
