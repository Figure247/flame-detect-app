---
name: evidence-driven-code-change
description: 'Use when implementing or debugging FlameDetect Electron, FastAPI, YOLO inference, upload, report, history, or model-management changes. Guides evidence-driven routing, minimal edits, and focused validation.'
argument-hint: 'Describe the requested behavior, file, symbol, or failing check.'
user-invocable: true
---

# FlameDetect Code Change

## When to Use

Use this skill for changes to the FlameDetect desktop app, including `main.js`, `preload.js`, `index.html`, `main.py`, model files, and runtime data handling.

## Procedure

1. Identify the most concrete available anchor: a named file, symbol, UI action, API route, failing behavior, command, or nearby implementation.
2. Trace the request across the smallest relevant boundary: renderer to preload to Electron main, or API route to model/data operation. If the first anchor only wires or forwards, follow one nearby hop to the owning implementation.
3. State one falsifiable hypothesis about the current behavior and one cheap check that could disconfirm it.
4. Choose the smallest reversible edit that tests the hypothesis. Preserve existing APIs, local patterns, and unrelated user changes.
5. Immediately run the narrowest available validation: `python -m py_compile main.py` for backend syntax, `npm run build` for packaging changes, or a focused manual UI/API check for behavior changes. Do not broaden exploration before this result is understood.
6. If validation fails and supports the hypothesis, repair the same slice and rerun the same check. If it falsifies the hypothesis, follow one nearby hop to the more direct controller and revise the hypothesis.
7. Repeat focused validation after each adjacent follow-up edit. Widen validation only when the change crosses a shared contract or the focused check passes but broader risk remains.
8. Finish by reporting the files changed, the validation performed, any remaining test gaps, and assumptions or blockers.

## Decision Rules

- Prefer an existing helper, abstraction, test, and command over a new one.
- Treat unrelated worktree changes as user-owned; do not revert them.
- Keep model artifacts and generated runtime files under the existing `models/`, `uploads/`, `reports/`, `logs/`, and `data/` conventions; do not hard-code a machine-specific model path.
- Preserve the Electron preload boundary and validate renderer changes through the exposed API rather than enabling unnecessary Node access.
- Do not use `start_backend.py` as a passive validation command because it installs dependencies before starting the server.
- Keep the first edit small enough that a failing check identifies the next move.
- Add or update focused tests when the behavior has a stable, testable boundary.
- If no executable validation exists, use the narrowest available static check and clearly state the limitation.

## Completion Checklist

- The controlling code path was identified.
- A falsifiable hypothesis and discriminating check were established before editing.
- The implementation is minimal and scoped to the request.
- At least one post-edit executable validation was run when available.
- Failures, residual risks, and test gaps are explicitly reported.
