# Dual-model Codex workflow

This repository uses a **Luna execution + Sol reasoning** workflow.

## Default routing

- Use `gpt-5.6-luna` for repository discovery, file reading, searches, commands,
  implementation, routine debugging, tests, packaging, and verification.
- Keep one execution owner. Luna remains responsible for the working tree and for
  validating the final result, even when Sol provides analysis.
- Do not ask Sol to repeat repository discovery or perform routine edits.

## Escalate to Sol only when needed

Use `gpt-5.6-sol` before implementation when at least one condition is true:

- The change requires an architectural or cross-module contract decision.
- A difficult bug remains unexplained after focused reproduction and evidence gathering.
- Several plausible solutions have materially different data-loss, compatibility,
  security, or performance trade-offs.
- The change touches persistence/recovery, concurrency, stage orchestration, security
  boundaries, or destructive migration behavior.
- Luna's first focused attempt failed and the next step requires deeper reasoning rather
  than more searching.

Do not escalate routine UI edits, localized fixes, dependency updates, packaging, or
straightforward test failures.

## Task Packet contract

Before asking Sol, Luna prepares a **1–3K token Task Packet**. Never send the complete
conversation, repository, logs, or broad file dumps. Include only evidence that changes
the decision.

```markdown
# Task Packet

## Goal
One concrete outcome and its acceptance criteria.

## Current behavior
Observed result, reproduction steps, and exact failure signal.

## Relevant code
- `path:line`: short excerpt or behavior summary
- `path:line`: short excerpt or behavior summary

## Constraints
Compatibility, safety, product, performance, and scope limits.

## Attempts and evidence
What Luna tried, what happened, and the smallest useful logs/test results.

## Decision needed
The precise questions Sol must answer.
```

Redact secrets, cookies, tokens, personal paths, and unrelated user data. Prefer symbols,
interfaces, state transitions, and short excerpts over whole files.

## Sol response contract

Ask Sol to return:

1. Root-cause or decision summary.
2. Recommended design and why.
3. Files/interfaces affected.
4. Ordered implementation steps.
5. Risks, rollback or compatibility notes.
6. Tests that prove the result.
7. Explicit unknowns or assumptions.

Sol should analyze and propose. It must not create a parallel implementation branch or
modify the working tree unless the user explicitly asks it to.

## Execution loop

1. Luna inspects the current tree and establishes a reproducible baseline.
2. Luna decides whether the escalation conditions are met.
3. If needed, Luna sends one compact Task Packet to Sol and receives one structured plan.
4. Luna reconciles that plan with the live working tree, then implements the smallest
   coherent change.
5. Luna runs focused tests first, then the relevant regression/build checks.
6. If evidence contradicts the Sol plan, Luna reports the contradiction and sends only
   the new decisive evidence in a revised Task Packet.
7. The final report states what Luna changed and tested, and mentions Sol only when its
   analysis materially affected the solution.

## Guardrails

- Never let Luna and Sol edit the same files concurrently.
- Never use model escalation as a substitute for reproducing a bug.
- Never resend unchanged context to Sol.
- One Sol consultation should normally be enough; a second requires new evidence.
- User instructions and repository safety rules override this routing policy.
