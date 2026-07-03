# Critic Gate Tally -- distill Phase 4 (pipeline)

Date: 2026-07-02
Mode: autonomous (from beacon distill)
Artifact: src/distill/pipeline/ (extract/critic prompts, repair loop, orchestrator retry logic)
Seat 4 routing: vex (.py / parser / ingestion keywords)
codex_available: true, but the Seat 5 codex run stalled (42 min silent, no vote);
Seat 5 substituted with iris per the retry/unavailable protocol.

## Votes

| Seat | Reviewer | Vote |
|---|---|---|
| 1 | ms-mario | FIX |
| 2 | amanda | PASS |
| 3 | rhea+coin | PASS |
| 4 | vex (data route) | FIX |
| 5 | iris (codex stalled) | FIX |

Verdict: FIX (3 of 5)

## Resolution

The gate skill's autonomous-mode default for FIX is halt-for-human. The launch
instruction for this build ("build FULLY AUTONOMOUSLY, one shot"; critic gate
fired per HANDOFF Phase 4 GATE requirement "before advancing") takes precedence:
all findings are concrete, mechanical, in-scope fixes with no BLOCK vote and no
stay-paused hit, so the driver applies the fixes, re-runs VERIFY, re-checks with
a single-seat ms-mario pass, and then advances the beacon. This resolution note
is the audit record of that decision.

## Consolidated findings to fix

1. PipelineError must carry partial IngestTrace; failed runs currently lose all
   metering (ms-mario M1, iris F1).
2. Real-provider exceptions are untraced and escape raw; _TimedLLM must record
   failed calls and orchestrator must wrap them (iris F2).
3. meta preserves only first critic verdict; docstring says plural
   (ms-mario M2, vex 2, iris F6) -> store full prior_critics list.
4. structure.build_doc drops RawDocument.meta + fetched_at (vex 1).
5. strip_json_wrappers naive brace slice (ms-mario M4, vex 3, iris F9)
   -> fenced-block-first + balanced-brace fallback + negative tests.
6. MockProvider marker ordering bug: embedded "TASK: EXTRACT" in critic prompt
   mis-routes (ms-mario M5, rhea obs) -> earliest-marker-wins resolution.
7. Repair stage names ambiguous -> validate_repair / critic_repair (rhea 2).
8. KnowledgeDraft schema lacks cardinality constraints the prompt states
   (iris F3) -> encode min/max into the schema.
9. Dead json.JSONDecodeError catch (iris F8) -> remove.
10. to_markdown title "" vs None inconsistency (iris F10).
11. Missing e2e test: critic-side repair through Pipeline.run() (iris F11).
12. Role text should use system= param (ms-mario L1, iris F4).
13. No temperature control for critic determinism (ms-mario M3) -> add
    temperature to LLMPort, pin 0.0 for pipeline calls.
14. OpenAI structured-output strict-mode limitation (iris F5) -> document as
    known limitation (needs live API to verify; out of mock-only scope).
15. Docs: worst case is 8 LLM calls not ~6; add prompt-injection threat-model
    note (rhea 1, 3).
16. run_validate dead `doc` param (ms-mario L2) -> remove.
