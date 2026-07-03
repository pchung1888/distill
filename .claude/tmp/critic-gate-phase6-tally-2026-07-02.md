# Critic Gate Tally -- distill Phase 6 (eval harness)

Date: 2026-07-02 (resolved after 5h-window reset, owner authorized continue)
Mode: autonomous (from beacon distill)
Artifact: evals/rubric.py, evals/run_evals.py, evals/golden/, tests/test_evals.py, CI step
Seat 4 routing: vex (eval data contracts / golden set)
Seat 5: iris directly (codex treated unavailable after the Phase 4 42-min stall;
a long watcher remains armed to record the late codex vote if it ever lands).

## Votes

| Seat | Reviewer | Vote |
|---|---|---|
| 1 | ms-mario | FIX |
| 2 | amanda | PASS |
| 3 | rhea+coin | FIX |
| 4 | vex (data route) | FIX |
| 5 | iris | FIX |

Verdict: FIX (4 of 5)

## Resolution

Same as Phase 4: launch instruction (fully autonomous one-shot) takes precedence
over the skill's halt-for-human default for FIX; all findings are mechanical and
in-scope. Fix pass + ms-mario re-check + advance.

## Consolidated findings to fix

1. [HIGH] The "LLM judge" is the pipeline critic re-invoked verbatim (same
   prompt/system/temperature/provider). Give the judge its OWN distinct prompt
   (per-fact grounding/entailment framing), its own marker, an optional
   --judge-provider override, and honest wording (it is only independent when
   provider differs; same-provider = self-consistency re-check).
   (ms-mario H1, iris headline)
2. Report must print pipeline critic confidence (spec wording) AND judge
   confidence as separate lines. (ms-mario M1, amanda nuance)
3. Mock-mode runtime caveat: when provider == mock, print a NOTE line that the
   run validates harness+rubric mechanics, not extraction quality. Also in the
   CI step name/comment. (ms-mario M2, iris)
4. Variance line: print n= case count and a small-N caveat; note mock values are
   hand-authored constants. (ms-mario M3, iris)
5. Meter the judge call: run_critic already returns the LLMResponse -- record a
   judge StageTrace per case and include it in totals (or a separate labeled
   judge line) so live cost is not understated. (rhea 1, iris)
6. Shield the judge path: any exception during the judge call makes that case
   FAILED (reported) instead of crashing the harness. (iris)
7. Provider-init/config errors (e.g. missing API key ValueError from
   get_provider/provider ctor) map to exit 2, not a raw traceback. (iris,
   ms-mario L1)
8. Missing tests: fact_floor fail; judge faithful=False fail; schema fail path;
   malformed mock_response.json -> exit 2; _check_mock_script_order guard fires
   -> exit 2; judge-error -> failed case. (ms-mario M4, rhea 2-3, vex 1)
9. Golden-JSON strictness: validate mock_response.json blocks with
   extra="forbid" variants so typo'd optional fields fail loudly at load time
   (production models keep extra="ignore"; the STRICTNESS lives in the eval
   loader only). (vex 2)
10. Surface judge issues/missing_points in the report for failing cases (cheap
    observability). (iris minor)
11. Document that schema_valid is tautological on the pipeline path (real for
    unit tests / future non-pipeline callers) -- docstring note. (iris)
