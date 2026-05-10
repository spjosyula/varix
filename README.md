# varix

Your AI agent gives a different answer every time you run it. varix tells you which step is causing it.

That's the whole pitch. No fixes, no suggestions — just a clear answer to "where is the randomness coming from?"

## The problem

You run your agent. Same input, same model, `temperature=0`. Different output anyway. Different tool calls. Sometimes a different final answer.

If your agent has multiple steps — say, classify → plan → respond — the question is: *which step* is the source? The others are just passing the variance along to the next one. Existing observability tools show you traces side by side and leave the detective work to you.

## What varix does

It runs your agent N times, watches every step under identical inputs, and answers three questions:

1. **Where does the variance start?** A step whose output changes even with fixed inputs is the *source*. Steps that change only because their inputs change are downstream noise.
2. **Does it actually reach the user?** Sometimes a flaky step's variance gets absorbed by the next step. You'd be surprised how often. varix tells you whether it propagates or gets normalized away.
3. **Why is it happening?** It puts each source into a category — provider-side (the model itself routed your request differently), tool-side (a tool returned different results for the same inputs), ordering (same calls, different sequence), prompt-side (residual sampling/temperature), or time/state (timestamps, RNG, that kind of thing).

Every answer ships with a confidence level — high, medium, low, or "cannot verify" — that varix can back up with the actual evidence it used.

## What varix doesn't do (for now)

- **Not a fix-suggester.** It diagnoses; you fix. We trust you.
- **Not a continuous monitor.** Run it on demand, in CI, or while debugging.
- **Not an evaluation framework.** It doesn't judge whether outputs are correct or good.
- **No web UI.** Terminal output and a JSON artifact you can re-read.
- **One input at a time.** Not a regression test runner.
- **Not a model-comparison tool.** Comparing pipeline-with-A vs pipeline-with-B is out of scope.

## Install

```powershell
pip install varix[gemini]            # or [anthropic], [langgraph]
setx GEMINI_API_KEY "your-key-here"  # Windows: restart your terminal after
```

On macOS or Linux, set the API key with `export` and add the line to your shell profile (`~/.zshrc` or `~/.bashrc`) so it persists:

```bash
export GEMINI_API_KEY="your-key-here"
```

## Quickstart

Three lines of agent code, one varix command, and you have a diagnosis.

```python
# agent.py
from varix.adapters.gemini import GeminiSingleCallAdapter
adapter = GeminiSingleCallAdapter()
```

```powershell
varix run agent.py --input "Why is the sky blue?" -n 3 --max-cost 0.05
```

You'll see a short report in your terminal, a JSON artifact saved under `~/.varix/runs/`, and the exact cost the call billed.

---

# Cookbook

Six recipes.

## 1. Diagnose variance in a multi-step pipeline

Most agents are multi-step (intent → plan → respond, or retrieve → rerank → answer). The varix value prop is *which step* is the source. Write an adapter that exposes each step:

```python
# agent.py
from varix.core import AdapterCapabilities, PipelineRun, Step, StepGraph, StepRun
from datetime import UTC, datetime

class MyAgent:
    def capabilities(self):
        return AdapterCapabilities(
            exposes_fingerprint=True,
            exposes_tool_calls=True,
            supports_replay=False,
        )

    async def pipeline_structure(self, pipeline_input):
        return StepGraph(steps=(
            Step(id="intent",  name="classifier", index=0),
            Step(id="plan",    name="planner",    index=1),
            Step(id="respond", name="writer",     index=2),
        ))

    async def run_pipeline(self, pipeline_input, seed=None):
        started = datetime.now(tz=UTC)
        # ... call your model 3 times, build StepRun for each step ...
        return PipelineRun(
            run_id=f"r-{started.timestamp()}",
            step_runs=(...),
            started_at=started,
            finished_at=datetime.now(tz=UTC),
        )

    async def replay_step(self, step_id, fixed_inputs, seed=None):
        raise NotImplementedError("supports_replay=False")

adapter = MyAgent()
```

A complete production-shaped template lives in the dogfood folder; copy and adapt it.

## 2. Read the report

`varix run` answers one question: **is your pipeline nondeterministic, and where?**

If everything is stable, you'll see:

```
No nondeterminism found in agent.py.

n=3 | $0.0002 | 14s | analysis 5dfcc218
```

If varix found one or more sources, you'll see a ranked list — propagating sources first, absorbed sources after — with the suggested next commands:

```
Found 1 source of nondeterminism in agent.py.

  step `plan`  ->  prompt-side, propagates downstream

n=3 | $0.0002 | 14s | analysis 5dfcc218

Next:
  varix impact plan       see how much this changes your output
  varix explain plan      see the evidence varix used
```

If outputs were stable but varix detected provider routing variance (different `system_fingerprint` across runs), it surfaces that as a separate signal — your pipeline is fine *today* but the underlying infrastructure shifted, which can bite you later:

```
Your pipeline's outputs were stable across 3 runs.

  varix did detect provider routing changes during the runs:
    step `respond`  ->  fingerprint changed (fp_a -> fp_b)

  This didn't affect your output - but it means the provider routed
  your requests to different model infrastructure. Future runs may
  behave differently.

n=3 | $0.0002 | 14s | analysis 5dfcc218

Next:
  varix explain respond      see the fingerprint evidence
```

**Confidence levels appear in `varix explain`, not `varix run`:**
- `high` — varix has direct evidence (e.g., differing `system_fingerprint`).
- `medium` — variance remains after ruling out provider, tool, ordering, time signals; most likely sampling.
- `low` — heuristic match (e.g., a tool name like `get_current_time`).
- `cannot verify` — the adapter doesn't expose what's needed; varix refuses rather than guesses.

## 3. Drill into a finding

`varix run` gives you the summary. To see *why* varix said what it said for a specific step:

```powershell
varix explain plan
```

You'll get a classification block with the actual evidence varix used. No re-run, no extra cost — it reads the saved JSON. The shape of the block depends on the classification:

For **provider-side** variance, you see the fingerprint table:

```
step `plan` was classified as provider-side variance, high confidence.

Why this classification:
  Across 3 runs of `plan`, system_fingerprint changed:

    fp_8a2f3c   used in 2 runs
    fp_b91d04   used in 1 run

  Different fingerprints mean the provider routed your requests to
  different model infrastructure. This is variance from the provider
  side - there is nothing in your pipeline causing it.
```

For **prompt-side** residual (medium confidence — varix ruled out provider, tool, ordering, time/state), you see the four ruled-out facts plus the actual observed outputs so you can audit varix's judgment yourself:

```
step `plan` was classified as prompt-side variance, medium confidence.

Why this classification:
  Across 3 runs of `plan` with identical inputs:
    - provider fingerprints were stable (fp_8a2f3c in all 3)
    - no tool calls were made
    - no time/state markers detected in outputs
    - the outputs themselves differed

The 3 runs varix observed:
  run 1: "I'll explain Rayleigh scattering, then connect..."
  run 2: "Let me start with the physics: when sunlight..."
  run 3: "The blue color comes from a phenomenon..."
```

`tool-side`, `ordering`, and `time/state` classifications each render their own evidence — tool result diffs, sequence diffs, heuristic markers — in the same shape. When the adapter doesn't expose what's needed, varix says `cannot verify` and tells you which capability to enable.

## 4. Quantify downstream impact

You found that `plan` is a source of variance. Does that variance actually reach the user, or does the next step absorb it?

```powershell
varix impact plan
```

`varix impact` answers in one of three shapes:

```
plan's variance changes the final output.

  3 of 3 runs reached a different final answer.

confidence: high
analysis: 5dfcc218

Next:
  varix explain plan      see the evidence varix used
```

When variance partially propagates (e.g., 5 runs but only 3 reach a different final answer because 2 collapse to the same modal output), the prose carries that nuance:

```
plan's variance changes the final output.

  3 of 5 runs reached a different final answer; 2 were absorbed.
```

When the downstream pipeline normalizes the variance away:

```
plan's variance is absorbed before the final output.

  5 different plan outputs produced only 1 final answer across 5 runs.
  The downstream pipeline normalized the differences.
```

## 5. Re-render a past analysis

Reports are saved to `~/.varix/runs/<analysis_id>.json`. To see one again:

```powershell
varix show 5dfcc218-8f25-49c5-a8a2-6a513f740598
```

Same body as the original `varix run`, with one addition — the receipt grows a `ran X ago` segment so you have temporal context when re-reading:

```
n=3 | $0.0007 | 14s | analysis 5dfcc218 | ran 2 hours ago
```

The JSON is the source of truth — `show`/`explain`/`impact` all read it without re-running anything.

## 6. Cap cost

Always set `--max-cost`:

```powershell
varix run agent.py --input "..." -n 5 --max-cost 0.10
```

If a run pushes total spend over the budget, varix halts mid-loop, writes a partial artifact with the runs that completed, and exits with a clear note in the report's `WARNING:` block. No surprise bills.

## 7. Single-call shortcut (no custom adapter needed)

If you just want to check whether a single Gemini call is reproducible — no multi-step pipeline:

```python
# agent.py
from varix.adapters.gemini import GeminiSingleCallAdapter
adapter = GeminiSingleCallAdapter(model="gemini-2.5-flash-lite")
```

```powershell
varix run agent.py --input "Tell me a fun fact" -n 3 --max-cost 0.05
```

You'll get a single-step report. Useful for sanity-checking model-level determinism before building a pipeline on top.

---

## Command reference

```
varix run <pipeline> --input "..." [-n N] [--max-cost D]   localize + classify
varix impact <step-id>                                      quantify downstream effect
varix explain <step-id>                                     show evidence for a finding
varix show <analysis-id>                                    re-render a past report
```

`<pipeline>` accepts either a file path (`varix run agent.py`) or an importable string (`varix run my_module:my_pipeline`).

## Things to know

1. **It costs API tokens.** Cost depends on your model and pipeline size. A 3-run, 3-step Gemini Flash Lite pipeline is around $0.0002. A 5-run, 8-step Claude Opus run can hit $1+. Always set `--max-cost` so you don't get a surprise bill.
2. **Low N gives directional answers.** With N=3, varix can tell deterministic apart from nondeterministic. It can't reliably tell "30% flaky" apart from "60% flaky" — bump N higher if you need that.
3. **Prompt-side is the catch-all category.** It fires when nothing else fits. Medium confidence by design — the actual culprit is usually sampling, but if some non-obvious source slipped through, this is where it ends up.
4. **Time/state detection is best-effort.** varix catches the obvious cases — clock-named tools, timestamps in output. Anything more subtle is on you.
5. **Tool-side checks need real tool calls.** If your tools have side effects, you'll want a mock layer; varix won't blindly hit production tools during replay.
6. **Some adapters can't see provider fingerprints.** Anthropic's, for one. In those cases varix says "cannot verify" instead of guessing.

## How varix decides what to say

- **Refuse rather than guess.** If varix can't verify a finding, it says so — never inflated.
- **Cost is always visible.** No hidden bills.
- **Confidence is always labeled.** high, medium, low, or "cannot verify."
- **You control the knobs.** Number of runs, cost cap, what gets mocked, when to dig deeper.
- **Each command answers one question.** No flag soup, no surprising side effects.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to file an issue or open a pull request.

## License

MIT — see [LICENSE](LICENSE).
