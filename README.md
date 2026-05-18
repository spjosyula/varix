# varix

Your agent gives a different answer every run — same input, same model, `temperature=0`. varix tells you which step is causing it.

No fixes and suggestions but an answer to *where the randomness is coming from.*

## The problem

Multi-step agent (classify → plan → respond). Same input, same model, same temperature. Different output anyway. The question is: **which step** is the source? The others just pass variance along. Existing trace viewers show you N runs side by side and leave the detective work to you.

## What varix does

Runs your agent N times on identical inputs and answers three questions:

1. **Where does the variance start?** A step whose output changes with fixed inputs is the *source*. Steps that change only because their inputs changed are downstream noise.
2. **Does it reach the user?** Sometimes a flaky step's variance is absorbed by the next step. varix tells you whether it propagates.
3. **Why?** It puts each source into one of five categories — provider, tool, ordering, prompt, time/state — each with labeled confidence (`high`, `medium`, `low`, or `cannot verify`).

## Install

```bash
pip install varix[gemini]
```

Set your Gemini API key:

```bash
# macOS / Linux
export GEMINI_API_KEY="your-key-here"

# Windows (PowerShell — restart your terminal after)
setx GEMINI_API_KEY "your-key-here"
```

Other providers and frameworks (Anthropic, OpenAI, LangGraph, CrewAI, ...) work through a small custom adapter — see [Multi-step pipelines](#multi-step-pipelines).

## 60-second quickstart

Check whether a single Gemini call is reproducible:

```python
# agent.py
from varix.adapters.gemini import GeminiSingleCallAdapter
adapter = GeminiSingleCallAdapter()
```

```bash
varix run agent.py --input "Why is the sky blue?" -n 3 --max-cost 0.05
```

You get a verdict in the terminal, a JSON artifact under `~/.varix/runs/`, and the exact billed cost.

## Multi-step pipelines

This is where varix earns its keep. Wrap your agent in an `Adapter` so varix can observe each step.

```python
# agent.py
from datetime import UTC, datetime
from varix.core import AdapterCapabilities, PipelineRun, Step, StepGraph, StepRun

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
        # call your model three times, build a StepRun per step
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

```bash
varix run agent.py --input "..." -n 3 --max-cost 0.10
```

The contract is four methods and three capability flags. The flags tell varix what you can honestly expose; classifiers emit `cannot verify` rather than guess when a flag is False.

## Reading the report

Stable pipeline:

```
No nondeterminism found in agent.py.

n=3 | $0.0002 | 14s | analysis 5dfcc218
```

varix found a source:

```
Found 1 source of nondeterminism in agent.py.

  step `plan`  ->  prompt-side, propagates downstream

n=3 | $0.0002 | 14s | analysis 5dfcc218

Next:
  varix impact plan       see how much this changes your output
  varix explain plan      see the evidence varix used
```

Outputs are stable but the provider routed differently (different `system_fingerprint` across runs) — your pipeline is fine *today*, but the substrate shifted:

```
Your pipeline's outputs were stable across 3 runs.

  varix did detect provider routing changes during the runs:
    step `respond`  ->  fingerprint changed (fp_a -> fp_b)
```

## Commands

| Command | What it does | Cost |
|---|---|---|
| `varix run <pipeline>` | Localize + classify against a fresh batch of runs | LLM tokens |
| `varix explain <step>` | Show the evidence behind a finding | free |
| `varix impact <step>` | Quantify how much the variance changes the final output | free |
| `varix show <id>` | Re-render a past analysis | free |
| `varix replay <id>` | Re-run classification over a saved artifact with the current code | free |
| `varix list` | List recent analyses, most-recent first | free |

`<pipeline>` accepts a file path (`agent.py`) or an importable string (`my_module:my_pipeline`).

## Cost control

Always set `--max-cost`:

```bash
varix run agent.py --input "..." -n 5 --max-cost 0.10
```

If a run pushes total spend over the budget, varix halts mid-loop, writes a partial artifact with the runs that completed, and exits with a clear `WARNING:` block. No surprise bills.

Rough numbers: a 3-run, 3-step Gemini Flash Lite pipeline is around $0.0002. A 5-run, 8-step Opus run can hit $1+.

## Sharing and replaying

The JSON at `~/.varix/runs/<id>.json` is the unit of work. Send it to a teammate; they can:

```bash
varix replay <analysis-id>
```

No adapter import, no LLM calls, no cost. varix re-runs classification over the saved runs with the current code. Old artifacts stay readable — varix carries forward every schema version it has shipped.

> Artifacts contain your prompts, tool calls, and model outputs verbatim. Review before sharing externally.

## Confidence levels

- `high` — direct evidence (e.g., differing `system_fingerprint`).
- `medium` — variance remains after ruling out provider, tool, ordering, and time/state — most likely sampling.
- `low` — heuristic match (e.g., a tool named `get_current_time`).
- `cannot verify` — the adapter doesn't expose what's needed; varix refuses rather than guesses.

## What varix doesn't do

- Not a fix-suggester. It diagnoses; you fix.
- Not a continuous monitor. Run it on demand or in CI.
- Not an evaluation framework. It doesn't judge whether outputs are *correct*.
- No web UI. Terminal + JSON.
- One input at a time. Not a regression test runner.
- Not a model-comparison tool.

## Things to know

1. **Low N is directional.** N=3 distinguishes deterministic from nondeterministic, not "30% flaky" from "60% flaky." Bump N if you need that.
2. **Prompt-side is the catch-all.** It fires when nothing else fits — medium confidence by design.
3. **Time/state detection is best-effort.** Clock-named tools and ISO timestamps in output trip it; subtler cases won't.
4. **Tool-side checks need real tool calls.** Mock side-effecting tools during replay.
5. **Some adapters can't see fingerprints** (e.g., Anthropic). In those cases varix says `cannot verify`.

## Design principles

- **Refuse rather than guess.** Every finding has a labeled confidence, including `cannot verify`.
- **Cost is always visible.** No hidden bills.
- **The artifact is the contract.** Old artifacts stay readable forever; varix migrates them forward when the schema bumps.
- **Each command answers one question.** No flag soup, no surprising side effects.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
