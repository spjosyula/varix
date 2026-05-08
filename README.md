# varix

**A nondeterminism classifier for agent pipelines.** Run your agent N times. Find which step is the source of variance, whether it propagates to the final output, and what category of nondeterminism it is.

No fixes. No suggestions. Just diagnosis you can trust.

## The problem

You run your agent. It works. You run it again with the same input, same model, same `temperature=0`. Different output. Different tool calls. Sometimes a different final answer.

There's no tool today that tells you *which step* is the source of the nondeterminism vs. which steps are merely propagating it downstream. Existing observability tools show you side-by-side traces and leave the diagnosis to you.

varix is that diagnostic tool.

## What v1 does

Three jobs, in order. Each one is honest about what it can prove and what it can't.

**1. Localize.** Run the pipeline N times, then for each step replay it N times with its inputs held fixed. A step is a *source* of nondeterminism if its output varies under fixed inputs. A step that varies only because its inputs varied is not a source — it's downstream noise.

**2. Quantify downstream impact (opt-in).** For each source step, take its varied outputs and feed each one through the rest of the pipeline (other steps' randomness held controlled). Measure whether the final output changes. This separates "step 4 is internally flaky but the pipeline absorbs it" from "step 4 is flaky and it changes your final answer 7 times out of 10."

**3. Classify the cause.** For each source step, label which category of nondeterminism is at play:

- **Provider-side** — different `system_fingerprint` across replays. High confidence.
- **Tool-side** — same tool, same args, different results. High confidence.
- **Ordering** — same set of tool calls, different sequence. High confidence.
- **Prompt-side** — variance remains after ruling out the above. Residual; medium confidence.
- **Time/state** — heuristic detection of timestamps, RNG, etc. Lowest confidence.

## What v1 is *not*

- **Not a fix-suggester.** Diagnosis only. We trust the engineer to take the fix from there.
- **Not a continuous monitor.** It's a CLI tool you run during debugging or in CI.
- **Not an eval framework.** It doesn't grade output quality or judge correctness.
- **Not a trace viewer.** No web UI. Output is a terminal report and a JSON artifact.
- **Not a multi-input regression tester.** v1 analyzes one input at a time.
- **Not a model comparison tool.** Diffing pipeline-with-A vs. pipeline-with-B is v2.

## Requirements

- **Step-level replayability.** varix needs to re-execute individual steps with their inputs held fixed. Pipelines that mutate global state inside steps, or that hide step boundaries from the framework, can't be fully analyzed.
- **Fixed pipeline structure.** v1 requires the same set of steps across all N runs. Branching/looping pipelines where the structure varies are not supported; varix detects this and refuses.
- **An adapter for your framework.** v1 ships with raw OpenAI and Anthropic SDK adapters and a LangGraph adapter.

## Command surface

```
varix run <pipeline> --input "..." [--n=5]    # localize + classify
varix impact <step-id>                         # quantify downstream effect
varix explain <step-id>                        # show evidence for a finding
varix show <run-id>                            # reopen a previous report
```

`<pipeline>` accepts either a file path (`varix run agent.py`) or an importable string (`varix run my_module:my_pipeline`).

## Caveats

These are terms you need to understand for the tool's output to mean what it says.

1. **varix costs API tokens to run.** A typical analysis is $0.50–2.00. Surfaced in every report; cap with `--max-cost`.
2. **Statistical confidence at low N is directional, not precise.** N=5 distinguishes "deterministic" from "nondeterministic." It does not reliably distinguish "30% flaky" from "60% flaky."
3. **The "prompt-side" classification is a residual.** It means "we ruled out the others." Labeled MEDIUM confidence.
4. **Time/state detection is heuristic.** We catch the obvious cases. Findings are labeled best-effort.
5. **Tool-side detection requires running tools live during replay.** Use `--mock-tools` if your tools have side effects; this trades tool-side detection for safety.
6. **Frameworks that strip provider metadata can't be analyzed for provider-side variance.** varix reports "provider-side detection unavailable" rather than guessing.

## Design principles

- **Refuse rather than guess.** Every analysis path has an "I cannot do this confidently" mode.
- **Cost is visible, never hidden.**
- **Confidence is labeled, never inflated.** HIGH / MEDIUM / LOW / UNAVAILABLE.
- **Engineer controls the knobs.** N, cost cap, mock vs. live tools, opt-in downstream sweep.
- **One verb does one thing.** No flag soup.

## Status

v1 is under active development. The full commit roadmap is tracked in the project plan; this README will gain a "Quickstart" section once the CLI lands.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). We use AI tools and disclose; commit messages are single-line; refuse rather than guess.

## License

MIT — see [LICENSE](LICENSE).
