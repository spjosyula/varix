# Artifact schema and compatibility policy

varix persists each analysis as one JSON file at
`~/.varix/runs/<analysis_id>.json`. The schema version is recorded as the
top-level `schema_version` field, which is also exposed in code as
`varix.core.SCHEMA_VERSION`.

The current shipping schema is **0.2**. Schema 0.1 is still readable; see "Reading older artifacts" below.

## Versioning rules

- The version is bumped **only** for breaking changes to the JSON shape
  (renaming a field, removing a field, changing a value type).
- The version is **never** bumped for additive changes (new optional fields,
  new enum values that older code can ignore).
- New varix versions only ever **write** the current schema.
- Old artifacts are always readable in their original schema. The library
  carries forward every read path it ever shipped.
- Reading an artifact whose `schema_version` is newer than the running
  varix is refused with `RefusalRequired`. Refusing is honest; guessing is
  not.

## Reading older artifacts

`_KNOWN_VERSIONS` in `src/varix/surface/storage.py` lists every schema
version varix can read. The tuple grows over time and never shrinks —
this is the "forever-readable" commitment that makes saved artifacts
trustworthy as bug-report units and replay inputs.

When the schema bumps, the `_migrate_to_current` function registers a
stepwise migration that converts old data into the current shape before
`PipelineAnalysis.from_dict` sees it. Migrations are stepwise
(0.1 → 0.2 → 0.3, not 0.1 → 0.3 directly). Each step is a clear comment
explaining what changed and why.

### Migration table

- **0.1 → 0.2.** Added optional `capabilities` field recording the
  adapter's `AdapterCapabilities` at run time. No data transformation
  needed — old artifacts simply lack the key, and `from_dict` treats
  missing as `None`. Consumers that need capabilities (e.g. `varix
  replay`) fall back to `varix.analysis.infer_capabilities`, which
  scans the runs themselves to derive what the adapter must have
  exposed.

## What's in a 0.2 artifact

Top-level fields:

- `analysis_id` — UUID-like identifier, also the filename.
- `pipeline_name` — human-readable label for the analyzed pipeline.
- `n` — how many times the pipeline was run.
- `metric_name` — the `VarianceMetric.name()` used (`"exact"` for ExactMatch).
- `schema_version` — `"0.2"`.
- `runs` — array of `PipelineRun` records.
- `findings` — array of `Finding` records.
- `started_at` / `finished_at` — ISO-8601 timestamps.
- `total_cost` — accumulated `CostSnapshot`.
- `step_replays` — `{step_id: [StepRun, ...]}` mapping for any replays
  collected during analysis.
- `notes` — list of strings; runtime + analysis-derived warnings.
- `capabilities` — recorded `AdapterCapabilities` (new in 0.2; null for
  legacy artifacts loaded via the 0.1 migration).

The exact field shape of each nested record matches what its
`to_dict()` / `from_dict()` pair produces — see
`src/varix/core/types.py` for the canonical definition.

## File layout

- One file per analysis, named `<analysis_id>.json`.
- JSON written with `indent=2` and `sort_keys=True` so diffs and `grep`
  are friendly.
- Writes are atomic: data lands in `<analysis_id>.json.tmp` first, then
  `os.replace` puts it in place. An interrupted save never leaves a
  partial file under the canonical name.
