# Contributing to varix

Varix is small enough that contributing is straightforward — here's how.

## Found a bug or have an idea?

Open an issue. Describe what you saw and what you expected. If you have a small repro (an agent file, a varix command), include it.

## Want to fix something or add a feature?

Open a pull request. The standard open-source flow:

1. **Fork the repo** on GitHub (button in the top right of the repo page).

2. **Clone your fork** locally:

   ```
   git clone https://github.com/<your-username>/varix.git
   cd varix
   ```

3. **Set up a local environment** so you can run the tests:

   ```
   python -m venv .venv
   .venv/Scripts/activate          # Windows
   source .venv/bin/activate       # macOS/Linux
   pip install -e ".[dev]"
   pytest
   ```

   If `pytest` passes, you're set up.

4. **Create a new branch** for your change:

   ```
   git checkout -b your-branch-name
   ```

5. **Make your changes.** Add tests if you're adding behaviour. Run `pytest` to make sure nothing broke.

6. **Commit and push** to your fork:

   ```
   git add .
   git commit -m "short summary of your change"
   git push origin your-branch-name
   ```

7. **Open a pull request** on GitHub from your branch into the main repo. We'll review and either merge it or tell you what we'd like adjusted.

## What you can contribute

- Bug fixes
- New features (open an issue first if it's substantial — saves you from wasted work if we'd say no)
- Documentation improvements
- New adapters for model providers we don't yet support
- Performance improvements
- Tests for things that aren't covered

## A note on the artifact schema

varix saves every analysis as a JSON file in `~/.varix/runs/`, and a lot of varix's value (especially `varix replay`) rests on those files staying readable. So the schema in `src/varix/core/types.py` is load-bearing in a way most internal code isn't.

The shape of it:

- **Adding an optional field is fine.** Old artifacts work; new varix writes the new field; old varix ignores it.
- **Adding a field that replay correctness depends on** needs a heuristic fallback in `varix.analysis.infer_capabilities` (or wherever the consumer lives), so legacy artifacts can still replay correctly. If no fallback is possible, replay should refuse cleanly on older artifacts rather than silently produce different findings.
- **Changing or removing a field** bumps `SCHEMA_VERSION`, registers a stepwise migration in `src/varix/surface/storage.py:_migrate_to_current`, and adds the old version to `_KNOWN_VERSIONS`. Old artifacts never stop being readable.

If your PR touches `PipelineAnalysis`, `Finding`, `Evidence`, or any other persisted type, mention which category your change falls in. See [docs/schema.md](docs/schema.md) for the current shape and the migration history.

## If you used AI

Just say so in the PR description. One line is enough — which tool, and what you used it for:

- *"Claude Sonnet 4.6 — drafted the classifier; I wrote and reviewed the tests."*
- *"Cursor for autocomplete; logic is mine."*
- *"No AI."*

It just helps reviewers know what to focus on. You're still responsible for what you submit, make sure you understand it and the tests pass.

## License

By contributing, you agree your contributions are licensed under the MIT License (see [LICENSE](LICENSE)).
