# Contributing to varix

Thanks for considering a contribution. varix is early — these are starting norms, not strict rules. We'll tighten them as the project matures.

## Local setup

```
git clone <repo>
cd varix
python -m venv .venv
.venv/Scripts/activate          # Windows
source .venv/bin/activate       # macOS/Linux
pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```

If those four commands succeed, you're set up.

## Before you start

- For non-trivial changes, open an issue first so we can align on the approach.
- For typo fixes, small docs tweaks, or obvious bugs, just open a PR.
- For new adapters, run your adapter against `varix.core.protocol_test_suite` before opening the PR — it's the contract the rest of varix relies on.

## AI use is normalized — disclose it

We use AI tools on this project. Claude, ChatGPT, Cursor, Copilot, all of them. There is no stigma. **The norm is disclosure, not abstinence.**

In your PR description, say which AI you used and roughly for what. One sentence:

- *"Claude Sonnet 4.6 — drafted the classifier logic; I wrote the tests."*
- *"Copilot for autocomplete only."*
- *"No AI."*

It's a practice, not a gate. It helps reviewers calibrate. You're still responsible for what you submit — read it, run the tests, make sure you understand it.

## Commits

- **Single-line commit message.** No verbose bodies, no `Co-Authored-By` trailer.
- Conventional prefix: `chore:`, `feat:`, `fix:`, `docs:`, `test:`, `refactor:`. Scope in parentheses if useful.
- One logical change per commit when reasonable.

Examples:

```
feat(core): adapter protocol
fix(exec): cost ledger off-by-one
docs: clarify schema versioning
```

## Code style

- Python 3.11+; modern type hints (`list[int]`, `X | None`).
- `ruff check` and `ruff format` should pass.
- `mypy --strict` for `src/varix/`.
- Async-first. `Adapter` methods are `async def`.

## Comments

Default to none. Write one when the *why* is non-obvious — a constraint, a workaround, a surprising invariant. Public Protocols get docstrings; internal helpers usually don't need them.

## Tests

- New code should land with tests.
- Tests should run without API keys — use `varix.adapters.fake.FakeAdapter` for anything that'd hit a model provider.
- Property-based tests (`hypothesis`) for domain types where round-trip matters.
- Integration tests live in `tests/integration/`.

## Refuse rather than guess

If you're unsure a finding is reliable, return `Confidence.UNAVAILABLE` with a clear reason rather than guess HIGH. The product's value is trust.

## License

By contributing, you agree your contributions are licensed under the MIT License (see [LICENSE](LICENSE)).
