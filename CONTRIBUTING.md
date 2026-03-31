# Contributing

This repo is primarily maintained as a portfolio project, but focused pull
requests and issue reports are welcome.

## Scope

Good contributions for this repo:

- bug fixes with a clear reproduction path
- documentation corrections
- local developer-experience improvements
- benchmark methodology improvements
- small, well-scoped architecture refinements

Please avoid opening large speculative refactors without first describing the
motivation and expected impact.

## Local Setup

```bash
cp .env.example .env
uv sync
make up
make train-quick
make seed-data
make feast-apply
make feast-materialize
make serve-perf
make smoke-test
```

## Pull Request Guidelines

- keep changes scoped and explain the problem being solved
- update docs when behavior or commands change
- do not commit generated artifacts from `reports/`, Ray temp state, or local
  Feast registry files
- prefer reproducible commands in the PR description
- if the change affects performance claims, include before/after numbers and
  describe the exact benchmark path used

## Validation

At minimum, run the checks relevant to your change. Common examples:

```bash
uv run python -m py_compile $(find services scripts -name '*.py')
make smoke-test
```

For performance-related changes, prefer:

```bash
make serve-perf
make load-test-local
make perf-breakdown
```

## Reporting Issues

When filing an issue, include:

- operating system
- Python version
- the command you ran
- the exact error output
- whether you were using `make serve` or `make serve-perf`
- if performance-related, the relevant `reports/load_test_summary_*.md`
