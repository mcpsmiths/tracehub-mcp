# Contributing to tracehub-mcp

Thanks for considering a contribution. This project is small enough that most of what you need is
already in [README.md](README.md#development) — architecture, backend abstraction, adding a new
tool or backend, and the full dev command reference (`uv sync`, `uv run pytest`,
`uv run ruff check`, `uv run mypy .`).

## Before opening a PR

1. `uv sync --dev` to install dependencies (this project uses [uv](https://docs.astral.sh/uv/), not
   pip, as the package manager).
2. Make your change, with tests. Bug fixes should include a regression test.
3. Run the full local gate the CI `lint` and `test` jobs run:
   ```bash
   uv run ruff check
   uv run mypy .
   uv run pip-audit
   uv run pytest
   ```
4. If `.pre-commit-config.yaml` is installed (`pre-commit install`), most of the above runs
   automatically on commit.

## Commit messages

This repo follows [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`,
`docs:`, `chore:`, `ci:`, etc.). Releases are cut automatically from commit history via
[Commitizen](https://commitizen-tools.github.io/commitizen/) — `feat:` bumps the minor version,
`fix:` bumps the patch version (this project is still `0.x`, so a `feat:` bumps `0.x.0`, not `1.0.0`,
per [SemVer's `major_version_zero` rule](https://semver.org/#spec-item-4)). Commit type matters for
more than style here.

## Reporting bugs / requesting features

Use the [issue templates](.github/ISSUE_TEMPLATE/) — they ask for the specific backend and version
info that's usually needed to reproduce a trace-parsing bug.

## Security issues

Do not open a public issue — see [SECURITY.md](SECURITY.md).
