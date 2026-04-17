# Contributing

Thank you for your interest in contributing to the U2Demo Time Series Forecasting library!

## Setup

```bash
# Install production dependencies
poetry install

# Install dev dependencies (ruff, pytest, pre-commit, pyright)
poetry install --with dev

# Install pre-commit hooks
pre-commit install

# Verify everything works
poetry run pytest
```

## Repository Hosting

Development happens on an internal GitLab instance and is mirrored to this GitHub
repository for public releases. As a result, the repo carries two CI configurations
that are kept in sync:

- `.github/workflows/ci.yml` — runs on pull requests and pushes to `main` on GitHub.
- `.gitlab-ci.yml` — runs on the internal GitLab instance.

External contributors only interact with the GitHub workflow; the GitLab file is kept
in the repo for transparency.

## Git Workflow

We maintain a **linear history** — no merge commits.

1. Create a feature branch from `main`
2. Make your changes — commit often, these will be squashed
3. Before opening a PR, rebase onto `main`:
   ```bash
   git fetch origin
   git rebase origin/main
   ```
4. Open a pull request against `main`
5. Merge using **squash merge** — the PR title becomes the commit message

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/). This is enforced by `gitlint` in CI.

Format: `<type>(<scope>): <description>`

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

Scope is optional. Description should be lowercase and not end with a period.

Examples:

```
feat(dataset): add rolling standard deviation feature
fix(predict): handle single-item time series
docs: update installation instructions
test: add coverage for holiday feature edge cases
refactor(model): extract metadata persistence logic
```

## Code Standards

- **Python**: >=3.12, <3.13
- **Linting and formatting**: [Ruff](https://docs.astral.sh/ruff/) (runs via pre-commit)
- **Line length**: 120 characters
- **Docstrings**: NumPy style (enforced by `ruff` `pydocstyle` convention)
- **Type annotations**: Required on all public functions (enforced by `ANN` rules and `pyright`)
- **Tests**: Located in `tests/`, using pytest

Run all checks locally:

```bash
pre-commit run --all-files
poetry run pytest
```

## Testing

Tests are organized by module and marked with pytest markers:

```bash
poetry run pytest                        # all tests
poetry run pytest tests/test_dataset.py  # single file
poetry run pytest -k "test_name"         # single test
poetry run pytest -m unit                # by marker (unit, integration, e2e)
```

When contributing a bug fix, please include a regression test. When adding a feature, include tests covering the main behavior and edge cases.

## Project Structure

```
u2demo_time_series_forecasting/
├── __init__.py      # Public API (train, predict, data models, FeatureName, ModelMetadata)
├── _version.py      # Single __version__ source
├── cli.py           # `u2demo-forecast` command-line entry point (see [project.scripts] in pyproject.toml)
├── data_models.py   # Pydantic v2 external I/O models
├── dataset.py       # DataFrame loading, validation, feature engineering
├── metadata.py      # ModelMetadata / TrainingMetadata (persisted alongside the model)
├── model.py         # AutoGluon TimeSeriesPredictor wrapper
├── train.py         # Training pipeline
├── predict.py       # Inference pipeline
├── diagnostic.py    # CUDA / GPU sanity check
├── utils.py         # Geocoding + shared helpers
├── constants/       # Column names, paths, training defaults, weather/holiday defaults, Settings
├── core/            # Cross-cutting concerns (logger)
└── features/        # Feature modules (temporal, lag, holidays, weather) + FeatureName + apply_features
tests/
```

## External API Dependencies

Holiday and weather features call external APIs (Nager.Date, Open-Meteo, Nominatim). These calls use tenacity retry logic and joblib disk caching. For local development, see `.env_template` for configurable API URLs, which are loaded via `u2demo_time_series_forecasting.constants.envs.Settings` (Pydantic Settings).
