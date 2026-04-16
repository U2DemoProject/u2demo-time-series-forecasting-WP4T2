<div align="center"><img src="U2DEMO_logo.png" alt="U2Demo Logo" width="200px" /></div>

# U2 Time Series Prediction

[![CI](https://github.com/U2DemoProject/u2demo-time-series-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/U2DemoProject/u2demo-time-series-prediction/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/U2DemoProject/u2demo-time-series-prediction/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Coverage](https://codecov.io/gh/U2DemoProject/u2demo-time-series-prediction/branch/main/graph/badge.svg)](https://codecov.io/gh/U2DemoProject/u2demo-time-series-prediction)

## Contents

- [Install](#install)
- [Use](#use)
  - [Environment variables](#environment-variables)
  - [Sanity check](#sanity-check)
  - [Train](#train)
  - [Predict](#predict)
- [Build](#build)
- [Test](#test)
- [Troubleshooting](#troubleshooting)

This document has been produced in the context of the U2Demo project. Views and opinions expressed in this document are
however those of the authors only and do not necessarily reflect those of the European Union or the European Climate,
Infrastructure and Environment Executive Agency (CINEA). Neither the European Union nor the granting authority can be
held responsible for them. U2Demo has received funding from the European Union's Horizon Europe programme under grant
agreement no. 101160684.

This is the output of the forecasting algorithms developed in task 4.2. It is the source code for the time series
prediction algorithm used in the U2Demo project. It is used to train models for the prediction of time series and
perform inference on them.

Models trained with this application will contain custom metadata that is relevant to the specific nature of the U2
demo project. This metadata will be read automatically when performing inference using this application. This ensures that
new data will only be accepted by models that were trained on sufficiently similarly-shaped training data.

Models can be trained using hourly or daily timestep frequency.

### Note on containerized usage

In order to perform training and inference using your system's GPU in a docker container, you must install the Nvidia
Container Toolkit by following the [official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Install

### With poetry

Optionally, create and activate a virtual environment using whichever method you prefer.

This repo uses Poetry version **2.2.1**

```sh
pip install poetry
poetry install
```

To run the [example notebooks](examples/), install the optional `examples` group:

```sh
poetry install --with examples
```

### As Docker image

Optionally, replace `latest` with a specific tag

```sh
docker pull u2/time-series-prediction:latest
```

If you are using the service definition as it appears in `docker-compose.yml`, you can set the environment variable
`U2_TIME_SERIES_PREDICTION_IMAGE` to `u2/time-series-prediction:latest` (or whatever other tag) and run:

```sh
docker compose pull
```

## Use

#### Environment variables

There are several environment variables which need to be set, although many will assume default values if left unset. To
set the .env vars for all run types, copy the file `.env_template` to `.env` and adjust values as necessary.

- **`U2_TIME_SERIES_PREDICTION_IMAGE`** default: `u2/time-series-prediction:latest`
    - Image to be used when running with `docker compose`

- **`LOG_LEVEL`** default: `INFO`
    - Logging level

- **`NOMINATIM_DOMAIN`** default: `nominatim.openstreetmap.org`
    - Domain of the Nominatim geocoder, used for geolocating named places for getting weather and holiday features

- **`NOMINATIM_USER_AGENT_NAME`** default: `u2-demo-time-series-prediction`
   - User agent name to be used when accessing Nominatim. It is against their TOS to do it anonymously.

- **`PUBLIC_HOLIDAY_API_URL`** default: `https://date.nager.at/api/v3`
    - URL for getting public holiday information

- **`WEATHER_HISTORICAL_API_URL`** default: `https://archive-api.open-meteo.com/v1/archive`
    - URL for getting historical weather data

- **`WEATHER_FORECAST_API_URL`** default: `https://api.open-meteo.com/v1/forecast`
    - URL for getting weather forceast data

- **`CACHE_DIR`** default: `.cache`
    - Folder which will store cached calls to all external APIs (Nominatim, Open-Meteo, Nager.Date)

> **Note:** All environment variables are optional. The application falls back to the built-in defaults shown above when a variable is unset.

### Sanity check

Check availability of CUDA and compatible devices with:

#### Python:

```sh
poetry run u2 sanity_check
```

#### Docker:

```sh
docker run --rm --gpus all u2/time-series-prediction:latest sanity_check
```

#### Docker Compose:

```sh
docker compose run --rm u2demo-time-series-prediction sanity_check
```

### Input format

Both `train` and `predict` take a single JSON file as input. The JSON must validate against
the `ForecastTrainingInput` or `ForecastInput` Pydantic models.

Minimal training payload:

```json
{
  "forecast_type": "stochastic",
  "time_parameters": { "timestep_minutes": 60, "horizon_hours": 24 },
  "model_config": {
    "model_id": "dummy_model",
    "model_version": "1.0.0",
    "model_preset": "fast_training",
    "lookback_steps": 500
  },
  "target_series": {
    "series_id": "meter_pavilion_1",
    "variable_name": "energyImportTotal",
    "unit": "kWh",
    "training_values": [
      { "time": "2024-09-18T13:00:00", "value": 25.57 },
      { "time": "2024-09-18T14:00:00", "value": 31.16 }
    ]
  },
  "location": { "latitude": 52.07, "longitude": 4.30 },
  "features": ["temporal", "lag", "holiday", "weather"],
  "num_weekly_lags": 2
}
```

The `features` list controls which feature engineering steps run. To train fully offline
(no calls to Nager.Date, Open-Meteo, or Nominatim), use `["temporal", "lag"]` and omit
`location`.

Trained models are written to `MODEL_DIR/<model_id>_<model_version>/` (default
`./models/dummy_model_1.0.0/`). Forecast outputs are written under `OUTPUT_DIR`
(default `./output/`) as `<model_id>_<model_version>_<forecast_date>.json`.

### Train

The `train` command reads a `ForecastTrainingInput` JSON file, validates it, engineers
temporal/lag/holiday/weather features, and fits an AutoGluon `TimeSeriesPredictor`.

#### Python

```sh
poetry run u2 train data/dummy_train.json
```

#### Docker

```sh
docker run --rm \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/output:/app/output" \
  u2/time-series-prediction:latest \
  train data/dummy_train.json
```

#### Docker Compose

```sh
docker compose run --rm u2demo-time-series-prediction train data/dummy_train.json
```

### Predict

The `predict` command reads a `ForecastInput` JSON file, loads the matching trained model
from `MODEL_DIR/<model_id>_<model_version>/`, replays the exact same feature engineering
pipeline used at training time, and writes a `Forecast` JSON under `OUTPUT_DIR`.

(All examples below assume `dummy_model_1.0.0` already exists in `./models`. Run the `train`
command above once to generate it.)

#### Python

```sh
poetry run u2 predict data/dummy_predict.json
```

#### Docker

```sh
docker run --rm \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/output:/app/output" \
  u2/time-series-prediction:latest \
  predict data/dummy_predict.json
```

#### Docker Compose

```sh
docker compose run --rm u2demo-time-series-prediction predict data/dummy_predict.json
```

## Build

To build a Docker image for this project, use the provided `build.sh` script. The script uses Poetry to export
dependencies to a new `requirements.txt` to account for any upgrades that might have ocurred between dependencies

(`--latest`) will tag the image as `latest` in addition to building with the specified `tag`

```sh
./build.sh dev --latest
```

## Test

The project uses `pytest` for testing. Tests are located in the `tests/` directory and can be run using Poetry.

To install `pytest`, use the command:

```sh
poetry install --with dev
```

### Test markers

Tests are organized into three categories using pytest markers:

| Marker | What it covers | Requirements |
|--------|---------------|--------------|
| `unit` | Fast, isolated tests with mocked externals | None |
| `integration` | Real AutoGluon models, real API calls | Network access, external APIs available |
| `e2e` | Full train-then-predict pipeline with Chronos2 | Chronos2 model download (~500MB first run) |

### Running tests

By default, `poetry run pytest` runs only **unit tests** (e2e and integration are excluded via `pytest.ini`). This is also what CI runs.

```sh
# Unit tests only (default, CI-safe)
poetry run pytest

# End-to-end tests (trains real Chronos2 models)
poetry run pytest -m e2e

# Integration tests (real AutoGluon models + external API calls)
poetry run pytest -m integration

# Everything
poetry run pytest -m "unit or integration or e2e"
```

Run a specific test file with:

```sh
poetry run pytest tests/test_dataset.py
```

To get a coverage summary (e.g. to paste into a MR description):

```sh
poetry run coverage report --format=markdown
```

## Troubleshooting

### `PermissionError` on model files after training in Docker

Pass `--user "$(id -u):$(id -g)"` to `docker run` so the container writes bind-mounted
files (under `./models`, `./output`, `./.cache`) as your host user instead of `root`.
The Docker examples above already include this flag.

If you inherit a model directory that was created without that flag, fix ownership once with:

```sh
sudo chown -R "$(id -u):$(id -g)" models/<model_id>_<model_version>
```

### CUDA / GPU not detected inside Docker

If `u2 sanity_check` reports no CUDA devices, install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
and confirm the container is launched with `--gpus all`.

### `dummy_predict.json` weather fetch fails with 400

`dummy_predict.json` contains hardcoded timestamps. The Open-Meteo forecast API only serves
data within ~3 months of today, so if the file has gone stale you will see a `400 Bad Request`
when running `poetry run u2 predict data/dummy_predict.json`.

Refresh the timestamps with:

```sh
poetry run python scripts/refresh_dummy_predict.py
```

### External API rate limits

Holiday and weather features call third-party APIs (Nager.Date, Open-Meteo, Nominatim).
Calls are cached on disk under `CACHE_DIR` (default `./.cache/`) and retried with
exponential backoff. To run completely offline, set `"features": ["temporal", "lag"]`
in your training input JSON and omit the `location` field.
