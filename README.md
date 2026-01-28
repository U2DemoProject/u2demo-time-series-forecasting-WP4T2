# U2 Time Series Prediction Final

This is the source code for the time series prediction algorithm used in the U2 Demo project. It is used to train models
for the prediction of time series and perform inference on them.

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

- **`CACHE_DIR=".cache"`**
    - Folder which will store cached calls to all external APIs


### Sanity check

Check availability of CUDA and compatible devices with:

#### Python:

```sh
python cli.py sanity_check
```

#### Docker:

```sh
docker run --rm --gpus all u2/time-series-prediction:latest sanity_check
```

#### Docker Compose:

```sh
docker compose run --rm u2-time-series-prediction sanity_check
```

### Train

The `train` command trains a new U2 time-series prediction model from an input CSV file. It loads and validates the input time-series CSV, engineers temporal, lag, holiday, and weather features, and constructs and fits an AutoGluon `TimeSeriesPredictor`.

#### Required Parameters

- **`time_series_csv`** (str)
  - Path to the input time-series CSV file.

- **`model_path`** (str)
  - Directory where the trained model and metadata will be stored.

- **`horizon`** (int)
  - Forecast horizon (number of future steps to predict).

- **`num_weekly_lags`** (int)
  - Number of weekly multiples used to derive lag and rolling windows.

#### Optional Parameters

- **`features`** (list[str], default: `["temporal", "lag", "holiday", "weather"]`)
  - Features to include in training. Can be chosen from `["temporal", "lag", "holiday", "weather"]`. Default uses all features.

- **`num_test_windows`** (int, default: `1`)
  - Number of backtesting windows used for evaluation when training.

- **`min_lookback`** (int, default: `1000`)
  - Minimum lookback used to form the training backtest split.

- **`location_col_name`** (str, default: `"location"`)
  - Name of the location column in the input CSV.

- **`timestamp_col_name`** (str, default: `"timestamp"`)
  - Name of the timestamp column in the input CSV.

- **`predict_value_col_name`** (str, default: `"value"`)
  - Name of the target value column in the input CSV.

- **`group_col_name`** (str, default: `"item_id"`)
  - Column identifying independent time-series groups.

- **`save_leaderboards`** (bool, default: `False`)
  - If `True`, writes averaged leaderboard metrics to the model directory.

- **`from_csv_kwargs`** (dict, optional)
  - Additional keyword arguments passed to `U2TimeSeriesPredictionDataset.from_csv()`.

- **`training_kwargs`** (dict, optional)
  - Additional keyword arguments forwarded to `TimeSeriesPredictor.fit()`. Useful for passing AutoGluon-specific parameters like `presets` (e.g., `{'presets': 'fast_training'}`).

**Note 1:** All the below examples are using specially selected parameters to make training fast and easy with sample
data

**Note 2:** Training a model via a docker container can produce model directories and files that are owned by `root`,
which can cause access problems later when performing inference, usually in the form of a `PermissionError`. After
training with a docker container, it is advised to change the ownership of the resultant folder with the command:

```sh
sudo chown -R <your user name> models/<created model folder>
```

#### Python:

```sh
python cli.py \
  train \
  "data/dummy_train.csv" \
  "models/dummy_model" \
  24 \
  2 \
  --predict-value-col-name="kwh_meter_value" \
  --save-leaderboards=True \
  --training_kwargs="{'presets': 'fast_training'}" \
  --min-lookback=500
```

#### Docker:

```sh
docker run --rm \
  --gpus all \
  -e LOG_LEVEL=INFO \
  -e NOMINATIM_DOMAIN=nominatim.openstreetmap.org \
  -e NOMINATIM_USER_AGENT_NAME=u2-demo-time-series-prediction \
  -e PUBLIC_HOLIDAY_API_URL=https://date.nager.at/api/v3 \
  -e WEATHER_HISTORICAL_API_URL=https://archive-api.open-meteo.com/v1/archive \
  -e WEATHER_FORECAST_API_URL=https://api.open-meteo.com/v1/forecast \
  -e CACHE_DIR=.cache \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/output:/app/output" \
  u2/time-series-prediction:latest \
  train \
  "data/dummy_train.csv" \
  "models/dummy_model" \
  24 \
  2 \
  --predict-value-col-name="kwh_meter_value" \
  --save-leaderboards=True \
  --training_kwargs="{'presets': 'fast_training'}" \
  --min-lookback=500
```

#### Docker Compose:

```sh
docker compose run --rm u2-time-series-prediction \
  train \
  "data/dummy_train.csv" \
  "models/dummy_model" \
  24 \
  2 \
  --predict-value-col-name="kwh_meter_value" \
  --save-leaderboards=True \
  --training_kwargs="{'presets': 'fast_training'}" \
  --min-lookback=500
```

### Predict

The `predict` command runs a full time-series forecasting pipeline on a CSV input file and writes predictions to disk. It loads a previously trained U2 time-series prediction model, loads and validates the input time-series CSV, ensures the model and data use the same timestamp frequency, extends the dataset by the forecast horizon, applies all required feature-engineering steps (temporal, lag, holiday, weather), splits past and future data for prediction, runs the model's `predict()` method, and saves the resulting forecast to a CSV file.

#### Required Parameters

- **`time_series_csv`** (str)
  - Path to the input time-series CSV file.

- **`model_path`** (str)
  - Path to the saved U2 model directory.

#### Optional Parameters

- **`output_prediction_csv`** (str, optional)
  - Path where the prediction CSV should be saved. If `None`, a timestamped filename under `./output` is generated (format: `output/prediction_YYYYMMDD-HHMMSS.csv`).

- **`location_col_name`** (str, default: `"location"`)
  - Name of the location column in the input CSV.

- **`timestamp_col_name`** (str, default: `"timestamp"`)
  - Name of the timestamp column in the input CSV.

- **`predict_value_col_name`** (str, default: `"value"`)
  - Name of the target value column in the input CSV.

- **`group_col_name`** (str, default: `"item_id"`)
  - Column identifying independent time-series groups.

- **`from_csv_kwargs`** (dict, optional)
  - Additional keyword arguments passed to `U2TimeSeriesPredictionDataset.from_csv()`.

#### Python

(All examples can only be run if `dummy_model` already exists in `./models`. Run one of the above `train` command examples to generate it.)

```sh
python cli.py \
  predict \
  data/dummy_predict.csv \
  "models/dummy_model"
```

#### Docker

```sh
docker run --rm \
  --gpus all \
  -e LOG_LEVEL=INFO \
  -e NOMINATIM_DOMAIN=nominatim.openstreetmap.org \
  -e NOMINATIM_USER_AGENT_NAME=u2-demo-time-series-prediction \
  -e PUBLIC_HOLIDAY_API_URL=https://date.nager.at/api/v3 \
  -e WEATHER_HISTORICAL_API_URL=https://archive-api.open-meteo.com/v1/archive \
  -e WEATHER_FORECAST_API_URL=https://api.open-meteo.com/v1/forecast \
  -e CACHE_DIR=.cache \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/output:/app/output" \
  u2/time-series-prediction:latest \
  predict \
  data/dummy_predict.csv \
  "models/dummy_model"
```

#### Docker Compose

```sh
docker compose run --rm u2-time-series-prediction \
  predict \
  "data/dummy_predict.csv" \
  "models/dummy_model"
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

Run all tests with:

```sh
poetry run pytest
```

Run a specific test file with:

```sh
poetry run pytest tests/test_dataset.py
```
