# U2 Time Series Prediction Final

## Install

### As Docker image

TODO

### Standalone Python

TODO

### For development

```sh
pip install poetry
poetry install
```

## Use

### Sanity check

Check availability of CUDA and compatible devices with:

```sh
python cli.py sanity_check
```

### Prediction

Perform a time series inference with:

```sh
python cli.py predict --time-series-csv "path/to/input.csv" --model-path "path/to/model"
```

## Build

TODO