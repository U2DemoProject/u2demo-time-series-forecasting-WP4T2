# U2Demo data models v2.0.0 schemas

Copies of the U2Demo data models v2.0.0 JSON schemas used by the tests to validate the v2 forecast output
(`ForecastV2`) and to compare the v2 vocabulary of the library with the schemas.

- Source: <https://github.com/U2DemoProject/U2DemoDataModels/tree/6075f75/v2.0.0>
- Commit: `6075f75fc05118e7f2e8b9413b2642de63f900ae`

The files keep their upstream path under `v2.0.0/`, so the `$ref` URI `https://u2demo.eu/schema/v2.0.0/<path>`
resolves to `<path>` in this directory (see `v2_validator` in `tests/conftest.py`).

Only the schemas reached from `api/components/forecast/ForecastPayloadDM.json` are copied, plus
`spatial/TimeParametersDM.json`. The payload schemas reference fragments of the entity schemas
(`ForecastDM.json#/properties/forecastId`), and only those fragments are evaluated: the references inside the
entity schemas (`OptimizationInstanceDM.json`, `AssetDM.json`, ...) are never resolved.

To update the copies, take the same paths from the new upstream commit, for example with a local clone:

```sh
git -C ../U2DemoDataModels show <commit>:v2.0.0/forecast/ForecastSeriesDM.json > forecast/ForecastSeriesDM.json
```

and update the commit above.
