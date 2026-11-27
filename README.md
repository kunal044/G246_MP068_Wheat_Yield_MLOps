# Wheat Yield per Acre (Nagpur)

This SCSE3040 MLOps mini project predicts wheat yield for the fictional Nagpur Kisan Producer Company. The estimate helps the company plan storage and transport before harvest. The regression target is `yield_qtl_per_acre`, measured in quintals per acre. We compare models by validation mean absolute error (MAE), in the same units.

**Stage 3 serving model** · Linear Regression, validation MAE **0.750357**, MLflow model version **1**.

## Group and team

| Group | Problem | Subject | Batch | Faculty |
|---|---|---|---|---|
| G246 | MP068 | SCSE3040 MLOps | 56 | Dr. Upasna Joshi |

| Enrollment Number | Name |
|---|---|
| S24CSEU1439 | Amaskapoor |
| S24CSEU1479 | Kunal |
| S24CSEU1531 | Sarthak Shukla |
| S24CSEU1451 | Charchit Kanchav |

## Repository structure

| Path | Purpose |
|---|---|
| `src/wheat_yield/` | Dataset analysis, cleaning, and training pipeline helpers |
| `params.yaml`, `requirements.txt` | Project configuration and pinned dependencies |
| `.dvc/`, `data/raw/*.dvc` | DVC configuration and original dataset pointers |
| `data/raw/holdout_features.csv` | Assignment-provided holdout features |
| `reports/findings.yaml` | Data quality findings and tracking references |
| `mlflow.db`, `models/model.joblib` | MLflow run and registry metadata, and the serialized serving pipeline |
| `app/`, `tests/` | FastAPI application, JSON logger, and automated tests |
| `Dockerfile`, `.dockerignore` | Container build and context exclusions |
| `.github/workflows/ci.yml` | Python test job and Docker smoke test |
| `k8s/` | Deployment and Service manifests |

## Environment setup

Run these commands from the repository root in PowerShell. Local verification used Python **3.11.9**. CI and Docker use Python **3.12**. Create the virtual environment only if it does not already exist.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Git, DVC, and data analysis

Git tracks source code, the DVC pointers, the holdout features, and the serving artifact. To rerun the analysis, restore the original `train.csv` and `later.csv` from the group’s assignment bundle into `data/raw/`. The DVC pointers record the file fingerprints and sizes. This checkout has no configured shared DVC remote. The raw training and later CSVs and generated files under `data/processed/` stay local. Serving and automated tests can run without those files.

```powershell
.\.venv\Scripts\python.exe src/wheat_yield/analyze.py
```

The script computes the original training file’s SHA-256 and runs the seven data quality checks. The recorded results are in [reports/findings.yaml](reports/findings.yaml).

| Data issue | Recorded finding |
|---|---|
| Leakage | `yield_from_mandi_receipt` is unavailable at prediction time |
| Sentinel value | `farm_acres = 999` in 69 rows |
| Duplicate records | 42 extra exact copies |
| Target typos | 12 affected rows |
| Unit mix-up | Absent |
| Stuck sensor | Absent |
| Spelling variants | Absent |

### Cleaning and preprocessing

`clean_data(df)` copies the input, drops exact duplicates and the leaking column, replaces the farm-size sentinel with a missing value, and divides target values above 100 by 10. The cleaning settings come from `params.yaml`. The training dataset went from **1,042 to 1,000 rows** after removing duplicates. The model pipeline imputes numeric values with medians, imputes categorical values with the most frequent category, and one-hot encodes categorical inputs with unknown categories ignored. The training helper uses a 20% validation split with random seed 42.

## MLflow experiments and champion

The three completed runs in `MP068_Wheat_Yield` recorded the original training file’s SHA-256 as `data_sha256` and logged validation MAE in `mlflow.db`.

| Model | Validation MAE (quintals per acre) |
|---|---:|
| **Linear Regression** | **0.750357** |
| Gradient Boosting | 0.892677 |
| Random Forest | 1.174861 |

Linear Regression had the lowest validation MAE. Its run ID is `96e9463c36bd467e87c9ae8c0117f150`. The registry model is `MP068_Wheat_Yield_Model`, version **1**, with alias `champion`. FastAPI loads the checked-in `models/model.joblib` at startup, so serving needs no running MLflow server.

To inspect the tracking records locally:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db
```

## Run and use the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

| Method and path | Result |
|---|---|
| `GET /health` | HTTP 200 with `{"status":"ok"}` |
| `GET /docs` | Interactive OpenAPI documentation |
| `POST /predict` | HTTP 200 with a numeric `prediction` in quintals per acre |

All eight fields shown below are required. Pydantic rejects extra fields, sowing delays greater than **65 days**, and negative farm sizes. Invalid requests return HTTP **422** before inference. The boundary values 65 days and 0 acres are accepted.

In a second PowerShell terminal while the server is running:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health'

$sample = @{
    rainfall_mm = 750
    fertilizer_kg = 85
    sowing_delay_days = 15
    irrigation_count = 4
    soil_ph = 6.8
    farm_acres = 5.5
    seed_variety = 'HD-2967'
    tehsil = 'Tehsil North'
}
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/predict' `
    -ContentType 'application/json' -Body ($sample | ConvertTo-Json)
```

The synthetic request returned the following response in both live checks:

```json
{"prediction":22.023605759252384}
```

To reproduce a validation error, set `$sample.sowing_delay_days = 66` and repeat the POST. To check farm size separately, restore the delay to 15 and set `$sample.farm_acres = -1`. Both checks return HTTP 422.

## Structured prediction logging

[app/logger.py](app/logger.py) uses Python’s standard logging module and a custom JSON formatter. Each successful prediction emits one line to stdout with an event name, the version of the deployed model, the returned prediction, and the inference latency in milliseconds measured with `perf_counter()`. The prediction logger omits request fields. Requests rejected by validation do not emit prediction events.

Two live requests returned HTTP 200 with the same prediction. Each emitted one JSON line. The observed latencies below are rounded to two decimal places.

```jsonl
{"event":"prediction","model_version":1,"prediction":22.023605759252384,"latency_ms":124.72}
{"event":"prediction","model_version":1,"prediction":22.023605759252384,"latency_ms":20.18}
```

## Automated tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

The Stage 3 local run reported **13 passed in 6.79 seconds**.

| Suite | Passed | Coverage |
|---|---:|---|
| Cleaning | 5 | Duplicates, leakage, sentinel values, target correction, and preservation of the input DataFrame |
| API | 7 | Health, prediction, required fields, validation, and boundary values |
| Logging | 1 | One stdout JSON event per successful prediction, matching model version and prediction, valid latency, and no prediction event for invalid requests |

## Docker

On a machine with Docker, run:

```powershell
docker build -t mp068-api:1.0 .
docker run --rm --name mp068-api -p 8000:8000 mp068-api:1.0
```

The image uses `python:3.12-slim` and starts Uvicorn on `0.0.0.0:8000`. It includes the saved serving pipeline. `.dockerignore` excludes datasets, local environments, notebooks, and MLflow tracking files from the build context. Use the same API requests on host port 8000 after startup.

## GitHub Actions

The [CI workflow](.github/workflows/ci.yml) runs on push and pull requests. It installs pinned dependencies under Python 3.12 and runs pytest. The Docker job then builds `mp068-api:1.0`, starts it with port 8000 published, and probes `/health` with bounded retries. It checks for container exit and prints container state and logs on failure.

The Python job, Docker build, and container health check passed for commit `266c521`. That run predates the local structured logging integration. [Verified GitHub Actions run 45940654597](https://github.com/kunal044/G246_MP068_Wheat_Yield_MLOps/actions/runs/45940654597)

## Kubernetes

The [Deployment](k8s/deployment.yaml) runs three replicas of `mp068-api:1.0` on container port **8000**. The [ClusterIP Service](k8s/service.yaml) `mp068-svc` selects Pods with `app: mp068-api` and forwards port **80** to **8000**.

Both probes call `/health` on port 8000. Readiness begins after **5 seconds** and controls whether the Pod receives traffic. Liveness begins after **30 seconds** and allows Kubernetes to restart the container after repeated failures. Both manifests passed local YAML and configuration consistency checks.

## Known limitations

- The committed `train.py` contains reusable helpers for preprocessing, data splitting, and hashing. Reproducing the original training run from a fresh clone still requires the group’s original CSVs and a driver that runs cleaning, fitting, evaluation, and MLflow logging.
- The tracked MLflow database preserves run and registry metadata. Local artifact URIs may be machine-specific. The tracked joblib supports serving from a fresh clone.
- Kubernetes verification covered manifest syntax and consistency. Deploying to a cluster also requires the image to be available to its nodes.
- The two local latency observations describe those requests only. Stage 3 verification covers serving, testing, packaging, deployment configuration, and prediction logging.
