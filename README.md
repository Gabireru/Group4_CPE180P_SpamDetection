# SMS Spam Detection: ONNX MLOps Pipeline

## Problem and SLAs
Detect spam SMS messages (binary text classification, UCI SMS Spam Collection).
SLAs: accuracy >= 90% (enforced in `train.py` and CI), inference latency <= 100 ms (checked per request and in tests).

## Architecture
```
Client node (Win11 / Ubuntu WSL)             Server node (Sakura Cloud VM, Ubuntu 24.04, 1-2 CPU, 2-4 GB)
 git, train.py, client.py  --HTTP POST-->    ufw: 22, 8000 only | SSH key auth only
 POST /v1/predict {"message": ...}           Docker (python:3.10-slim) -> FastAPI + ONNX Runtime
 <-- {label, spam_probability, latency_ms, drift_alert}     logs: docker logs / tmux; secrets: /etc/spam-api/env
```

## Pipeline
| Stage | File |
|---|---|
| Preprocess (schema check, impute/drop empty, dedupe, normalize, TF-IDF L2 scaling) | `train.py`, `features.py` |
| Train (scikit-learn), export `.onnx`, `onnx.checker`, ONNX-vs-sklearn parity | `train.py` |
| REST API, logging, latency SLA warning, drift z-test | `app.py` |
| Client / demo | `client.py` (`--drift-demo`) |
| Container | `Dockerfile`, `requirements.txt` |
| CI/CD (train -> test -> build -> push -> SSH deploy) | `.github/workflows/ci.yml`, `deploy/` |

## Quick start
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
python train.py                      # writes models/
pytest -q tests
uvicorn app:app --port 8000          # then: python client.py "Free prize!"
```
Server: run `deploy/setup_server.sh` once, then `deploy/deploy.sh` (or push to `main`).
CI secrets needed: `SSH_PRIVATE_KEY`, `SAKURA_HOST`, `SAKURA_USER`.

## Security
Secrets only via env vars / `/etc/spam-api/env` (see `.env.example`); `.env`, keys and `*.pem` are gitignored;
`chmod 600 ~/.ssh/id_rsa`; container runs as non-root; optional `X-API-Key` auth.

## Known gotchas
- ONNX `StringNormalizer` needs an `en_US.UTF-8` locale (absent in `python:3.10-slim`), so lowercasing is done in `features.normalize()` for both training and serving.
- Port collision: `docker rm -f spam-api` before re-running (done in `remote_run.sh`).
