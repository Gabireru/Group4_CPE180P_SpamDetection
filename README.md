# SMS Spam Detection: ONNX MLOps Pipeline

## Problem and SLAs
Detect spam SMS messages (binary text classification, UCI SMS Spam Collection).
SLAs: accuracy >= 90% (enforced in `train.py` and CI), inference latency <= 100 ms (checked per request and in tests).
Inference Latency SLA <= 100% (checked per request in `app.py` and client verification)
ONXX Parity >= 99% prediction agreement between `scikit-learn` and `ONNX Runtime`.

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
| Virtual Machine Hardening & Deployment | `setup_server.sh, remote_run.sh, deploy.sh, ci.yml` |

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

## Running each script
All commands run from the project root, with the virtual environment active (`source venv/bin/activate`).

### `train.py`: preprocess, train, export ONNX
```bash
python train.py                                   # defaults: data/SMSSpamCollection.txt -> models/
python train.py --data data/SMSSpamCollection.txt --out models
```
Writes `models/spam_model.onnx`, `models/reference_stats.json` (drift baseline) and `models/metrics.json`.
It validates the model with `onnx.checker`, checks ONNX vs scikit-learn agreement, and exits non-zero if accuracy is below 90%.
Re-run it whenever the data or model code changes.

### `app.py`: the REST API
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Needs `models/` to exist (run `train.py` first). Optional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `API_KEY` | unset (auth off) | If set, requests must send the `X-API-Key` header |
| `MODEL_DIR` | `models` | Where the `.onnx` and stats files live |
| `LATENCY_SLA_MS` | `100` | Logs a warning when a request is slower |
| `DRIFT_WINDOW` | `50` | Number of requests per drift check |
| `DRIFT_Z` | `4.0` | z-score threshold that raises a drift alert |

| Endpoint | Description |
|---|---|
| `POST /v1/predict` | Body `{"message": "..."}`; returns label, spam probability, latency (ms), drift flag |
| `GET /v1/drift` | Latest drift state and per-feature z-scores |
| `GET /health` | Liveness check |

Quick check with curl:
```bash
curl -X POST http://localhost:8000/v1/predict \
     -H "Content-Type: application/json" \
     -H "X-API-Key: $API_KEY" \
     -d '{"message": "Free entry! Text WIN to 80085"}'
```
Interactive API docs are at `http://localhost:8000/docs`.

### `client.py`: send requests to the API
Run it in a second terminal while the API is up.
```bash
export SERVER_URL="http://localhost:8000"    # or http://<vm-ip>:8000
export API_KEY="<same key as the server>"    # read from /etc/spam-api/env
python client.py                             # runs built-in ham and spam test samples
python client.py "Free entry! Text WIN to 80085" "see you at lunch"
python client.py --drift-demo                # sends 60 out-of-distribution requests to trigger drift alert
```
Each line prints the label, spam probability, server latency, round-trip time and drift flag.
On Windows PowerShell, set variables with `$env:SERVER_URL="http://<ip>:8000"`.

### `tests/`: automated tests
```bash
pytest -q tests
```
Covers health, spam vs ham, the 100 ms latency SLA, input validation and drift alerting.

### `features.py`
Shared helpers (`normalize`, `msg_features`) imported by `train.py` and `app.py`. It is not run directly.

### Docker
```bash
python train.py                              # generates models/ before building the image
sudo docker build -t spam-api:latest .
sudo docker rm -f spam-api 2>/dev/null || true
sudo docker run -d --name spam-api --restart unless-stopped \
  -p 8000:8000 --env-file /etc/spam-api/env spam-api:latest
sudo docker logs -f spam-api                 
```

### Local Virtual Machine Deployment and WSL Portforwarding
# 1. Copy setup script to VM and execute as sudo (installs Docker, ufw ports 22 & 8000, fail2ban, SSH hardening, generates API key)
scp setup_server.sh vboxuser@<vm-ip>:~/
ssh -t vboxuser@<vm-ip> "chmod +x ~/setup_server.sh && sudo ~/setup_server.sh"

# 2. Retrieve the generated API key for client authentication
sudo cat /etc/spam-api/env

# 3. Optional WSL-to-VM SSH Port Forwarding (forwards local port 8000 to VM port 8000 over SSH)
ssh -L 8000:localhost:8000 vboxuser@<vm-ip> -N -f

### `deploy/`: Sakura Cloud VM (OPTIONAL)
```bash
# 1. One-time, on the VM (as sudo): installs Docker, ufw (22 + 8000), fail2ban, SSH hardening, generates the API key
sudo bash deploy/setup_server.sh

# 2. Manual deploy from your workstation (needs `docker login ghcr.io` first)
SAKURA_HOST=<vm-ip> SAKURA_USER=sakura IMAGE=ghcr.io/<github-user>/spam-api ./deploy/deploy.sh

# 3. Read the server's API key for the client
ssh -i ~/.ssh/id_rsa sakura@<vm-ip> "sudo cat /etc/spam-api/env"
```
`deploy/remote_run.sh` runs on the VM (called by `deploy.sh` and by CI). It pulls the image and restarts the container.
Pushing to `main` does the build, push and deploy from step 2 automatically through GitHub Actions.

### Live demo order
1. Start the API (`uvicorn` or Docker) and keep the log window visible (`docker logs -f spam-api` in tmux).
2. `python client.py "Free entry! Text WIN to 80085" "see you at lunch"` shows labels and latency in ms.
3. `python client.py --drift-demo` shows the `DRIFT ALERT` in the server logs and `drift_alert=True` in the client.

## Security
Secrets only via env vars / `/etc/spam-api/env` (see `.env.example`); `.env`, keys and `*.pem` are gitignored;
`chmod 600 ~/.ssh/id_rsa`; container runs as non-root; optional `X-API-Key` auth.

## Known gotchas
- ONNX `StringNormalizer` needs an `en_US.UTF-8` locale (absent in `python:3.10-slim`), so lowercasing is done in `features.normalize()` for both training and serving.
- Port collision: `docker rm -f spam-api` before re-running (done in `remote_run.sh`).
