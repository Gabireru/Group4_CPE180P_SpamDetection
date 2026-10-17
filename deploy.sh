#!/usr/bin/env bash
# Manual deploy from the client workstation (Windows 11 WSL / Ubuntu).
# Usage: SAKURA_HOST=1.2.3.4 SAKURA_USER=ubuntu IMAGE=ghcr.io/you/spam-api ./deploy/deploy.sh
set -euo pipefail
: "${SAKURA_HOST:?}" "${SAKURA_USER:?}" "${IMAGE:?}"
python train.py && docker build -t "$IMAGE:latest" . && docker push "$IMAGE:latest"
ssh -i ~/.ssh/id_rsa "$SAKURA_USER@$SAKURA_HOST" "IMAGE=$IMAGE bash -s" < deploy/remote_run.sh
