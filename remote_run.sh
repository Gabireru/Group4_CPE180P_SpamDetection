#!/usr/bin/env bash
# Executed on the VM (by CI or by deploy.sh). Expects $IMAGE.
set -euo pipefail
sudo docker pull "$IMAGE:latest"
sudo docker rm -f spam-api 2>/dev/null || true
sudo docker run -d --name spam-api --restart unless-stopped \
  -p 8000:8000 --env-file /etc/spam-api/env "$IMAGE:latest"
sleep 2 && curl -fs localhost:8000/health && echo " deployed"
# Live logs (demo):  tmux new -s logs 'sudo docker logs -f spam-api'
