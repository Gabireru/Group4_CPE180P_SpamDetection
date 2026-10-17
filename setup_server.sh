#!/usr/bin/env bash
# One-time hardening + Docker install on the Sakura Cloud VM (Ubuntu 24.04). Run as root/sudo.
set -euo pipefail
apt-get update && apt-get install -y docker.io ufw fail2ban tmux
systemctl enable --now docker fail2ban

# Packet filtering: only SSH (22) and the API (8000)
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw allow 8000/tcp
ufw --force enable

# SSH hardening: key auth only, no root login
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
systemctl restart ssh

# Secrets live here (root-only), never in git
install -d -m 700 /etc/spam-api
[ -f /etc/spam-api/env ] || { echo "API_KEY=$(openssl rand -hex 24)" > /etc/spam-api/env; chmod 600 /etc/spam-api/env; }
echo "Done. API key stored in /etc/spam-api/env"
