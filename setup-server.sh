#!/usr/bin/env bash
set -euo pipefail

# Setup script for deploying dtcc-sim on a fresh Ubuntu server.
#
# Prerequisites:
#   - Ubuntu 22.04 or 24.04
#   - SSH key or GitHub token with access to dtcc-platform/dtcc-dataset-downloader (private repo)
#
# Usage:
#   bash setup-server.sh
#
# To use a GitHub token instead of SSH:
#   GITHUB_TOKEN=ghp_xxx bash setup-server.sh

INSTALL_DIR="${INSTALL_DIR:-/opt/dtcc-sim}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"
HOST_PORT="${HOST_PORT:-8000}"

# Determine git clone URL style based on whether a token is provided
if [ -n "${GITHUB_TOKEN:-}" ]; then
    GH_PREFIX="https://${GITHUB_TOKEN}@github.com"
else
    GH_PREFIX="git@github.com:"
fi

echo "==> Installing Docker"
if ! command -v docker &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
    sudo usermod -aG docker "$USER"
    echo "Docker installed. You may need to log out and back in for group membership to take effect."
else
    echo "Docker already installed, skipping."
fi

echo "==> Cloning repositories"
sudo mkdir -p "$INSTALL_DIR"
sudo chown "$USER:$USER" "$INSTALL_DIR"

if [ ! -d "$INSTALL_DIR/.git" ]; then
    git clone "${GH_PREFIX}dtcc-platform/dtcc-sim.git" "$INSTALL_DIR"
else
    echo "dtcc-sim already cloned, pulling latest."
    git -C "$INSTALL_DIR" pull
fi

if [ ! -d "$INSTALL_DIR/dtcc-dataset-downloader/.git" ]; then
    git clone -b develop "${GH_PREFIX}dtcc-platform/dtcc-dataset-downloader.git" "$INSTALL_DIR/dtcc-dataset-downloader"
else
    echo "dtcc-dataset-downloader already cloned, pulling latest."
    git -C "$INSTALL_DIR/dtcc-dataset-downloader" pull
fi

if [ ! -d "$INSTALL_DIR/dtcc-tetgen-wrapper/.git" ]; then
    git clone "${GH_PREFIX}dtcc-platform/dtcc-tetgen-wrapper.git" "$INSTALL_DIR/dtcc-tetgen-wrapper"
else
    echo "dtcc-tetgen-wrapper already cloned, pulling latest."
    git -C "$INSTALL_DIR/dtcc-tetgen-wrapper" pull
fi

echo "==> Building Docker image"
cd "$INSTALL_DIR"
sudo docker build -t dtcc-sim .

echo "==> Starting container"
sudo docker rm -f dtcc-sim 2>/dev/null || true
sudo docker run -d \
    --name dtcc-sim \
    --restart unless-stopped \
    -p "${HOST_PORT}:${CONTAINER_PORT}" \
    dtcc-sim

echo ""
echo "==> Done. dtcc-sim is running on port ${HOST_PORT}."
echo "    Test with: curl http://localhost:${HOST_PORT}/"
