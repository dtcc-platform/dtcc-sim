#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DTCC_SIM_IMAGE="${DTCC_SIM_IMAGE:-dtcc-sim}"
export DTCC_SIM_TAG="${DTCC_SIM_TAG:-local}"
export DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
export DTCC_TETGEN_WRAPPER_REF="${DTCC_TETGEN_WRAPPER_REF:-main}"
export TETGEN_VERSION="${TETGEN_VERSION:-v1.6.0}"

echo "Building ${DTCC_SIM_IMAGE}:${DTCC_SIM_TAG} via docker compose"

cd "${SCRIPT_DIR}"
docker compose build dtcc-sim dtcc-sim-worker
