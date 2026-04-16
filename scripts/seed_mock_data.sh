#!/usr/bin/env bash
set -euo pipefail

uv run python -m backend.data_generation.generator "$@"
