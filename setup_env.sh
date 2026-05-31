#!/usr/bin/env bash
set -euo pipefail

uv venv
uv pip install -e /Users/mikkeldahl/carbontracker-tui
uv pip install -e .
