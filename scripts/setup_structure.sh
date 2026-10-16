#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
  src/agents \
  src/db \
  src/engine \
  src/alerting \
  tests/data \
  tests/engine \
  tests/agent

for package in agents db engine alerting; do
  touch "src/${package}/__init__.py"
done

printf 'Repository structure created.\n'
