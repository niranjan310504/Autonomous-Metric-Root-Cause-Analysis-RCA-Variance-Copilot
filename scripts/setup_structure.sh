#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
  src/core \
  src/data \
  src/engine \
  src/agent \
  src/ui \
  tests/data \
  tests/engine \
  tests/agent

for package in core data engine agent ui; do
  touch "src/${package}/__init__.py"
done

printf 'Repository structure created.\n'
