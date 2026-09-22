#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_GCP_CREATE:-}" != "YES" ]]; then
  printf '%s\n' 'Refusing cloud creation. Set CONFIRM_GCP_CREATE=YES only after explicit user approval.' >&2
  exit 64
fi

exec terraform apply "$@"
