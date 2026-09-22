#!/usr/bin/env bash
set -euo pipefail

PROJECT="${1:?project required}"
INSTANCE="${2:?instance required}"
ZONE="${3:?zone required}"
DEADLINE_SECONDS="${4:?deadline seconds required}"
START="$(date +%s)"

while true; do
  NOW="$(date +%s)"
  if (( NOW - START >= DEADLINE_SECONDS )); then
    gcloud compute instances stop "$INSTANCE" --project "$PROJECT" --zone "$ZONE" --quiet
    exit 0
  fi
  STATUS="$(gcloud compute instances describe "$INSTANCE" --project "$PROJECT" --zone "$ZONE" --format='value(status)' 2>/dev/null || true)"
  if [[ "$STATUS" != "RUNNING" ]]; then
    exit 0
  fi
  sleep 60
done
