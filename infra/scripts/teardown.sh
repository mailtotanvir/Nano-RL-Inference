#!/usr/bin/env bash
set -euo pipefail

PROJECT="${1:?project required}"
INSTANCE="${2:?instance required}"
ZONE="${3:?zone required}"

LABEL="project=mini-ai-stack"
CURRENT_LABEL="$(gcloud compute instances describe "$INSTANCE" --project "$PROJECT" --zone "$ZONE" --format='value(labels.project)' 2>/dev/null || true)"
if [[ "$CURRENT_LABEL" != "mini-ai-stack" ]]; then
  printf '%s\n' "refusing to delete unlabeled instance: $INSTANCE" >&2
  exit 64
fi
gcloud compute instances delete "$INSTANCE" --project "$PROJECT" --zone "$ZONE" --quiet
gcloud compute instances list --project "$PROJECT" --filter="labels.$LABEL" --format='table(name,zone,status)'
