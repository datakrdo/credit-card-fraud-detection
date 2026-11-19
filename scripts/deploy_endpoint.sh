#!/usr/bin/env bash
# Point the `fraud-scoring` endpoint at the model version currently aliased @champion.
# Usage: scripts/deploy_endpoint.sh <profile> [catalog] [team_group]
# The team group gets CAN_QUERY only; the deploying user keeps CAN_MANAGE.
set -euo pipefail
PROFILE=${1:?profile}; CATALOG=${2:-fraud}; TEAM_GROUP=${3:-users}; NAME=fraud-scoring
VERSION=$(databricks model-versions get-by-alias "$CATALOG.ml.fraud_model" champion --profile "$PROFILE" | jq -r .version)
CFG=$(mktemp); trap 'rm -f "$CFG"' EXIT
sed "s/__CATALOG__/$CATALOG/g; s/__VERSION__/$VERSION/g" serving/endpoint.json > "$CFG"
if databricks serving-endpoints get "$NAME" --profile "$PROFILE" >/dev/null 2>&1; then
  databricks serving-endpoints update-config "$NAME" --json @"$CFG" --profile "$PROFILE"
else
  databricks serving-endpoints create --json "$(jq --arg n "$NAME" '{name: $n, config: .}' "$CFG")" --profile "$PROFILE"
fi
ENDPOINT_ID=$(databricks serving-endpoints get "$NAME" --profile "$PROFILE" | jq -r .id)
databricks serving-endpoints update-permissions "$ENDPOINT_ID" --profile "$PROFILE" \
  --json "{\"access_control_list\": [{\"group_name\": \"$TEAM_GROUP\", \"permission_level\": \"CAN_QUERY\"}]}"
# Inference table via AI Gateway (the legacy auto_capture_config is no longer accepted).
# Not available on every workspace (e.g. Free Edition): do not fail the deploy over it.
databricks serving-endpoints put-ai-gateway "$NAME" --profile "$PROFILE" \
  --json "{\"inference_table_config\": {\"catalog_name\": \"$CATALOG\", \"schema_name\": \"ml\", \"table_name_prefix\": \"fraud_scoring\", \"enabled\": true}}" \
  || echo "WARNING: inference table not enabled (unsupported here); endpoint is deployed without request logging."
