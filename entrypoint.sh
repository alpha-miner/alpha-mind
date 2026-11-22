#!/bin/sh
set -eu

export DB_VENDOR="${DB_VENDOR:-mysql}"
export FACTOR_TABLES="${FACTOR_TABLES:-factor_momentum}"
: "${DB_URI:?Set DB_URI to the database connection URL}"
: "${JUPYTER_TOKEN:?Set JUPYTER_TOKEN for the notebook server}"

exec jupyter lab --ip="0.0.0.0" --port=8080 --allow-root \
    --ServerApp.token="${JUPYTER_TOKEN}" --ServerApp.password=''
