#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build, optionally validate, and push the IWSLT26 DockerHub image on an H100 host.

Required environment:
  DOCKERHUB_REPO       DockerHub repository, e.g. user/cascade-simul-iwslt26
  HF_TOKEN             Hugging Face token with access to the gated Gemma model
    or HF_TOKEN_FILE   Path to a file containing that token

Optional environment:
  DOCKERHUB_USERNAME   DockerHub username for non-interactive login
  DOCKERHUB_TOKEN      DockerHub token/password for non-interactive login
  IMAGE_TAG            Image tag; defaults to the current git commit or UTC timestamp
  PUSH                 1 to push after build, 0 to leave the image local (default: 1)
  RUN_VALIDATION       auto, 1, or 0. auto validates only when VALIDATION_WAV exists
  VALIDATION_WAV       WAV used for one-clip smoke validation
  VALIDATION_PRESET    main_low_latency or main_high_latency (default: main_low_latency)
  VALIDATION_TGT_LANG_CODE
                       de, it, or zh (default: de)
  NO_CACHE             1 to pass --no-cache to docker build
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

log() {
  printf '[build_push] %s\n' "$*" >&2
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

command -v docker >/dev/null 2>&1 || die "docker is not installed on this host"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DOCKERHUB_REPO="${DOCKERHUB_REPO:-}"
[[ -n "$DOCKERHUB_REPO" ]] || die "DOCKERHUB_REPO is required"

if [[ -z "${HF_TOKEN_FILE:-}" && -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]]; then
  die "set HF_TOKEN_FILE or HF_TOKEN/HUGGING_FACE_HUB_TOKEN for the BuildKit secret"
fi

IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short=12 HEAD 2>/dev/null || date -u +%Y%m%d%H%M%S)}"
PUSH="${PUSH:-1}"
RUN_VALIDATION="${RUN_VALIDATION:-auto}"
VALIDATION_WAV="${VALIDATION_WAV:-data/smoke/alignatt_smoke18.wav}"
VALIDATION_PRESET="${VALIDATION_PRESET:-main_low_latency}"
VALIDATION_TGT_LANG_CODE="${VALIDATION_TGT_LANG_CODE:-de}"
DOCKER_PROGRESS="${DOCKER_PROGRESS:-plain}"

IMAGE_REF="${DOCKERHUB_REPO}:${IMAGE_TAG}"
LATEST_REF="${DOCKERHUB_REPO}:latest"

TOKEN_FILE_CREATED=""
VALIDATION_TMP=""
cleanup() {
  if [[ -n "$TOKEN_FILE_CREATED" ]]; then
    rm -f "$TOKEN_FILE_CREATED"
  fi
  if [[ -n "$VALIDATION_TMP" ]]; then
    rm -rf "$VALIDATION_TMP"
  fi
}
trap cleanup EXIT

if [[ -n "${HF_TOKEN_FILE:-}" ]]; then
  [[ -f "$HF_TOKEN_FILE" ]] || die "HF_TOKEN_FILE does not exist: $HF_TOKEN_FILE"
  BUILD_HF_TOKEN_FILE="$HF_TOKEN_FILE"
else
  TOKEN_FILE_CREATED="$(mktemp)"
  chmod 600 "$TOKEN_FILE_CREATED"
  printf '%s' "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" >"$TOKEN_FILE_CREATED"
  BUILD_HF_TOKEN_FILE="$TOKEN_FILE_CREATED"
fi

build_args=(
  docker build
  --pull
  --progress="$DOCKER_PROGRESS"
  --secret "id=hf_token,src=${BUILD_HF_TOKEN_FILE}"
  -t "$IMAGE_REF"
  -t "$LATEST_REF"
)
if [[ "${NO_CACHE:-0}" == "1" ]]; then
  build_args+=(--no-cache)
fi
build_args+=(.)

log "building $IMAGE_REF and $LATEST_REF"
DOCKER_BUILDKIT=1 "${build_args[@]}"

should_validate=0
case "$RUN_VALIDATION" in
  1|true|yes) should_validate=1 ;;
  0|false|no) should_validate=0 ;;
  auto)
    if [[ -f "$VALIDATION_WAV" ]]; then
      should_validate=1
    else
      log "validation skipped; VALIDATION_WAV not found: $VALIDATION_WAV"
    fi
    ;;
  *) die "RUN_VALIDATION must be auto, 1, or 0" ;;
esac

if [[ "$should_validate" == "1" ]]; then
  [[ -f "$VALIDATION_WAV" ]] || die "VALIDATION_WAV does not exist: $VALIDATION_WAV"
  VALIDATION_TMP="$(mktemp -d)"
  mkdir -p "$VALIDATION_TMP/in" "$VALIDATION_TMP/out"
  cp "$VALIDATION_WAV" "$VALIDATION_TMP/in/audio.wav"
  # Simulstream resolves wavlist entries relative to the wavlist directory.
  printf '%s\n' "audio.wav" >"$VALIDATION_TMP/in/wavlist.txt"

  log "validating $IMAGE_REF on one clip"
  docker run --gpus all --rm --ipc=host \
    -e "PRESET=$VALIDATION_PRESET" \
    -e "TGT_LANG_CODE=$VALIDATION_TGT_LANG_CODE" \
    -v "$VALIDATION_TMP/in:/io/wavs:ro" \
    -v "$VALIDATION_TMP/out:/io/out" \
    "$IMAGE_REF" \
    infer /io/wavs/wavlist.txt /io/out/metrics.jsonl

  [[ -s "$VALIDATION_TMP/out/metrics.jsonl" ]] || die "validation produced no metrics.jsonl"
  log "validation completed"
fi

if [[ "$PUSH" == "1" ]]; then
  if [[ -n "${DOCKERHUB_USERNAME:-}" && -n "${DOCKERHUB_TOKEN:-}" ]]; then
    log "logging in to DockerHub as $DOCKERHUB_USERNAME"
    printf '%s' "$DOCKERHUB_TOKEN" | docker login --username "$DOCKERHUB_USERNAME" --password-stdin
  fi

  log "pushing $IMAGE_REF"
  docker push "$IMAGE_REF"
  if [[ "$IMAGE_REF" != "$LATEST_REF" ]]; then
    log "pushing $LATEST_REF"
    docker push "$LATEST_REF"
  fi
else
  log "PUSH=0; leaving image local"
fi

log "done: $IMAGE_REF"
