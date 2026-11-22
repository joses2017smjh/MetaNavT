#!/usr/bin/env bash
# /opt/metanavit/deploy.sh on the single host: infra/modules/single-host's cloud-init writes it there from
# deploy/single-host/deploy.sh, with compose.yml beside it (doc/deploy.md, "The live demo: one host on the AWS Free
# plan").
#
#   deploy.sh sha256:<64 hex>   deploy that image of IMAGE_REPOSITORY (app.env), in order, each step ending the deploy
#                               if it fails:
#                                 1. read the secrets again (below), then pull the image by digest, while the api
#                                    still serves;
#                                 2. stop the api, then run the schema migrations and the index step once, in a
#                                    one-off container of the new image (the command of the ECS migrate task). One
#                                    after the other, not side by side: each loads the embedder, and two of them do not
#                                    fit a small host's memory. The demo is down from here until step 4 passes;
#                                 3. start the stack on the new image (docker compose up), and
#                                 4. wait until every service's health check passes: the api's is GET /health from
#                                    inside its container.
#                               If step 2 fails, or the new api never turns healthy, the previous image is started
#                               again and the deploy fails (a migration is not undone: they only move forward). The
#                               deploy workflow runs this through SSM Run Command
#                               (.github/workflows/deploy-single-host.yml); the first boot runs it from cloud-init; a
#                               rollback is a deploy of the previous digest.
#   deploy.sh --start           metanavit.service at every boot: read the secrets again, then `docker compose up -d`
#                               on the image the last deploy left in image.env.
#
# Secrets go into secrets.env under /run (memory only), root-only: CloudFront's origin header value and the optional
# Anthropic key from Parameter Store (SecureString, read with the instance's role), and the Postgres password, made
# here once and kept in /var/lib/metanavit (it never leaves the host). Each must be [A-Za-z0-9_-]+, so the env file
# needs no quoting.
set -euo pipefail
umask 077

HOME_DIR=${METANAVIT_HOME:-/opt/metanavit}
RUN_DIR=${METANAVIT_RUN:-/run/metanavit}
STATE_DIR=${METANAVIT_STATE:-/var/lib/metanavit}
RETRY_SECONDS=${METANAVIT_RETRY_SECONDS:-10}
WAIT_SECONDS=${METANAVIT_WAIT_SECONDS:-1200}
DIGEST='^sha256:[0-9a-f]{64}$'
VALUE='^[A-Za-z0-9_-]+$'

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

# The value of NAME in app.env (NAME=value lines written by Terraform); the file is read, never executed.
setting() {
  local value
  value=$(sed -n "s/^$1=//p" "$HOME_DIR/app.env" | tail -n 1)
  [ -n "$value" ] || die "$1 is not set in $HOME_DIR/app.env"
  printf '%s' "$value"
}

# A SecureString's value on stdout; exit 3 when no such parameter exists. Other errors are retried (at the first
# boot the instance's role can take a moment to be usable), then reported.
parameter() {
  local name=$1 attempt value
  for attempt in 1 2 3 4 5 6; do
    if value=$(aws ssm get-parameter --region "$(setting AWS_REGION)" --name "$name" --with-decryption \
                 --query Parameter.Value --output text 2>"$RUN_DIR/aws.err"); then
      printf '%s' "$value"
      return 0
    fi
    if grep -q ParameterNotFound "$RUN_DIR/aws.err"; then
      return 3
    fi
    log "reading $name failed (attempt $attempt): $(tr '\n' ' ' < "$RUN_DIR/aws.err")" >&2
    sleep "$RETRY_SECONDS"
  done
  return 1
}

write_secrets() {
  mkdir -p "$RUN_DIR" "$STATE_DIR"
  chmod 700 "$RUN_DIR" "$STATE_DIR"
  local origin_name key_name origin key status=0 password
  origin_name=$(setting ORIGIN_VERIFY_PARAMETER)
  key_name=$(setting ANTHROPIC_KEY_PARAMETER)
  origin=$(parameter "$origin_name") || die "cannot read CloudFront's origin header value ($origin_name)"
  key=$(parameter "$key_name") || status=$?
  case $status in
    0) ;;
    3) key=""; log "no Anthropic key stored ($key_name): answers are off, search works" ;;
    *) die "cannot read the Anthropic key parameter ($key_name)" ;;
  esac
  if [ ! -s "$STATE_DIR/postgres-password" ]; then
    head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$STATE_DIR/postgres-password"
  fi
  password=$(cat "$STATE_DIR/postgres-password")
  [[ $origin =~ $VALUE ]] || die "the origin header value ($origin_name) is not [A-Za-z0-9_-]+"
  [[ $password =~ $VALUE ]] || die "$STATE_DIR/postgres-password is not [A-Za-z0-9_-]+"
  [ -z "$key" ] || [[ $key =~ $VALUE ]] || die "the Anthropic key ($key_name) is not [A-Za-z0-9_-]+"
  printf 'ORIGIN_VERIFY_SECRET=%s\nANTHROPIC_API_KEY=%s\nPOSTGRES_PASSWORD=%s\n' "$origin" "$key" "$password" \
    > "$RUN_DIR/secrets.env.new"
  mv -f "$RUN_DIR/secrets.env.new" "$RUN_DIR/secrets.env"
}

# docker compose on /opt/metanavit with its three env files; IMAGE_ENV names the image's (image.env by default).
compose() {
  docker compose --project-name metanavit --project-directory "$HOME_DIR" --file "$HOME_DIR/compose.yml" \
    --env-file "$HOME_DIR/app.env" --env-file "${IMAGE_ENV:-$HOME_DIR/image.env}" --env-file "$RUN_DIR/secrets.env" \
    "$@"
}

# After a failed step: the image the stack ran before this deploy, started again (nothing on a first deploy). Not
# waited on: the deploy has failed already, and a second wait could outlast Run Command's execution timeout.
restore() {
  local previous=$1
  [ -n "$previous" ] || return 0
  log "starting the previous image again (${previous#METANAVIT_IMAGE=})" >&2
  printf '%s\n' "$previous" > "$HOME_DIR/image.env"
  compose up --detach --remove-orphans || log "the previous image did not start either" >&2
}

deploy() {
  local digest=$1 image previous=""
  [[ $digest =~ $DIGEST ]] || die "not an image digest: $digest (sha256:<64 hex>)"
  image="$(setting IMAGE_REPOSITORY)@$digest"
  write_secrets
  if [ -f "$HOME_DIR/image.env" ]; then
    previous=$(cat "$HOME_DIR/image.env")
  fi

  log "1. pull $image (the api still serves)"
  docker pull --quiet "$image"
  printf 'METANAVIT_IMAGE=%s\n' "$image" > "$RUN_DIR/image.env.next"

  log "2. stop the api, then the migrations and the index step in a one-off container of the new image"
  IMAGE_ENV="$RUN_DIR/image.env.next" compose stop api
  if ! IMAGE_ENV="$RUN_DIR/image.env.next" compose run --rm -T migrate; then
    restore "$previous"
    die "the migrations or the index step failed"
  fi

  log "3. the stack on $image"
  mv -f "$RUN_DIR/image.env.next" "$HOME_DIR/image.env"
  log "4. waiting for every service's health check (at most ${WAIT_SECONDS} s)"
  if ! compose up --detach --remove-orphans --wait --wait-timeout "$WAIT_SECONDS"; then
    compose ps --all >&2 || true
    compose logs --no-color --tail 50 api >&2 || true
    restore "$previous"
    die "the stack on $image did not become healthy"
  fi
  log "health: $(compose exec -T api python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=10).read().decode())")"
  # every image no container uses: the previous api image, so the disk keeps room for the next pull
  docker image prune --all --force > /dev/null
  log "DEPLOYED $image"
}

start() {
  write_secrets
  [ -f "$HOME_DIR/image.env" ] || die "no image deployed yet ($HOME_DIR/image.env): run deploy.sh sha256:<digest>"
  compose up --detach --remove-orphans
}

main() {
  mkdir -p "$RUN_DIR"
  chmod 700 "$RUN_DIR"
  # one deploy or start at a time (the first boot's deploy and metanavit.service, a deploy and a manual run)
  exec 9> "$RUN_DIR/deploy.lock"
  flock --wait 3600 9 || die "another deploy.sh still holds $RUN_DIR/deploy.lock"
  case ${1:-} in
    --start) start ;;
    sha256:*) deploy "$1" ;;
    *) echo "usage: $0 sha256:<64 hex> | --start" >&2; exit 2 ;;
  esac
}

main "$@"
