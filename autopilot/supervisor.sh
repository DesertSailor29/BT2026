#!/usr/bin/env bash
# Autopilot supervisor: watches the autopilot jobs (one per lane: gpu, cpu); whenever it ends (or a session asked for a wake-up, or nothing
# runs), it starts a *new* headless Claude Code session in the repo with a wake-up report. The session reads
# autopilot/MISSION.md (the handoff), decides the next step, starts it with autopilot/ap, writes the notes, pushes,
# and exits. No session ever waits for a job; the supervisor does the waiting (costs nothing).
#
# Start on UCloud (detached, like every job), after installing + logging in Claude Code (maisi/GUIDE.md):
#   setsid nohup bash -c 'source "$PROJ/BT2026/nnunet/env.sh" && bash "$PROJ/BT2026/autopilot/supervisor.sh" --hours 47' \
#     > "$PROJ/logs/autopilot_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
# Smoke test first (~5 min, two tiny sessions, no GPU):  ... supervisor.sh --smoke
# Restart after a crash: the same command; the deadline of the first start is kept ($AP_DIR/deadline).
# Stop: autopilot/ap stop 'reason'  (or kill the supervisor pid; jobs keep running either way).
#
# Options: --hours H | --until 'date spec'  deadline (first start only; = when the UCloud job ends, minus margin)
#          --smoke                          end-to-end test in $PROJ/autopilot_smoke
# Env: AP_DIR ($PROJ/autopilot)  AP_BRANCH (autopilot)  CLAUDE_BIN (claude)  AP_MODEL (account default)
#      AP_PERMISSION_MODE (bypassPermissions)  AP_MAX_BUDGET_USD (unset)  AP_SESSION_MIN (120: hard session timeout)
#      AP_MAX_SESSIONS (40)  AP_MAX_FAILS (6)  AP_POLL (60 s)  AP_HEARTBEAT_H (8: wake a session at least this often)
#      AP_FINAL_MIN (60: the final session starts this long before the deadline)  AP_RETRY_MIN (10: first retry wait)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
: "${PROJ:?source nnunet/env.sh first}"
SMOKE=0 HOURS="" UNTIL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --hours) HOURS="$2"; shift 2 ;;
    --until) UNTIL="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
if [ "$SMOKE" = 1 ]; then export AP_DIR="${AP_DIR:-$PROJ/autopilot_smoke}"; rm -rf "$AP_DIR"; HOURS=0.5; fi
export AP_DIR="${AP_DIR:-$PROJ/autopilot}"
BRANCH="${AP_BRANCH:-autopilot}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
MODE="${AP_PERMISSION_MODE:-bypassPermissions}"
SESSION_MIN="${AP_SESSION_MIN:-120}"
MAX_SESSIONS="${AP_MAX_SESSIONS:-40}"
MAX_FAILS="${AP_MAX_FAILS:-6}"
POLL="${AP_POLL:-60}"
HEARTBEAT_S=$(python3 -c "print(int(float('${AP_HEARTBEAT_H:-8}')*3600))")
FINAL_S=$(( ${AP_FINAL_MIN:-60} * 60 ))
AP="$HERE/ap"
mkdir -p "$AP_DIR/sessions" "$AP_DIR/jobs" "$PROJ/logs"
log() { echo "$(date '+%F %T') $*"; }
die() { log "FATAL: $*"; exit 1; }

# one supervisor per AP_DIR
old="$(cat "$AP_DIR/supervisor.pid" 2>/dev/null || true)"
if [ -n "$old" ] && [ "$old" != $$ ] && kill -0 "$old" 2>/dev/null; then die "supervisor already running (pid $old)"; fi
echo $$ > "$AP_DIR/supervisor.pid"

# deadline: set once, kept across restarts
if [ ! -f "$AP_DIR/deadline" ]; then
  if [ -n "$UNTIL" ]; then date -d "$UNTIL" +%s > "$AP_DIR/deadline" || die "bad --until"
  elif [ -n "$HOURS" ]; then python3 -c "import time; print(int(time.time() + float('$HOURS') * 3600))" > "$AP_DIR/deadline"
  else die "first start needs --hours H or --until 'date' (when the UCloud job ends, minus a margin)"; fi
fi
DEADLINE="$(cat "$AP_DIR/deadline")"
log "supervisor pid $$, AP_DIR $AP_DIR, deadline $(date -d "@$DEADLINE" '+%a %d %b %H:%M %Z'), mode $MODE, smoke $SMOKE"

command -v "$CLAUDE_BIN" > /dev/null || die "no '$CLAUDE_BIN' on PATH (install Claude Code, maisi/GUIDE.md)"
# bypassPermissions refuses to run as root unless the environment says it is a sandbox (UCloud jobs are containers)
[ "$(id -u)" = 0 ] && [ "$MODE" = bypassPermissions ] && export IS_SANDBOX=1

# all autopilot work happens on its own branch, main stays as the user left it
cd "$REPO" || die "no repo"
cur="$(git branch --show-current)"
if [ "$cur" != "$BRANCH" ]; then
  [ -z "$(git status --porcelain --untracked-files=no)" ] || die "uncommitted changes on $cur: commit or stash first"
  git fetch -q origin 2>/dev/null || true
  if git show-ref -q "refs/heads/$BRANCH"; then git switch -q "$BRANCH"
  elif git show-ref -q "refs/remotes/origin/$BRANCH"; then git switch -q -c "$BRANCH" --track "origin/$BRANCH"
  else git switch -q -c "$BRANCH"; fi
  log "switched $cur -> $BRANCH"
fi
[ "$SMOKE" = 1 ] || git push -q -u origin "$BRANCH" 2>&1 | tail -2 || log "push failed (sessions will retry)"

LANES="gpu cpu"  # one watched job per lane (ap start --lane gpu|cpu)
job_alive() {  # job_alive <lane>
  [ -f "$AP_DIR/job_$1" ] || return 1
  local JOB_PID=""
  # shellcheck disable=SC1090
  source "$AP_DIR/job_$1"
  [ -n "$JOB_PID" ] && kill -0 "$JOB_PID" 2>/dev/null
}
any_alive() { local l; for l in $LANES; do job_alive "$l" && return 0; done; return 1; }
ended_unhandled() {  # ended_unhandled <lane>
  [ -f "$AP_DIR/job_$1" ] && ! job_alive "$1" || return 1
  local JOB_EXIT_FILE=""
  # shellcheck disable=SC1090
  source "$AP_DIR/job_$1"
  [ "$(cat "$AP_DIR/handled_$1" 2>/dev/null)" != "$JOB_EXIT_FILE" ]
}
any_ended_unhandled() { local l; for l in $LANES; do ended_unhandled "$l" && return 0; done; return 1; }
wake_due() { [ -f "$AP_DIR/wake_at" ] && [ "$(date +%s)" -ge "$(cat "$AP_DIR/wake_at")" ]; }
wake_future() { [ -f "$AP_DIR/wake_at" ] && [ "$(date +%s)" -lt "$(cat "$AP_DIR/wake_at")" ]; }

report() {  # the wake-up report (the "hook" a session reacts to)
  local event="$1" n="$2" now; now=$(date +%s)
  echo "# Wake-up report: session $n, event \`$event\`"
  echo
  echo "- now: $(date '+%a %d %b %Y %H:%M %Z')"
  echo "- deadline: $(date -d "@$DEADLINE" '+%a %d %b %H:%M %Z'), $(python3 -c "print(round(($DEADLINE-$now)/3600, 2))") h left (final session starts $(( FINAL_S / 60 )) min before)"
  echo "- sessions so far: $((n - 1)) (max $MAX_SESSIONS); failed sessions in a row: $FAILS"
  echo "- branch: $(git -C "$REPO" branch --show-current), last commit: $(git -C "$REPO" log -1 --format='%h %s (%cr)')"
  case "$event" in
    job_finished) echo "- meaning: a watched job ended (see the lanes below). Evaluate it, then start the next step for that lane." ;;
    check) echo "- meaning: a wake-up you (or the heartbeat) asked for; jobs are still running. Check progress/health; usually just set the next wake-up and end." ;;
    idle) echo "- meaning: no job is running (first session, a wake-up you asked for, or the last session failed / started nothing). If the job shown below has not been evaluated yet in notes/AUTOPILOT.md, evaluate it first; then decide and start the next step." ;;
    final) echo "- meaning: FINAL session. Do not start GPU work. Write the summary for the user (MISSION.md 'Final session'), push, then \`autopilot/ap stop final\`." ;;
    smoke1|smoke2) echo "- meaning: smoke test, follow the instructions above." ;;
  esac
  echo
  local lane shown=0
  for lane in $LANES; do
    [ -f "$AP_DIR/job_$lane" ] || continue
    shown=1
    local JOB_NAME JOB_PID JOB_LOG JOB_START JOB_EXIT_FILE JOB_CMD_FILE rc t_end
    # shellcheck disable=SC1090
    source "$AP_DIR/job_$lane"
    echo "## $lane lane: $JOB_NAME"
    echo
    echo "- command: \`$(tr '\n' ' ' < "$JOB_CMD_FILE" 2>/dev/null)\`"
    echo "- pid $JOB_PID, started $(date -d "@$JOB_START" '+%a %d %b %H:%M'), log \`$JOB_LOG\`"
    if job_alive "$lane"; then
      echo "- state: RUNNING for $(python3 -c "print(round(($now-$JOB_START)/3600, 2))") h"
    else
      [ "$(cat "$AP_DIR/handled_$lane" 2>/dev/null)" = "$JOB_EXIT_FILE" ] && echo "- (this end was already reported to an earlier session)"
      if [ -f "$JOB_EXIT_FILE" ]; then
        read -r rc t_end < "$JOB_EXIT_FILE"
        echo "- state: ENDED $(date -d "@$t_end" '+%a %d %b %H:%M') with exit code **$rc** after $(python3 -c "print(round(($t_end-$JOB_START)/3600, 2))") h"
      else
        echo "- state: GONE without an exit file (killed? OOM? UCloud restart?): check the log"
      fi
    fi
    echo "- error-looking lines in the log: $(grep -c -i -E 'error|traceback|exception|killed|nan' "$JOB_LOG" 2>/dev/null; true)"
    echo
    echo '```'
    tail -n 50 "$JOB_LOG" 2>/dev/null | cut -c1-300
    echo '```'
    echo
  done
  [ "$shown" = 1 ] || { echo "## No job has been started by the autopilot yet."; echo; }
  echo "## Machine"
  echo
  echo '```'
  ls -1 "$PROJ" 2>/dev/null | grep -E '__(ETA|DONE|STOPPED)_' | head -20
  command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  command -v nvidia-smi > /dev/null && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | head -5
  df -h "$PROJ" 2>/dev/null | tail -1
  echo '```'
  local prev; prev=$(ls -1 "$AP_DIR"/sessions/*.out.md 2>/dev/null | tail -1)
  if [ -n "$prev" ]; then
    echo
    echo "## Last session's final message ($(basename "$prev"))"
    echo
    tail -n 40 "$prev" | cut -c1-400
  fi
  if [ -n "${LAST_ERR:-}" ] && [ -s "$LAST_ERR" ]; then
    echo
    echo "## stderr of the last failed session"
    echo '```'
    tail -n 20 "$LAST_ERR" | cut -c1-300
    echo '```'
  fi
}

prompt_head() {
  local event="$1"
  if [ "$event" = smoke1 ]; then cat <<'EOF'
SMOKE TEST of the BT2026 autopilot (no research work, do not read MISSION.md). Run exactly these and report each
result in one line: (1) `autopilot/ap status`; (2) `autopilot/ap start smoke 'sleep 20; echo smoke-job-done'`;
(3) `git push --dry-run origin HEAD` (only checks credentials); (4) `nvidia-smi -L || true`. Then end.
EOF
  elif [ "$event" = smoke2 ]; then cat <<'EOF'
SMOKE TEST part 2. Report in one line the job name, exit code and the last log line from the wake-up report below,
then run `autopilot/ap stop smoke-ok` and end.
EOF
  else cat <<EOF
You are an unattended autopilot session of the BT2026 research project (UCloud, started by autopilot/supervisor.sh).
The user is away for days and will not answer: never ask questions, decide and record why. First read
autopilot/MISSION.md completely (your handoff; where it differs from CLAUDE.md it wins), then the newest entries of
notes/AUTOPILOT.md (the journal of earlier sessions). Then act on the wake-up report below, following MISSION.md
"Every session". Do not wait for jobs inside this session: start the next job with autopilot/ap, set a wake-up if
useful, write the notes, commit + push to branch $BRANCH, and end. AP_DIR=$AP_DIR.

EOF
  fi
}

run_session() {
  local event="$1" n sid base rc
  n=$(( $(cat "$AP_DIR/counter" 2>/dev/null || echo 0) + 1 ))
  echo "$n" > "$AP_DIR/counter"
  sid="$(python3 -c 'import uuid; print(uuid.uuid4())')"
  base="$AP_DIR/sessions/$(printf %03d "$n")_${event}"
  { prompt_head "$event"; report "$event" "$n"; } > "$base.prompt.md"
  echo "$sid" > "$base.session_id"
  local lane JOB_EXIT_FILE
  for lane in $LANES; do
    if ended_unhandled "$lane"; then
      # shellcheck disable=SC1090
      source "$AP_DIR/job_$lane"
      echo "$JOB_EXIT_FILE" > "$AP_DIR/handled_$lane"
    fi
  done
  rm -f "$AP_DIR/wake_at"
  log "session $n ($event) id $sid -> $base.out.md"
  local args=(-p --session-id "$sid" --permission-mode "$MODE" --settings "$HERE/settings.json" --output-format text)
  [ -n "${AP_MODEL:-}" ] && args+=(--model "$AP_MODEL")
  [ -n "${AP_MAX_BUDGET_USD:-}" ] && args+=(--max-budget-usd "$AP_MAX_BUDGET_USD")
  (cd "$REPO" && AP_DIR="$AP_DIR" timeout --kill-after=60 "${SESSION_MIN}m" "$CLAUDE_BIN" "${args[@]}" \
      < "$base.prompt.md" > "$base.out.md" 2> "$base.err.log")
  rc=$?
  log "session $n ended rc $rc ($(wc -c < "$base.out.md") bytes out); resume with: claude --resume $sid"
  LAST_ERR="$base.err.log"
  return $rc
}

FAILS=0 FINAL_DONE=0 LAST_ERR=""
[ "$SMOKE" = 1 ] && SMOKE_STEP=1
while true; do
  [ -f "$AP_DIR/STOP" ] && { log "STOP: $(cat "$AP_DIR/STOP")"; break; }
  now=$(date +%s)
  n_done=$(cat "$AP_DIR/counter" 2>/dev/null || echo 0)
  [ "$n_done" -ge "$MAX_SESSIONS" ] && { "$AP" stop "max sessions ($MAX_SESSIONS) reached" > /dev/null; continue; }

  if [ "$SMOKE" = 1 ]; then
    if [ "$SMOKE_STEP" = 1 ]; then run_session smoke1; SMOKE_STEP=2; continue; fi
    if any_ended_unhandled; then run_session smoke2; continue; fi
    if ! any_alive; then [ "$now" -ge "$DEADLINE" ] && { log "smoke: timed out"; break; }; fi
    sleep 5; continue
  fi

  if [ "$FINAL_DONE" = 0 ] && [ "$now" -ge $(( DEADLINE - FINAL_S )) ]; then
    event=final
  elif any_ended_unhandled; then event=job_finished
  elif any_alive; then
    if wake_due; then event=check
    elif [ ! -f "$AP_DIR/wake_at" ] && [ $(( now - $(stat -c %Y "$AP_DIR/counter" 2>/dev/null || echo "$now") )) -ge "$HEARTBEAT_S" ]; then event=check
    else sleep "$POLL"; continue; fi
  elif wake_future; then sleep "$POLL"; continue
  elif wake_due; then event=idle
  elif [ "$FAILS" -gt 0 ] && [ "$now" -lt "${RETRY_AT:-0}" ]; then sleep "$POLL"; continue
  else event=idle; fi

  run_session "$event"; rc=$?
  if [ "$event" = final ]; then FINAL_DONE=1; [ -f "$AP_DIR/STOP" ] || "$AP" stop "deadline: final session done" > /dev/null; continue; fi
  if [ -f "$AP_DIR/STOP" ] || any_alive || wake_future; then
    FAILS=0
  else
    FAILS=$((FAILS + 1))
    wait_min=$(( ${AP_RETRY_MIN:-10} * (1 << (FAILS - 1)) )); [ "$wait_min" -gt 120 ] && wait_min=120
    RETRY_AT=$(( $(date +%s) + wait_min * 60 ))
    log "session $(cat "$AP_DIR/counter") left nothing running, no wake-up, no STOP (rc $rc): failure $FAILS/$MAX_FAILS, retry in $wait_min min"
    tail -n 5 "$LAST_ERR" 2>/dev/null | sed 's/^/  stderr: /'
    [ "$FAILS" -ge "$MAX_FAILS" ] && "$AP" stop "$MAX_FAILS failed sessions in a row (last stderr: $LAST_ERR)" > /dev/null
  fi
done
log "supervisor ends; a running job (if any) keeps running"
rm -f "$AP_DIR/supervisor.pid"
