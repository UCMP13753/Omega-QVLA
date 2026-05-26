#!/bin/bash
#
# Bench watchdog with auto-resume.
#
# Polls every 60s (high mode) or 1200s (night mode). For each running bench
# in 4method_bench/, snapshots driver cmdline+environ for active shards;
# detects shards whose eval log hasn't advanced; kills + relaunches stuck
# shards on a fresh port.
#
# Mode toggle:
#   echo high  > /ceph/workspace/xinyu/custon_asr/.bench_mode
#   echo night > /ceph/workspace/xinyu/custon_asr/.bench_mode
#
# Per-shard resume limit: 2 (avoid infinite loops).

set -uo pipefail

ROOT_DIR=/ceph/workspace/xinyu/custon_asr
ALERTS="${ROOT_DIR}/results/watchdog_alerts.log"
RETRY_FILE=/tmp/watchdog_retries.txt
MODE_FILE="${BENCH_MODE_FILE:-/ceph/workspace/xinyu/custon_asr/.bench_mode}"
MAX_RETRIES=${MAX_RETRIES:-2}
PORT_BUMP=${PORT_BUMP:-200}
touch "$ALERTS" "$RETRY_FILE"

read_mode() {
    [ -f "$MODE_FILE" ] || { echo high; return; }
    local m=$(cat "$MODE_FILE" 2>/dev/null | tr -d '[:space:]' | tr A-Z a-z)
    [ "$m" = "night" ] && echo night || echo high
}

config_for_mode() {
    case "$1" in
        night) echo "1200 1500 2700" ;;     # poll warn kill
        *)     echo "60 300 600" ;;
    esac
}

is_done() {
    local D="$1" s="$2"
    [ -f "$D/summaries/shard_$s.json" ]
}

retry_count() {
    local key="$1"
    grep -c "^${key}\$" "$RETRY_FILE" 2>/dev/null || echo 0
}

bump_retry() {
    local key="$1"
    echo "$key" >> "$RETRY_FILE"
}

# Find driver pid for cfg + shard
find_driver_pid() {
    local cfg="$1" shard_id="$2"
    ps -ef | grep "run_libero_duquant.*--run-shard $shard_id " | grep -F "$cfg" | grep -v grep | awk 'NR==1{print $2}'
}

# Snapshot driver cmdline + env for active shard (cheap, idempotent — overwrites)
snapshot_shard() {
    local cfg="$1" shard_id="$2"
    local D="${ROOT_DIR}/results/4method_bench/${cfg}"
    local pid=$(find_driver_pid "$cfg" "$shard_id")
    [ -z "$pid" ] && return
    [ -r "/proc/$pid/cmdline" ] || return
    tr '\0' '\n' < "/proc/$pid/cmdline" > "$D/.shard_${shard_id}.cmdline" 2>/dev/null
    tr '\0' '\n' < "/proc/$pid/environ" > "$D/.shard_${shard_id}.env" 2>/dev/null
}

# Kill all processes related to a shard (driver + server + eval workers)
kill_shard_processes() {
    local cfg="$1" shard_id="$2"
    # Match by output dir + log_suffix to avoid hitting other benches
    ps -ef | grep "log_suffix shard_${shard_id}" | grep -F "$cfg" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
    ps -ef | grep "run_libero_duquant.*--run-shard $shard_id " | grep -F "$cfg" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
    # Free GPU memory if those PIDs were holding GPU contexts
    sleep 2
}

# Relaunch a previously-killed shard from saved cmdline + env, with bumped port.
# Cleans the shard's stale outputs first so progress logs don't mix.
relaunch_shard() {
    local cfg="$1" shard_id="$2"
    local D="${ROOT_DIR}/results/4method_bench/${cfg}"
    local cmd_file="$D/.shard_${shard_id}.cmdline"
    local env_file="$D/.shard_${shard_id}.env"
    if [ ! -f "$cmd_file" ] || [ ! -f "$env_file" ]; then
        echo "[$(date '+%F %T')] CANNOT_RELAUNCH ${cfg}/shard_${shard_id}: no saved cmdline/env (snapshot never captured)" >> "$ALERTS"
        return 1
    fi

    # Wipe stale outputs for this shard so new log starts clean
    rm -f "$D/logs/eval/libero_eval_libero_goal_shard_${shard_id}.log"
    rm -f "$D/logs/server_shard_${shard_id}.log"
    rm -f "$D/logs/driver_shard_${shard_id}.log"
    rm -f "$D/logs/eval_shard_${shard_id}.stdout.log"
    rm -rf "$D/rollouts/shard_${shard_id}"
    rm -f "$D/summaries/shard_${shard_id}.json"

    # Read cmdline as array
    local args=()
    while IFS= read -r line; do args+=("$line"); done < "$cmd_file"

    # cmdline format: python script --run-shard <id> <gpu> <port> [tasks] suite ...
    local old_port="?"
    local new_port="?"
    if [ "${args[2]:-}" = "--run-shard" ] && [ -n "${args[5]:-}" ]; then
        old_port=${args[5]}
        new_port=$((old_port + PORT_BUMP))
        args[5]="$new_port"
    fi

    # Read env into KEY=VALUE form for `env -i`
    local env_kv=()
    while IFS= read -r line; do
        case "$line" in
            *=*) env_kv+=("$line") ;;
        esac
    done < "$env_file"

    local resume_log="$D/logs/shard_${shard_id}_resume_$(date +%s).log"
    nohup env -i "${env_kv[@]}" "${args[@]}" \
        > "$resume_log" 2>&1 &
    local new_pid=$!
    echo "[$(date '+%F %T')] RELAUNCH ${cfg}/shard_${shard_id} pid=${new_pid} port=${old_port}->${new_port} log=${resume_log##*/}" >> "$ALERTS"
}

while true; do
    now=$(date +%s)
    mode=$(read_mode)
    read POLL_S WARN_S KILL_S <<< "$(config_for_mode "$mode")"

    for D in "${ROOT_DIR}/results/4method_bench/"*/; do
        [ -d "$D" ] || continue
        cfg=$(basename "$D")
        # Only watch sixpack_* and tp_* benches (skip stale dirs)
        case "$cfg" in
            sixpack_*|tp_*|wmse_*|smoke5_*|screen_*) ;;
            *) continue ;;
        esac
        for s in 0 1 2 3; do
            log="$D/logs/eval/libero_eval_libero_goal_shard_$s.log"
            [ -f "$log" ] || continue
            if is_done "$D" "$s"; then continue; fi

            mtime=$(stat -c %Y "$log" 2>/dev/null)
            age=$((now - mtime))
            key="${cfg}/shard_$s"

            # Skip abandoned benches (log >24h stale) — there's nothing to resume.
            if [ "$age" -ge 86400 ]; then continue; fi

            # Snapshot cmdline+env while driver is still alive
            snapshot_shard "$cfg" "$s"

            if [ "$age" -ge "$KILL_S" ]; then
                local_retries=$(retry_count "$key")
                if [ "${local_retries:-0}" -ge "$MAX_RETRIES" ]; then
                    last=$(grep -F "EXHAUSTED $key" "$ALERTS" 2>/dev/null | tail -1)
                    [ -z "$last" ] && echo "[$(date '+%F %T')] EXHAUSTED $key (retries=${local_retries}, age=${age}s)" >> "$ALERTS"
                    continue
                fi
                ep=$(grep "episodes completed so far" "$log" 2>/dev/null | tail -1 | grep -oE "[0-9]+" | head -1)
                echo "[$(date '+%F %T')] KILL $key stuck for ${age}s ep=${ep:-?} retry=${local_retries:-0}/${MAX_RETRIES}" >> "$ALERTS"
                kill_shard_processes "$cfg" "$s"
                bump_retry "$key"
                relaunch_shard "$cfg" "$s"
            elif [ "$age" -ge "$WARN_S" ]; then
                # One WARN per shard per stuck-incident: skip if KILL already issued
                last_warn=$(grep -F "WARN $key" "$ALERTS" 2>/dev/null | tail -1)
                last_kill=$(grep -F "KILL $key" "$ALERTS" 2>/dev/null | tail -1)
                if [ -z "$last_warn" ] || [ -n "$last_kill" ]; then
                    ep=$(grep "episodes completed so far" "$log" 2>/dev/null | tail -1 | grep -oE "[0-9]+" | head -1)
                    echo "[$(date '+%F %T')] WARN $key no-write ${age}s ep=${ep:-?}" >> "$ALERTS"
                fi
            fi
        done

        # Auto-merge if all 4 shard summaries present and merged_summary missing
        if [ "$(ls "$D/summaries/"shard_*.json 2>/dev/null | wc -l)" = "4" ] && [ ! -f "$D/merged_summary.json" ]; then
            # The bench script merges via run_libero_duquant_benchmark_multi_gpu.py;
            # if shard summaries are present but merge wasn't done (driver crashed),
            # do a minimal aggregation here.
            python3 - <<EOF >> "$ALERTS" 2>&1
import json, glob, os
D = "$D"
shards = sorted(glob.glob(os.path.join(D, "summaries", "shard_*.json")))
total_ep = 0; total_succ = 0; tasks = []
for f in shards:
    d = json.load(open(f))
    total_ep += d["total_episodes"]; total_succ += d["total_successes"]
    tasks.extend(d.get("task_summaries", []))
merged = {"total_episodes": total_ep, "total_successes": total_succ,
          "total_success_rate": total_succ/total_ep if total_ep else 0,
          "task_summaries": tasks, "auto_merged_by": "bench_watchdog"}
with open(os.path.join(D, "merged_summary.json"), "w") as f:
    json.dump(merged, f, indent=2)
print(f"[$(date '+%F %T')] MERGED ${cfg}: {total_succ}/{total_ep} = {total_succ/total_ep*100:.1f}%")
EOF
        fi
    done
    sleep "$POLL_S"
done
