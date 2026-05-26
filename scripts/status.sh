#!/bin/bash
# Real-time pipeline status dashboard.

ROOT=/ceph/workspace/xinyu/custon_asr/results
clear 2>/dev/null

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  custon_asr pipeline status — $(date +%T)"
echo "═══════════════════════════════════════════════════════════════════════════════"

echo ""
echo "▶ GPU state"
echo "──────────────────────────────────────────"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    awk -F',' '{
        used=$2/1024; total=$3/1024;
        bar=""; pct=int($4);
        nbars=int(pct/5);
        for(i=0;i<nbars;i++) bar=bar"█";
        for(i=nbars;i<20;i++) bar=bar"·";
        printf "  GPU %s | %5.1f / %5.1f GB | %s %3d%%\n", $1, used, total, bar, pct
    }'

echo ""
echo "▶ Builds (16 total)"
echo "──────────────────────────────────────────"
done_n=0; run_n=0
for d in "$ROOT"/multisuite_packs/*/; do
    [ -d "$d" ] || continue
    label=$(basename "$d")
    if [ -f "$d/quantized.pt" ]; then
        done_n=$((done_n+1))
    else
        run_n=$((run_n+1))
        nlayer=$(grep -c "^\[ASPQ-GPTQ\] saved" "$d/build.log" 2>/dev/null)
        printf "  ⏳ %-30s %3d/180 layers\n" "$label" "$nlayer"
    fi
done
total_b=$((done_n + run_n))
echo "  ✓ $done_n done, ⏳ $run_n running, $((16 - total_b)) queued"

echo ""
echo "▶ Bench progress (96 configs target)"
echo "──────────────────────────────────────────"
done_n=0; run_n=0; total=0
declare -A METHOD_DONE METHOD_TOTAL
for method in duquant quantvla pure_gptq aspq_gptq_K16 aspq_gptq_K32 aspq_gptq_K64; do
    METHOD_DONE[$method]=0
    METHOD_TOTAL[$method]=16  # 4 suites × 2 dits × 2 bits
done
for d in "$ROOT"/4method_bench/*/; do
    [ -d "$d" ] || continue
    label=$(basename "$d")
    total=$((total+1))
    method=""
    case "$label" in
        duquant_*)   method="duquant" ;;
        quantvla_*)  method="quantvla" ;;
        pure_gptq_*) method="pure_gptq" ;;
        aspq_gptq_K16_*) method="aspq_gptq_K16" ;;
        aspq_gptq_K32_*) method="aspq_gptq_K32" ;;
        aspq_gptq_K64_*) method="aspq_gptq_K64" ;;
    esac
    if [ -f "$d/merged_summary.json" ]; then
        done_n=$((done_n+1))
        [ -n "$method" ] && METHOD_DONE[$method]=$(( ${METHOD_DONE[$method]:-0} + 1 ))
    else
        # Check progress
        eps=0
        for log in "$d"/logs/eval/*.log; do
            [ -f "$log" ] || continue
            e=$(grep "^# episodes" "$log" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
            [ -n "$e" ] && eps=$((eps+e))
        done
        [ "$eps" -gt 0 ] && run_n=$((run_n+1))
    fi
done
echo "  by method:"
for method in duquant quantvla pure_gptq aspq_gptq_K16 aspq_gptq_K32 aspq_gptq_K64; do
    d=${METHOD_DONE[$method]}
    t=${METHOD_TOTAL[$method]}
    bar=""
    for i in $(seq 0 $((d-1))); do bar=$bar"█"; done
    for i in $(seq $d $((t-1))); do bar=$bar"·"; done
    printf "    %-15s [%s] %2d/%2d\n" "$method" "$bar" "$d" "$t"
done
echo ""
echo "  total: ✓ $done_n done, ⏳ $run_n running, $((96 - done_n - run_n)) queued"

echo ""
echo "▶ Recent results"
echo "──────────────────────────────────────────"
ls -t "$ROOT"/4method_bench/*/merged_summary.json 2>/dev/null | head -8 | while read f; do
    label=$(basename "$(dirname "$f")")
    s=$(jq -r '"\(.total_successes)/\(.total_episodes) (\(.total_success_rate*100|round)%)"' "$f" 2>/dev/null)
    printf "  %-45s %s\n" "$label" "$s"
done
echo ""
