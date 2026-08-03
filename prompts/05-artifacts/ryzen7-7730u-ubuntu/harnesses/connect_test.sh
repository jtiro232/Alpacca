#!/bin/bash
# Package M: does the resident-server workflow actually avoid the load, and
# does it fall back cleanly when nothing is listening?
set -u
BIN=/home/ubuntu/alpaccaroo-venv/bin/alpaccaroo
MODEL=${1:-qwen3bmed}
PORT=${2:-8099}
OUT=${3:-/tmp/claude-1000/-home-ubuntu/e6768af3-d0a0-46f9-bd84-31033f28a776/scratchpad/connect}
mkdir -p "$OUT"
PROMPT="Name three colours."

echo "===== 1. fallback FIRST, with no server listening ====="
# ordering matters: prove the fallback path works before a server exists,
# so a passing result cannot be a server answering by accident
/usr/bin/time -f "%e s wall" $BIN run "$MODEL" --connect "http://127.0.0.1:$PORT" \
    "$PROMPT" -n 24 --temp 0 --seed 7 > "$OUT/fallback.out" 2> "$OUT/fallback.err"
echo "--- stderr ---"; cat "$OUT/fallback.err"
echo "--- stdout ---"; cat "$OUT/fallback.out"

echo
echo "===== 2. baseline: plain one-shot, no --connect ====="
/usr/bin/time -f "%e s wall" $BIN run "$MODEL" "$PROMPT" -n 24 --temp 0 --seed 7 \
    > "$OUT/oneshot.out" 2> "$OUT/oneshot.err"
echo "--- stderr ---"; cat "$OUT/oneshot.err"
echo "--- stdout ---"; cat "$OUT/oneshot.out"

echo
echo "===== 3. start a resident server ====="
$BIN serve "$MODEL" --port "$PORT" > "$OUT/serve.log" 2>&1 &
SERVE_PID=$!
echo "serve pid $SERVE_PID, waiting for /health..."
for i in $(seq 1 180); do
    if python3 -c "
import sys,urllib.request
try:
    urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=2)
except Exception:
    sys.exit(1)
" 2>/dev/null; then echo "up after ${i}s"; break; fi
    sleep 1
done

echo
echo "===== 4. connected one-shot (twice: first and repeat) ====="
for n in 1 2; do
  /usr/bin/time -f "%e s wall" $BIN run "$MODEL" --connect "http://127.0.0.1:$PORT" \
      "$PROMPT" -n 24 --temp 0 --seed 7 > "$OUT/connected$n.out" 2> "$OUT/connected$n.err"
  echo "--- run $n stderr ---"; cat "$OUT/connected$n.err"
  echo "--- run $n stdout ---"; cat "$OUT/connected$n.out"
done

echo
echo "===== 5. --connect with no URL (uses ALPACCAROO_HOST/PORT) ====="
ALPACCAROO_PORT=$PORT $BIN run "$MODEL" --connect "$PROMPT" -n 16 --temp 0 --seed 7 \
    > "$OUT/envurl.out" 2> "$OUT/envurl.err"
echo "--- stderr ---"; cat "$OUT/envurl.err"
echo "--- stdout ---"; cat "$OUT/envurl.out"

echo
echo "===== 6. asking a DIFFERENT model of a server that holds \$MODEL ====="
$BIN run llama1b --connect "http://127.0.0.1:$PORT" "$PROMPT" -n 16 --temp 0 --seed 7 \
    > "$OUT/wrongmodel.out" 2> "$OUT/wrongmodel.err"
echo "--- stderr ---"; cat "$OUT/wrongmodel.err"
echo "--- stdout ---"; cat "$OUT/wrongmodel.out"

echo
echo "===== teardown ====="
kill $SERVE_PID 2>/dev/null
wait $SERVE_PID 2>/dev/null
echo "done"
