#!/bin/bash
# ArduPilot SITL with its *built-in* physics model (no Gazebo).
#
# This is the reference "does ArduPilot actually fly" configuration: `--model
# quad` runs SIM_Quad in-process, so there is no lockstep, no Gazebo renderer
# and no llvmpipe CPU starvation. Use it to validate control/logging; use the
# Gazebo scripts when you need camera frames.
set -uo pipefail
APM_DIR=${APM_DIR:-/home/drone/ardupilot}
REPO=${REPO:-/mnt/e/drone-vla}
SITLLOG=/tmp/apm_native.log
PORT=${APM_PORT:-5760}

pkill -9 -f 'arducopte[r]' 2>/dev/null
pkill -9 -f 'sim_vehicl[e]' 2>/dev/null
sleep 2
: > "$SITLLOG"

cd "$APM_DIR" || exit 1
setsid nohup python3 Tools/autotest/sim_vehicle.py \
  -v ArduCopter --model quad --no-mavproxy --no-rebuild \
  > "$SITLLOG" 2>&1 < /dev/null &

echo "waiting for SITL ..."
for i in $(seq 1 60); do
  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "SITL listening on tcp:$PORT after ${i}s"
    break
  fi
  sleep 1
done
tail -3 "$SITLLOG"
