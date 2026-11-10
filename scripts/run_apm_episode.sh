#!/bin/bash
# Full ArduPilot episode: Gazebo (baylands) -> SITL -> camera capture -> pilot.
#
#   APM_WORLD=baylands_apm_cam.sdf APM_STEPS=300 RUN_DIR=... scripts/run_apm_episode.sh
#
# Outputs (in $RUN_DIR): frames.npy (T,256,256,3), frame_ts.csv, actions.csv,
# START / START.done markers.

set -uo pipefail

REPO=${REPO:-/mnt/e/drone-vla}
APM_DIR=${APM_DIR:-/home/drone/ardupilot}
APM_GZ_DIR=${APM_GZ_DIR:-/home/drone/ardupilot_gazebo}
APM_WORLD=${APM_WORLD:-baylands_apm_cam.sdf}
APM_VEHICLE=${APM_VEHICLE:-ArduCopter}
APM_FRAME=${APM_FRAME:-gazebo-iris}
APM_STEPS=${APM_STEPS:-300}
SIZE=${SIZE:-256}
RUN_DIR=${RUN_DIR:-$REPO/outputs/data/rotor_apm/run1}

PY=/usr/bin/python3          # pymavlink + gz bindings live here
GZLOG=/tmp/apm_gz.log
SITLLOG=/tmp/apm_sitl.log
CAPLOG=/tmp/apm_capture.log

# The <world name="..."> inside the SDF is what names the clock topic; it does
# not have to match the filename (baylands_apm_cam.sdf declares "baylands").
WORLD=""
for candidate in "$REPO/sim/ardupilot/worlds/$APM_WORLD" "$APM_GZ_DIR/worlds/$APM_WORLD"; do
  [ -f "$candidate" ] || continue
  WORLD=$(sed -n "s/.*<world name=['\"]\([^'\"]*\)['\"].*/\1/p" "$candidate" | head -1)
  [ -n "$WORLD" ] && break
done
WORLD=${WORLD:-${APM_WORLD%.sdf}}
echo "world name: $WORLD"

export GZ_SIM_SYSTEM_PLUGIN_PATH="$APM_GZ_DIR/build:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export GZ_SIM_RESOURCE_PATH="$REPO/sim/ardupilot/models:$REPO/sim/ardupilot/worlds:$APM_GZ_DIR/models:$APM_GZ_DIR/worlds:${GZ_SIM_RESOURCE_PATH:-}"

echo "===== ARDUPILOT EPISODE $(date) world=$APM_WORLD steps=$APM_STEPS ====="
mkdir -p "$RUN_DIR"
rm -f "$RUN_DIR"/START "$RUN_DIR"/START.done "$RUN_DIR"/actions.csv \
      "$RUN_DIR"/frames.npy "$RUN_DIR"/frame_ts.csv

echo "--- cleanup ---"
for _ in 1 2 3; do
  pkill -9 -f 'gz sim' 2>/dev/null
  pkill -9 -f 'arducopter' 2>/dev/null
  pkill -9 -f 'sim_vehicle.py' 2>/dev/null
  pkill -9 -f 'gz_frame_capture' 2>/dev/null
  sleep 2
  n=$(pgrep -fc 'gz sim|arducopter' 2>/dev/null || true)
  [ "${n:-0}" = "0" ] && break
done

echo "--- gazebo ---"
: > "$GZLOG"
setsid nohup gz sim -v2 -r -s "$APM_WORLD" > "$GZLOG" 2>&1 < /dev/null &
wready=0
for i in $(seq 1 180); do
  if timeout 10 gz topic -l 2>/dev/null | grep -q "^/world/$WORLD/clock"; then
    echo "WORLD_READY ${i}s"; wready=1; break
  fi
  sleep 1
done
[ "$wready" = "1" ] || { echo WORLD_LOAD_FAIL; tail -5 "$GZLOG"; exit 1; }

echo "--- sitl ---"
: > "$SITLLOG"
cd "$APM_DIR" || exit 1
setsid nohup python3 Tools/autotest/sim_vehicle.py \
  -v "$APM_VEHICLE" -f "$APM_FRAME" --model JSON \
  --add-param-file="$REPO/sim/ardupilot/params/sitl_relaxed.parm" \
  --no-mavproxy --no-rebuild > "$SITLLOG" 2>&1 < /dev/null &

echo "--- wait for camera topic + SITL ---"
TOPIC=""
for i in $(seq 1 90); do
  TOPIC=$(timeout 15 gz topic -l 2>/dev/null | grep '/camera/image$' | head -1)
  [ -n "$TOPIC" ] && { echo "TOPIC ${i}s: $TOPIC"; break; }
  sleep 1
done
[ -n "$TOPIC" ] || { echo NO_CAMERA_TOPIC; gz topic -l 2>/dev/null | head -12; exit 1; }

for i in $(seq 1 60); do
  grep -q "ArduPilot Ready" "$SITLLOG" 2>/dev/null && { echo "SITL_READY ${i}s"; break; }
  sleep 1
done

echo "--- capture ---"
cd "$REPO" || exit 1
setsid nohup "$PY" scripts/gz_frame_capture.py \
  --topic "$TOPIC" --frames "$APM_STEPS" --size "$SIZE" \
  --trigger "$RUN_DIR/START" --done "$RUN_DIR/START.done" \
  --outdir "$RUN_DIR" > "$CAPLOG" 2>&1 < /dev/null &
CAP_PID=$!
sleep 3

echo "--- flight ($APM_STEPS steps) ---"
timeout -k 10 300 "$PY" scripts/apm_flight_logger.py \
  --steps "$APM_STEPS" --trigger "$RUN_DIR/START" --out "$RUN_DIR/actions.csv" \
  2>&1 | grep -aE "connecting|connected|GUIDED|armed|takeoff|final|trigger|landing|saved|AP: Arm|refused" | tail -12

for i in $(seq 1 120); do
  kill -0 "$CAP_PID" 2>/dev/null || { echo "capture exited ${i}s"; break; }
  sleep 1
done
kill -TERM "$CAP_PID" 2>/dev/null
sleep 4
tail -2 "$CAPLOG"

sleep 2
pkill -9 -f 'arducopter' 2>/dev/null
pkill -9 -f 'sim_vehicle.py' 2>/dev/null
pkill -9 -f 'gz sim' 2>/dev/null

echo "--- artefacts ---"
ls -la "$RUN_DIR" 2>/dev/null | grep -E "frames.npy|actions.csv|frame_ts" || true
echo "===== DONE $(date) ====="
