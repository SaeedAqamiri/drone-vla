#!/bin/bash
# ArduPilot + Gazebo episode using the PX4 x500 airframe.
#
#   scripts/run_apm_x500_episode.sh
#   APM_STEPS=300 RUN_DIR=... scripts/run_apm_x500_episode.sh
#
# Data flow (this is the whole point of the x500 variant):
#
#   Gazebo x500 IMU/GPS/baro --(ArduPilotPlugin, JSON FDM :9002)--> ArduPilot SITL
#   ArduPilot SERVO_OUTPUT_RAW --(MAVLink tcp:5762)--> apm_motor_bridge.py
#        --> /x500/command/motor_speed (gz.msgs.Actuators) --> thrust
#   ArduPilot <--(MAVLink tcp:5760)--> apm_flight_logger.py (our pilot)
#
# Two MAVLink clients are needed, hence SERIAL1 (5762) for the bridge and
# SERIAL0 (5760) for the pilot.

set -uo pipefail

REPO=${REPO:-/mnt/e/drone-vla}
APM_DIR=${APM_DIR:-/home/drone/ardupilot}
APM_GZ_DIR=${APM_GZ_DIR:-/home/drone/ardupilot_gazebo}
PX4_MODELS=${PX4_MODELS:-/home/drone/PX4-Autopilot/Tools/simulation/gz/models}
PX4_PLUGINS=${PX4_PLUGINS:-/home/drone/PX4-Autopilot/build/px4_sitl_default/src/modules/simulation/gz_plugins}

APM_WORLD=${APM_WORLD:-field_apm_x500.sdf}
APM_STEPS=${APM_STEPS:-300}
SIZE=${SIZE:-256}
MODEL_NAME=${MODEL_NAME:-x500}
RUN_DIR=${RUN_DIR:-$REPO/outputs/data/rotor_apm/run_x500}

PY=/usr/bin/python3
GZLOG=/tmp/apm_x500_gz.log
SITLLOG=/tmp/apm_x500_sitl.log
BRIDGELOG=/tmp/apm_x500_bridge.log
CAPLOG=/tmp/apm_x500_capture.log

WORLD=$(sed -n "s/.*<world name=['\"]\([^'\"]*\)['\"].*/\1/p" \
  "$REPO/sim/ardupilot/worlds/$APM_WORLD" | head -1)
WORLD=${WORLD:-field}

export GZ_SIM_SYSTEM_PLUGIN_PATH="$APM_GZ_DIR/build:$PX4_PLUGINS:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export GZ_SIM_RESOURCE_PATH="$REPO/sim/ardupilot/models:$REPO/sim/ardupilot/worlds:$PX4_MODELS:$APM_GZ_DIR/models:$APM_GZ_DIR/worlds:${GZ_SIM_RESOURCE_PATH:-}"

echo "===== APM x500 EPISODE $(date) world=$WORLD model=$MODEL_NAME ====="
mkdir -p "$RUN_DIR"
rm -f "$RUN_DIR"/START "$RUN_DIR"/START.done "$RUN_DIR"/actions.csv \
      "$RUN_DIR"/frames.npy "$RUN_DIR"/frame_ts.csv

echo "--- cleanup ---"
for _ in 1 2 3; do
  pkill -9 -f 'gz si[m]' 2>/dev/null
  pkill -9 -f 'arducopte[r]' 2>/dev/null
  pkill -9 -f 'sim_vehicl[e]' 2>/dev/null
  pkill -9 -f 'apm_motor_bridg[e]' 2>/dev/null
  pkill -9 -f 'gz_frame_captur[e]' 2>/dev/null
  sleep 2
  n=$(pgrep -fc 'gz si[m]|arducopte[r]' 2>/dev/null || true)
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
  -v ArduCopter -f gazebo-iris --model JSON \
  --add-param-file="$REPO/sim/ardupilot/params/sitl_relaxed.parm" \
  --no-mavproxy --no-rebuild > "$SITLLOG" 2>&1 < /dev/null &

echo "--- wait for camera + SITL ---"
TOPIC=""
for i in $(seq 1 90); do
  TOPIC=$(timeout 15 gz topic -l 2>/dev/null | grep '/camera/image$' | head -1)
  [ -n "$TOPIC" ] && { echo "TOPIC ${i}s: $TOPIC"; break; }
  sleep 1
done
[ -n "$TOPIC" ] || { echo NO_CAMERA_TOPIC; exit 1; }

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
  --steps "$APM_STEPS" --motor-topic "$MODEL_NAME" \
  --trigger "$RUN_DIR/START" --out "$RUN_DIR/actions.csv" \
  2>&1 | tee /tmp/apm_flight.log \
  | grep -aE "connecting|connected|GUIDED|motor bridge|armed|stabilis|heading|takeoff|final|trigger|landing|saved|AP: Arm|refused" | tail -14

for i in $(seq 1 120); do
  kill -0 "$CAP_PID" 2>/dev/null || { echo "capture exited ${i}s"; break; }
  sleep 1
done
kill -TERM "$CAP_PID" 2>/dev/null
sleep 4
tail -2 "$CAPLOG"

sleep 2
pkill -9 -f 'arducopte[r]' 2>/dev/null
pkill -9 -f 'sim_vehicl[e]' 2>/dev/null
pkill -9 -f 'gz si[m]' 2>/dev/null

echo "--- artefacts ---"
ls -la "$RUN_DIR" 2>/dev/null | grep -E "frames.npy|actions.csv|frame_ts" || true
echo "===== DONE $(date) ====="
