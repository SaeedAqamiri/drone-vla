function initMissionMap(id, data) {
  const element = document.getElementById(id);
  if (!element || typeof L === "undefined") return;
  const map = L.map(id, { zoomControl: true, attributionControl: false });
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    opacity: 0.55,
  }).addTo(map);

  const points = (data.points && data.points.length >= 2) ? data.points : [data.start, data.end];
  const start = L.circleMarker(points[0], { radius: 7, color: "#36d399", fillOpacity: 0.9 }).addTo(map);
  const end = L.circleMarker(points[points.length - 1], { radius: 7, color: "#fb7185", fillOpacity: 0.9 }).addTo(map);
  start.bindTooltip("شروع");
  end.bindTooltip("پایان");
  points.slice(1, -1).forEach((point, index) => {
    L.circleMarker(point, { radius: 5, color: "#fbbf24", fillOpacity: 0.9 })
      .addTo(map)
      .bindTooltip(`waypoint ${index + 1}`);
  });
  L.polyline(points, { color: "#5ea0ff", weight: 3, dashArray: "6 6" }).addTo(map);

  if (data.polygon && data.polygon.length >= 3) {
    L.polygon(data.polygon, { color: "#5ea0ff", weight: 1, fillOpacity: 0.08 }).addTo(map);
  }

  if (data.overlay && data.overlay.url && data.overlay.bounds) {
    L.imageOverlay(data.overlay.url, data.overlay.bounds, { opacity: 0.8 }).addTo(map);
    map.fitBounds(data.overlay.bounds, { padding: [20, 20] });
  } else if (data.polygon && data.polygon.length >= 3) {
    map.fitBounds(data.polygon, { padding: [30, 30] });
  } else {
    map.fitBounds(points, { padding: [40, 40] });
  }
  window._missionMap = map;
}

function initSimMap(id) {
  const element = document.getElementById(id);
  if (!element || typeof L === "undefined") return;
  const center = [35.6892, 51.389];
  const map = L.map(id, { zoomControl: true, attributionControl: false }).setView(center, 15);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, opacity: 0.6 }).addTo(map);
  const marker = L.circleMarker(center, { radius: 7, color: "#5ea0ff", fillOpacity: 0.9 }).addTo(map);
  window._simMap = map;
  window._simMarker = marker;
}

function pollJob(jobId) {
  const statusEl = document.getElementById("job-status");
  const progressEl = document.getElementById("job-progress");
  const messageEl = document.getElementById("job-message");
  const countEl = document.getElementById("job-count");

  async function tick() {
    try {
      const response = await fetch(`/api/jobs/${jobId}`);
      const data = await response.json();
      if (statusEl) statusEl.textContent = data.status;
      if (messageEl) messageEl.textContent = data.message || "";
      if (countEl) countEl.textContent = `${data.progress} / ${data.total}`;
      if (progressEl) {
        const pct = data.total ? Math.round((data.progress / data.total) * 100) : 0;
        progressEl.style.width = `${pct}%`;
      }
      if (data.status === "done" || data.status === "error") {
        setTimeout(() => window.location.reload(), 700);
        return;
      }
    } catch (error) {
      if (messageEl) messageEl.textContent = "خطا در ارتباط با سرور";
    }
    setTimeout(tick, 1500);
  }
  tick();
}

function pollSim() {
  async function tick() {
    try {
      const response = await fetch("/api/sim/status");
      const data = await response.json();
      const running = document.getElementById("sim-status");
      const connected = document.getElementById("sim-connected");
      const pid = document.getElementById("sim-pid");
      const state = document.getElementById("sim-mission-state");
      if (running) running.textContent = data.running ? "روشن" : "خاموش";
      if (connected) connected.textContent = data.connected ? "متصل" : "قطع";
      if (pid) pid.textContent = data.pid || "—";
      if (state) state.textContent = data.mission_status || "—";
      const ev = data.ev || {};
      const evState = document.getElementById("sim-ev-state");
      const evError = document.getElementById("sim-ev-error");
      if (evState) evState.textContent = ev.active ? `active (${ev.source || "truth"})` : "off";
      if (evError) evError.textContent = ev.error || "—";
      const evMatch = document.getElementById("sim-ev-match");
      if (evMatch) {
        const last = ev.last || {};
        evMatch.textContent = last.matched
          ? `${last.source || ""} · inliers ${last.inliers ?? "—"} · kf ${last.keyframe ?? "—"}`
          : (last.source === "map" ? "no match" : "—");
      }
      const truthError = document.getElementById("sim-truth-error");
      if (truthError) {
        const last = ev.last || {};
        truthError.textContent = last.truth_error_m !== undefined
          ? `${last.truth_error_m} m`
          : "—";
      }

      const telemetry = data.telemetry || {};
      const lat = document.getElementById("sim-lat");
      const lon = document.getElementById("sim-lon");
      const alt = document.getElementById("sim-alt");
      const yaw = document.getElementById("sim-yaw");
      if (lat && telemetry.lat !== undefined) lat.textContent = Number(telemetry.lat).toFixed(6);
      if (lon && telemetry.lon !== undefined) lon.textContent = Number(telemetry.lon).toFixed(6);
      if (alt && telemetry.alt_rel !== undefined) alt.textContent = `${Number(telemetry.alt_rel).toFixed(1)} m`;
      if (yaw && telemetry.yaw !== undefined) yaw.textContent = `${Number(telemetry.yaw).toFixed(0)}°`;

      if (window._simMarker && telemetry.lat !== undefined && telemetry.lon !== undefined) {
        const point = [telemetry.lat, telemetry.lon];
        window._simMarker.setLatLng(point);
        if (window._simMap) window._simMap.panTo(point);
      }

      const camera = data.camera || {};
      const cameraBadge = document.getElementById("sim-camera-badge");
      const cameraImage = document.getElementById("sim-camera");
      const cameraNote = document.getElementById("sim-camera-note");
      if (cameraBadge) {
        cameraBadge.textContent = camera.available ? "live" : (camera.running ? "waiting" : "offline");
      }
      if (cameraImage && data.running && camera.available) {
        cameraImage.style.visibility = "visible";
        cameraImage.src = `/api/sim/camera.jpg?t=${Date.now()}`;
      }
      if (cameraNote) {
        cameraNote.textContent = camera.available
          ? `آخرین فریم ${camera.age_s}s پیش · ${camera.meta && camera.meta.width ? camera.meta.width + "×" + camera.meta.height : ""}`
          : "هنوز فریمی از دوربین نرسیده است.";
      }

      const map = data.map || {};
      const mapStatus = document.getElementById("map-status");
      const mapFrames = document.getElementById("map-frames");
      const mapDir = document.getElementById("map-dir");
      if (mapStatus) mapStatus.textContent = map.running ? `در حال ثبت (${map.elapsed_s || 0}s)` : "متوقف";
      if (mapFrames) mapFrames.textContent = map.frames ?? 0;
      if (mapDir) mapDir.textContent = map.out_dir || "—";

      const log = document.getElementById("sim-log");
      if (log && data.log) log.textContent = data.log.join("\n");
      const badge = document.getElementById("sim-log-badge");
      if (badge) badge.textContent = data.running ? "live" : "offline";
    } catch (error) {
      /* retry */
    }
    setTimeout(tick, 2000);
  }
  tick();
}

async function startSim() {
  const select = document.getElementById("sim-mission");
  if (!select) return;
  const body = new FormData();
  body.append("mission_id", select.value);
  const denied = document.getElementById("sim-gps-denied");
  if (denied && denied.checked) {
    body.append("gps_denied", "1");
    const source = document.getElementById("sim-ev-source");
    const mapDir = document.getElementById("sim-map-dir");
    const gpsDisable = document.getElementById("sim-gps-disable");
    const initPrior = document.getElementById("sim-init-prior");
    if (source) body.append("source", source.value);
    if (mapDir) body.append("map_dir", mapDir.value);
    body.append("disable_gps", gpsDisable && gpsDisable.checked ? "1" : "0");
    body.append("init_prior", initPrior && initPrior.checked ? "1" : "0");
  }
  const response = await fetch("/api/sim/start", { method: "POST", body });
  const data = await response.json();
  if (!data.ok) alert(data.message || "شروع SITL ناموفق بود");
  setTimeout(() => window.location.reload(), 500);
}

async function stopSim() {
  await fetch("/api/sim/stop", { method: "POST" });
  setTimeout(() => window.location.reload(), 500);
}

async function startMapRecord() {
  const select = document.getElementById("sim-mission");
  if (!select) return;
  const body = new FormData();
  body.append("mission_id", select.value);
  body.append("altitude", document.getElementById("map-altitude")?.value || "30");
  body.append("duration", document.getElementById("map-duration")?.value || "120");
  const response = await fetch("/api/sim/map/start", { method: "POST", body });
  const data = await response.json();
  alert(data.message || "شروع ثبت نقشه");
}

async function stopMapRecord() {
  await fetch("/api/sim/map/stop", { method: "POST" });
  alert("ثبت نقشه متوقف شد");
}

async function benchOffboard() {
  const response = await fetch("/api/sim/offboard/bench", { method: "POST" });
  const data = await response.json();
  alert(data.message || "bench offboard");
}

async function disableGps() {
  const response = await fetch("/api/sim/gps/disable", { method: "POST" });
  const data = await response.json();
  alert(data.message || "دستور قطع GPS ارسال شد");
}

async function flySim() {
  const select = document.getElementById("sim-mission");
  if (!select) return;
  const body = new FormData();
  body.append("mission_id", select.value);
  const response = await fetch("/api/sim/fly", { method: "POST", body });
  const data = await response.json();
  alert(data.message || "دستور پرواز ارسال شد");
}
