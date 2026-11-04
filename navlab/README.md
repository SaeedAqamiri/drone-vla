# NavLab

پلتفرم محلی آزمایش ناوبری بدون GPS با رابط وب فارسی/RTL.

> راهنمای کامل آزمایشگاه مکانیابی تصویری (SGLang، deep matcherها، tile store، ویدیو و benchmark) در [`LAB_README.md`](../LAB_README.md).

## اجرا

```bash
cd ~/drone-vla
env_flight/bin/python -m navlab
# http://127.0.0.1:8080
```

اگر وابستگی‌ها نبودند:

```bash
uv pip install --python env_flight/bin/python \
  fastapi "uvicorn[standard]" sqlmodel pyproj shapely python-multipart aiofiles
```

## صفحات

| صفحه | کار |
|---|---|
| `/` | ساخت مأموریت، وضعیت سرویس‌ها |
| `/missions/<id>` | نقشه، waypoint/geofence، آپلود ماهواره/فریم، تولید، تطبیق، Jobها و Results |
| `/quick` | آزمایش سریع: یک تایل ماهواره + یک فریم واقعی |
| `/sim` | شروع/توقف SITL، تله‌متری، دوربین زنده، پرواز مسیر |
| `/hardware` | BOM سخت‌افزار، دانلود Excel و ایستگاه 4090 |
| `/settings` | وضعیت ComfyUI/BFL و قرارداد سرویس تولید |

## جریان کاری

1. مأموریت را با lat/lon شروع و پایان بساز (ارتفاع، سرعت، عرض کوریدور). نقاط میانی را در textarea هر خط `lat,lon` وارد کن.
2. نقشهٔ ماهواره‌ای (GeoTIFF بهتر است) را آپلود کن. اگر CRS داشته باشد، کوریدور و GSD محاسبه می‌شود.
3. یک فریم هوایی واقعی آپلود کن (برای ارزیابی تطبیق).
4. تولید را اجرا کن: `dummy` (آفلاین)، `comfy` (لوکال) یا `bfl` (API).
5. تطبیق را اجرا کن و در `/runs/<id>` نتیجه را ببین.

آستانهٔ success: `inliers >= 15`، `inlier_ratio >= 0.15` و **`coverage >= 0.03`**.
coverage جلوی false positiveهای هم‌خط (رودخانه/جاده) را می‌گیرد.

## بک‌اندهای تولید

| بک‌اند | نیاز | وضعیت |
|---|---|---|
| `dummy` | هیچ | تست پایپ‌لاین |
| `sglang` | سرویس FLUX.3 Klein روی `172.16.20.117:30010` | **فعال/تست‌شده**؛ ~۲–۳s برای 768px، batch=2 |
| `comfy` | ComfyUI + مدل int8 | روی 8GB تست شد |
| `bfl` | `BFL_API_KEY` | آماده، بدون کلید |
| `internal` | سرویس 4090 (قرارداد `/v1/generate`) | فاز بعد |

قرارداد سرویس داخلی:

```http
POST /v1/generate   { image, prompt, size, seed, style_refs[] }
→ { job_id }

GET /v1/jobs/{id} → { status, image_url, seconds, cost }
GET /v1/health
```

## BOM اکسل

فایل آماده: `outputs/navlab/hardware_bom.xlsx` (دکمهٔ دانلود در `/hardware` هم هست).

- شیت‌ها: `Summary`، `Drone_BOM`، `GPU_Station`، `Available`، `Notes`
- قیمت‌ها دلاری‌اند؛ نرخ ارز را در `Summary!B10` وارد کن تا ستون‌های تومان خودکار محاسبه شوند.
- ساخت مجدد:
  ```bash
  env_flight/bin/python scripts/make_hardware_xlsx.py
  ```

## SITL

- PX4 + Gazebo Harmonic (`x500_mono_cam_down` + world `baylands`)، `HEADLESS=1`، home روی نقطهٔ شروع مأموریت.
- شروع/توقف از UI؛ تله‌متری از MAVSDK (`udpin://0.0.0.0:14540`).
- دوربین رو به پایین Gazebo (۲۵۶×۲۵۶) به‌صورت JPEG در `/api/sim/camera.jpg` استریم می‌شود.
- «پرواز مسیر»: arm → takeoff → goto هر waypoint → land. ارتفاع دمو حداکثر ۵۰m.
- geofence نرم‌افزاری: فاصله از polyline مسیر؛ در صورت عبور، فرود و خطا.
- لاگ PX4 در حافظه (ring buffer) نگه داشته می‌شود، نه روی دیسک.
- HITL در فاز بعد (Gazebo Classic یا SIH).

## آزمایشگاه مکانیابی تصویری (`navlab.lab`)

لایهٔ قابل‌تکرار برای benchmark کردن ایده‌های matching روی تایل‌های متریک ماهواره و سکانس فریم واقعی.

```bash
# ۱) استخراج فریم‌های نزدیک به عمود از UAV-VisLoc
env_flight/bin/python -m navlab.lab sequence \
  --root data/raw/uav_visloc/UAV_VisLoc_dataset --map 03 \
  --near-nadir 15 --max-frames 50 --out outputs/navlab/lab/uav03_queries.json

# ۲) ساخت tile store متریک north-up (GSD هدف یا ground-m)
env_flight/bin/python -m navlab.lab tile-store \
  --source data/raw/uav_visloc/UAV_VisLoc_dataset/03/satellite03.tif \
  --queries outputs/navlab/lab/uav03_queries.json \
  --tile-ground-m 700 --tile-px 768 --out outputs/navlab/lab/uav03_tiles

# ۳) تولید رفرنس هوایی (dummy برای تست harness، comfy/bfl برای واقعی)
env_flight/bin/python -m navlab.lab generate \
  --store outputs/navlab/lab/uav03_tiles --out outputs/navlab/lab/uav03_tiles/generated_refs \
  --backend dummy --view nadir --size 768x768 --altitude 466

# ۴) یک run یا یک benchmark ماتریسی
env_flight/bin/python -m navlab.lab run \
  --store outputs/navlab/lab/uav03_tiles \
  --queries outputs/navlab/lab/uav03_queries.json \
  --references outputs/navlab/lab/uav03_tiles/generated_refs/references_nadir.json \
  --matcher sift --mode gray --confirmation 2 --out outputs/navlab/lab/run_1
env_flight/bin/python -m navlab.lab bench \
  --config experiments/lab/uav03_bench.yaml --out outputs/navlab/lab/bench_uav03
```

- خروجی هر run: `summary.json`، `results.csv`، `frames.csv`، `report.md` و overlayهای per-run.
- `frames.csv` شامل `error_m`، `prior_error_m`، `latency_ms` و متریک‌های match (`inliers`, `coverage`, `scale`, ...) است.
- ورودی ویدیو: `env_flight/bin/python -m navlab.lab video --video flight.mp4 --telemetry telemetry.csv --fps 2 --out ...` (بدون telemetry، `has_truth=False` و فقط matching/prior سنجیده می‌شود).
- matcherهای کلاسیک: `sift`, `orb`, `akaze`. matcherهای deep: `loftr`, `lightglue`, `superpoint_lightglue`.
  `kornia==0.8.3` از کش `uv` نصب شد و وزن‌های LoFTR/DISK از mirror (Gitee/ghproxy) گرفته شدند.
  برای deep فعلاً `--max-dim 768 --scale-tolerance 1.0` بده.
- سرویس SGLang (FLUX.3 Klein) روی `172.16.20.117:30010`:
  ```bash
  FLUX3_USER=ava ssh -l "$FLUX3_USER" 172.16.20.117 'cd /home/ava/flux3-klein && bash api/sglang_start.sh'
  curl -s http://172.16.20.117:30010/v1/models
  ```
  اگر سرویس پایین بود، همین اسکریپت را روی سرور اجرا کن. آدرس در `NAVLAB_SGLANG_URL` قابل override است.
  اتصال مستقیم: `curl http://172.16.20.117:30010/v1/models` و `POST /v1/images/edits`؛ نمونهٔ کامل curl/Python در [`LAB_README.md`](../LAB_README.md#sglang--flux3-klein-توصیهشده).
- نتیجهٔ benchmark روی صحنهٔ مصنوعی (۱۰ فریم، reference تولیدشده با SGLang):
  `sift/orb/akaze/loftr/lightglue` در mode `gray` همگی **۱۰/۱۰ fix** با `precision=1.0` و `p95 ≈ ۹.۶–۹.۹m`؛
  `edges` برای تصاویر تولیدشده ضعیف است. یعنی در این صحنه، رفرنس تولیدشده هم‌ارز ماهوارهٔ خام است.
- نکتهٔ مقیاس: `tile_ground_m` را نزدیک footprint واقعی دوربین انتخاب کن؛ در غیر این صورت `scale` gate ممکن است match درست را رد کند.
- JP2های چرخیده حالا با `local_crs` به north-up و GSD درست warp می‌شوند؛ `inspect_image` هم `gsd_xy_m` و `rotated` را برمی‌گرداند.

## محدودیت‌های فعلی

- بدون احراز هویت؛ فقط localhost/LAN قابل اعتماد.
- آپلود و تولید serialized است؛ اجرای هم‌زمان ComfyUI و SITL روی ۱۵GB RAM توصیه نمی‌شود.
- دوربین واقعی/HITL هنوز وصل نشده؛ فقط دوربین Gazebo (۲۵۶×۲۵۶).
- نقشهٔ آنلاین (XYZ با API key) هنوز پیاده نشده؛ فعلاً آپلود.
- geofence نرم‌افزاری است، نه پارامتر geofence خود PX4.
- HITL و GPS-denied با EV تزریقی فاز بعد است.
