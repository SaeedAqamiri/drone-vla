# NavLab

پلتفرم محلی آزمایش ناوبری بدون GPS با رابط وب فارسی/RTL.

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

## محدودیت‌های فعلی

- بدون احراز هویت؛ فقط localhost/LAN قابل اعتماد.
- آپلود و تولید serialized است؛ اجرای هم‌زمان ComfyUI و SITL روی ۱۵GB RAM توصیه نمی‌شود.
- دوربین واقعی/HITL هنوز وصل نشده؛ فقط دوربین Gazebo (۲۵۶×۲۵۶).
- نقشهٔ آنلاین (XYZ با API key) هنوز پیاده نشده؛ فعلاً آپلود.
- geofence نرم‌افزاری است، نه پارامتر geofence خود PX4.
- HITL و GPS-denied با EV تزریقی فاز بعد است.
