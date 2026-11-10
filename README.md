# drone-vla — راهنمای کامل وضعیت و راه‌اندازی

> آخرین به‌روزرسانی: 2026-09-30 | سخت‌افزار: RTX 4060 Laptop 8GB / 15GB RAM / i7-12700H / Ubuntu 22.04
> درایور فعلی: **NVIDIA 580.178.04** (دان‌گرید از 595 — برای رندر Isaac لازم بود)
Isaac Lab 2.1.0 نصب شد روی `env_isaac45` (پوشهٔ `IsaacLab-2.1/`) — لاگر rotor فوتورئال تست شد ✓

---

## ۱) نقشهٔ کلی — چی داریم و چرا

هدف پروژه: **پرواز پهپاد در شبیه‌ساز → ضبط (ویدیو + اکشن + دستور) → fine-tune مدل World-Action (FLUX 3 Action) → پهپادی که به دستور گوش می‌ده**

سه محیط پایتون جدا داریم (با هم قاطی نشن):

| محیط | پایتون | محتوا | کاربرد |
|---|---|---|---|
| `env_isaac45` | 3.10 | Isaac Sim **4.5** + Isaac Lab 2.1 + torch 2.5.1 cu124 (+ rsl_rl) | **رندر، دوربین و RL** (بعد از دانگرید درایور 580 زنده شد!) |
| `env_flight` | 3.11 | MAVSDK 3.17 + pymavlink + pyulog + pandas + lerobot + huggingface_hub | پرواز PX4، لاگ‌گیری، دیتاست |
| `flux-action/.venv` | 3.12 | ریپو BFL + torch 2.10 cu128 + natten + bnb | مدل FLUX 3 Action |

به‌علاوه: `PX4-Autopilot/` (SITL بیلدشده + Gazebo Harmonic 8.15) و `models/flux-3-action-base/` (25.4GB وزن‌های مدل).

---

## ۲) سناریوی اصلی: تولید دیتای پهپاد (پایدار — Gazebo)

### ترتیب اجرا (۳ ترمینال):

```bash
# ── ترمینال ۱: شبیه‌ساز ──
cd ~/drone-vla/PX4-Autopilot
HEADLESS=1 make px4_sitl gz_x500_mono_cam
# بدون HEADLESS=1 → پنجره گرافیکی
# بوت سالم = پیام "Startup script returned successfully"
# ⚠ هر پرواز باید روی سیمِ تازه‌بوت باشد (بعد از فرود، آرم خراب می‌شود)

# ── ترمینال ۲: فیلم‌بردار (پایتون سیستمی!) ──
/usr/bin/python3 ~/drone-vla/scripts/gz_frame_capture.py \
  --topic /world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image \
  --frames 5000 \
  --trigger ~/drone-vla/outputs/data/rotor_gz/run1/START \
  --done ~/drone-vla/outputs/data/rotor_gz/run1/START.done \
  --outdir ~/drone-vla/outputs/data/rotor_gz/run1
# می‌ماند تا تریگر بخورد

# ── ترمینال ۳: خلبان ──
cd ~/drone-vla
env_flight/bin/python scripts/gz_flight_logger.py --steps 300 \
  --trigger outputs/data/rotor_gz/run1/START \
  --out outputs/data/rotor_gz/run1/actions.csv
# آرم → START (فیلم‌بردار شروع) → ۲۰ ثانیه پرواز → فرود
```

### سرهم‌کردن به اپیزود استاندارد:

```bash
env_flight/bin/python scripts/merge_rotor_episode.py \
  --frames-dir outputs/data/rotor_gz/run1 \
  --actions outputs/data/rotor_gz/run1/actions.csv \
  --steps 300 --task "take off, fly forward, and hover" \
  --out outputs/data/rotor_gz/ep_0001.npz
```

**قرارداد اپیزود (فرمت `rotor` مدل FLUX 3 Action):**
- `frames (T,256,256,3) uint8` — ویدیوی دوربین روی پهپاد
- `action (T,4) float32` در [-1,1] — `[fwd, lat, up, yaw_rate]` @ 15Hz
- `index.json` — لیست اپیزودها با برچسب `task` (دستور متنی؛ بدون پیشوند)
- ⚠ برچسب باید **واقعاً** توصیف رفتار خلبان باشد

### نقش هر فایل:

| فایل | نقش |
|---|---|
| `scripts/gz_frame_capture.py` | سابسکرایب تاپیک تصویر Gazebo (پروتکل trigger/done) |
| `scripts/gz_flight_logger.py` | خلبان اسکریپتی + لاگ اکشن‌ها (MAVSDK offboard velocity) |
| `scripts/merge_rotor_episode.py` | همترازی timestamp → NPZ نهایی |
| `scripts/flight_log.py` | لاگر سادهٔ تله‌متری (بدون دوربین) |
| `scripts/isaac_quad_logger.py` | لاگ اپیزود RL در Isaac Lab (obs+action، بدون دوربین) |
| `scripts/isaac_rotor_logger.py` | لاگر rotor فوتورئال روی Isaac — **✓ تست شد** (با `env_isaac45`) |
| `scripts/render_sensor45.py` | تست رندر دوربین Isaac 4.5 (تأییدشده ✓) |
| `scripts/flux3_quant_test.py` | تست اجرای FLUX 7B با nf4/int8 روی 8GB |

---

## ۳) سناریوی FLUX 3 Action (مدل — اجرا روی همین لپ‌تاپ)

```bash
cd ~/drone-vla/flux-action
.venv/bin/python ~/drone-vla/scripts/flux3_quant_test.py --quant nf4
```

- وضعیت: **nf4 (4-bit) کار می‌کند** — peak 6.80GB، خروجی `(1,32,4)` اکشن، ~1.1s هر پلن
- int8 و bf16 روی 8GB جا نمی‌شوند (تست شد)
- مدل base هنوز پهپاد بلد نیست — با fine-tune روی دیتای بالا یاد می‌گیرد (GPU کرایه‌ای: 4090، recipe خود BFL در `flux-action` docs)
- درایور 580 = CUDA 12.8 → همهٔ venvها سازگارند

---

## ۴) سناریوی Isaac Sim

```bash
# رندر/دوربین (4.5 — تست‌شده ✓):
env_isaac45/bin/python scripts/render_sensor45.py
# خروجی: outputs/isaac45_sensor_frame.png

# لاگ اپیزود RL quad:
env_isaac45/bin/python scripts/isaac_quad_logger.py --headless --num_envs 64 --steps 5000

# GUI خالی:
env_isaac45/bin/isaacsim
```

> ⚠ `env_isaaclab` (Isaac Sim 5.1) در پاک‌سازی 2026-09-30 حذف شد — رندر 5.1 روی این سیستم هرگز کار نکرد و همه‌چیز به 4.5 + Isaac Lab 2.1 منتقل شده بود. اسکریپت `isaac_quad_logger.py` روی 4.5 تست نشده.

---

## ۵) تله‌های مهم (خلاصه — جزئیات در NOTES.md)

1. **آرم PX4 فقط روی بوت تازه** — بعد از فرود، ری‌استارت سیم
2. پایتون‌ها را قاطی نکن (جدول بالا)
3. پروسه‌های بلند: `setsid nohup ... &`؛ kill با PID نه pkill
4. RAM 15GB: کار سنگین GPU (رندر Isaac) → اول swap 32GB بساز/فعال کن:
   `sudo fallocate -l 32G /swapfile2 && sudo chmod 600 /swapfile2 && sudo mkswap /swapfile2 && sudo swapon /swapfile2`
   (فایل swap در پاک‌سازی 2026-09-30 حذف شد؛ با ریبوت هم سواپ می‌پرد — چک کن با `swapon --show`)
5. Isaac 4.5 همیشه این آرگ‌ها را بخواهد:
   `--/rtx-transient/scenedb/maxInstancesLimit=4000000 --/rtx-transient/scenedb/forceMaxTLASInstancesLimit=true --/renderer/activeGpu=0 --/renderer/multiGpu/enabled=False`
6. بعد از Isaac 4.5 اگر torch را آپدیت کنی می‌شکند — روی 2.5.1 بمان
7. rollback درایور: `sudo apt install nvidia-driver-595-open` + reboot

---

## ۶) نقشهٔ راه باقی‌مانده

- [ ] صحنهٔ دارای landmark (در Gazebo یا Isaac — هر دو آماده) + خلبان waypoint-follower
- [ ] جمع‌آوری ~۳۰۰-۸۰۰ اپیزود با دستورهای متنوع → index.json
- [ ] fine-tune FLUX 3 Action روی GPU کرایه‌ای (RunPod 4090 ~$0.4/h)
- [ ] حلقهٔ بسته: policy آموزش‌دیده جای خلبان در شبیه‌ساز
- [x] (اختیاری) Isaac Lab 2.1 روی Isaac 4.5 → لاگر روتور فوتورئال ✓ نصب و تست شد

---

## ۷) sat2aerial — ناوبری بدون GPS (ماهواره ← هوایی)

بستهٔ `sat2aerial/` برای سنجش ایدهٔ «تولید تصویر هوایی از ماهواره و بعد matching» اضافه شد.
متریک اصلی: تعداد inlierهای تطبیق بین تصویر تولیدشده و فریم واقعی پهپاد، در مقایسه با baseline ماهوارهٔ خام؛
به‌علاوهٔ `cons_inliers` برای تشخیص جعل هندسه (حفظ شدن یا نشدن ساختمان/جاده).

```bash
env_flight/bin/python -m sat2aerial selfcheck --out /tmp/sat2aerial_selfcheck
env_flight/bin/python -m sat2aerial demo --out outputs/sat2aerial/demo --sites 4 --clean
env_flight/bin/python -m unittest discover -s tests -v
```

بک‌اندها: `dummy` (آفلاین)، `comfy` (لوکال int8 روی 8GB — **تست‌شده**)، `bfl` (API با `BFL_API_KEY`)،
`flux3` (diffusers؛ روی 8GB عملاً محدود). برای نتیجهٔ معتبر `--seeds 3` بده: مدل به seed حساس است و یک seed
هندسه را حفظ می‌کند و seed دیگر شهر را جابه‌جا می‌سازد. جزئیات: `sat2aerial/README.md` و بخش ۱۲ در `NOTES.md`.

### ۷.۱) چطور بدون GPS تست کنیم — تصویر تست از کجا می‌آید؟

مشکل شبیه‌سازی این است که از دنیای دلخواه (AirSim/Gazebo) orthophoto وجود ندارد، پس هر عدد matching
فقط «فاصلهٔ sim2real» را می‌سنجد. دو دستور این را حل می‌کنند:

```bash
# L1: رندر 2.5D روی همان نقشه‌ای که localizer با آن تطبیق می‌دهد (ground truth دقیق)
env_flight/bin/python -m sat2aerial flythrough --ortho site.tif --out outputs/orthofly/site \
  --frames 120 --alt 80 --path snake --length 400 --ref-jitter 40 \
  --pairs-dir outputs/orthofly/site/pairs

# خطای متری + نرخ رد شدن
env_flight/bin/python -m sat2aerial localize --run outputs/orthofly/site --match-mode gray

# L0: پرواز واقعی دوربین‌دار (UAV-VisLoc v1/v2) → همان فرمت، پس همان ارزیاب
env_flight/bin/python scripts/import_uav_visloc.py --src data/uav_visloc/raw --sites 07 \
  --out data/uav_visloc/processed/07
env_flight/bin/python -m sat2aerial localize --run data/uav_visloc/processed/07 --matcher ncc
```

نتیجهٔ دموی مصنوعی (۳۰ فریم، ۳۰۰ متر): `success 0.77`، میانهٔ خطا در فریم‌های پذیرفته‌شده **۰.۲۵ m**،
هیچ فریم «قبول اما اشتباه» بالای ۵ متر. نکتهٔ مهم: `--match-mode gray` برای فریم nadir الزامی است
(`edges` روی نقشه، inlierهای هم‌خط روی جاده می‌دهد و homography را در راستای جاده آزاد می‌گذارد).
جدول کامل و محدودیت‌ها: `sat2aerial/README.md`.

روی تصویر **واقعی** پهپاد همین منطق برعکس می‌شود. دو نسخهٔ دیتاست دو رفتار متفاوت دارند:

| نسخه | تصویر | matcher | نرخ fix | میانهٔ خطا |
|---|---|---|---|---|
| v2 (چیپ‌های ۱۳۵ px) | ۱۴۱۴ px، شمال-بالا | `ncc` | ۰.۸۷ | ۱۱.۸ m (بدون بایاس **۲.۷۶ m**) |
| v1 (همان zip ۱۷ گیگی) | ۳۰۰۰×۲۰۰۰، چرخیده ~۹۰° | `ncc --rotate-search 360` | ۰.۸۳ | **۱.۷۰ m** |
| v1 | همان | `features/edges` (SIFT) | ۰.۰۳ | — |

⚠ این عدد فقط برای فلایت **۰۷** است و فقط **طول‌محور** است: کریدور ۰۷ دور خودِ خط پرواز بریده
شده (۴۹ m ارتفاع کریدور در برابر ۱۸۹ m ارتفاع فریم)، پس محور عرضی by construction صفر است و
آن ۱ متر، دقت ۱-بعدیِ واقعی ولی محدود است. سوییپ کامل ۱۱ فلایت: ۰۷ → ۱.۰۲ m؛ بقیه ۲۸ تا ۸۵ m.

**علت واقعی (۲۰۲۶-۱۱-۰۸):** تست «رتبهٔ حقیقت» نشان داد NCC در جای درست، در صدک ۴۲–۷۴ بین هزاران
آفست است — یعنی **مکان درست از یک مکان تصادفی قابل تفکیک نیست**. پس خطای ۲۸–۸۵ متری باگ matcher
نیست: تصویر مرجع و مسیر پهپاد co-registration ندارند (موزاییک Google Earth در برابر GPS خام
WGS84). فاصلهٔ زمانی هم توضیح‌دهنده نیست (۰۶ با ۱ ماه → ۲۸ m، ۰۷ با ۴.۹ سال → ۱ m). جزئیات و
اعداد FLUX در `STATUS_2026-11-06.md` §۹.۷.

در v1 دوربین ~۹۰ درجه نسبت به شمالِ نقشه چرخیده و کریدور ماهواره فقط ۴۳٪ ارتفاع فریم است؛ به همین
دلیل SIFT می‌شکند و همبستگی متراکم بدون جست‌وجوی چرخش هم. اسکریپت `scripts/plan_capture.py` هم
تعداد فریم یک پرواز ۱۰ کیلومتری را می‌دهد: ۱۰۰–۳۰۰ متر، ۲۰۰ m/s، هم‌پوشانی ۷۰–۸۰٪ →
**۱۳۲ تا ۵۹۰ فریم**. برای اجرای کل دیتاست با یک دستور:

```bash
env_flight/bin/python scripts/run_uav_visloc.py --src data/uav_visloc/raw_v1 \
  --out data/uav_visloc/processed_v1 --frame-stride 5 --rotate-search auto
```

### ۷.۲) L2 — دنیای Gazebo از همان نقشه (حلقهٔ بسته)

```bash
# زمین = خود orthophoto، مختصات ENU گره‌خورده به lat/lon، ساختمان‌های OSM
env_flight/bin/python scripts/make_geo_world.py --ortho site.tif --name site1 \
  --out outputs/geo --extent-m 800 --tile-m 200 --tile-px 1024 \
  --center-lat 47.397 --center-lon 8.546 --buildings b.geojson

source outputs/geo/site1/env.sh
gz sim -r outputs/geo/site1/worlds/site1.sdf

# تأیید اینکه دوربین رو به پایینِ شبیه‌ساز با همان نقشه تطبیق می‌دهد
env_flight/bin/python scripts/check_geo_world.py --run outputs/geo/site1 --alt 260
```

`world = ENU` (`x=East, y=North, z=Up`) و `(0,0)` همان `--center-lat/lon` است، پس خروجی
`localize`/`poses.csv` مستقیماً به `VISION_POSITION_ESTIMATE` وصل می‌شود.

**تأییدشده روی Gazebo Harmonic 8.15 (headless, llvmpipe):** ۵۰۲ inlier، coverage ۰.۹۰،
scale `0.99971`، rotation `-0.004°`، جابه‌جایی `0.41 m`. یعنی بافت زمین نه flip می‌خواهد و نه
محورها جابه‌جا شده‌اند؛ با `heading=0` تصویر دوربین شمال‌بالا است.

---

## ۸) NavLab — رابط وب فارسی برای تست ناوبری

پلتفرم MVP اضافه شد: مأموریت (lat/lon)، آپلود GeoTIFF، تولید تصویر هوایی، تطبیق، نتایج، و
SITL (PX4 + Gazebo) با تله‌متری و «پرواز تا نقطهٔ پایان».

```bash
env_flight/bin/python -m navlab
# http://127.0.0.1:8080
```

- UI فارسی/RTL، صفحات: داشبورد، مأموریت، آزمایش سریع، SITL، سخت‌افزار، تنظیمات.
- SITL: دوربین رو به پایین Gazebo زنده در UI، waypoint + geofence نرم‌افزاری، پرواز مسیر و land.
- BOM اکسل: `outputs/navlab/hardware_bom.xlsx` (دانلود از `/hardware`).
- بک‌اند تولید: `dummy` (آفلاین)، `comfy` (لوکال)، `bfl` (API). سرویس 4090 بعداً با همان قرارداد `/v1/generate`.
- آستانهٔ موفقیت تطبیق: inliers ≥ 15، ratio ≥ 0.15 و **coverage ≥ 0.03**.
- SITL تست شد: اتصال MAVSDK، takeoff، ۳ waypoint، geofence و land. HITL فاز بعد (Gazebo Classic یا SIH).
- جزئیات: `navlab/README.md`؛ تست‌ها: ۱۶/۱۶.

### مرور نتایج مکان‌یابی دیداری

برای دیدن فریم‌های پذیرفته‌شده، پرخطا و ردشده در آزمایش‌های محلی:

```bash
cd /home/arman/drone-vla
env_flight/bin/python scripts/localization_review.py --port 8765
```

در مرورگر همین ماشین `http://127.0.0.1:8765/` را باز کنید. دکمهٔ «نمایش نقاط تطبیق» برای مدل‌های پشتیبانی‌شده نقاط را دوباره محاسبه می‌کند؛ نتیجهٔ تصمیم قبلی تغییر نمی‌کند. گزارش وضعیت و محدودیت‌ها در [docs/GPS_DENIED_STATUS_FA.md](docs/GPS_DENIED_STATUS_FA.md) و خروجی‌های عددی گزیده در [docs/evidence/visual_localization](docs/evidence/visual_localization/README.md) هستند.
