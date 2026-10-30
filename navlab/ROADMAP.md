# NavLab Roadmap

> وضعیت: 2026-10-28 — پلتفرم MVP + SITL + دوربین + waypoint/geofence آماده است.
> GPS-denied با EV از truth کار می‌کند؛ GPS-denied با map-match نیمه‌کاره است.

## وضعیت کلی

| بخش | وضعیت | یادداشت |
|---|---|---|
| UI فارسی/RTL + مأموریت/نقشه | ✅ | ۹ صفحه، FastAPI + HTMX + Leaflet |
| آپلود ماهواره/فریم + تولید (dummy/comfy/bfl) | ✅ | bfl بدون کلید |
| تطبیق + نتایج + آزمایش سریع | ✅ | SIFT + RANSAC + coverage |
| SITL (PX4 + Gazebo) start/stop/تله‌متری | ✅ | `x500_mono_cam_down` + `baylands` |
| دوربین رو به پایین زنده | ✅ | JPEG 256×256، ۵fps |
| waypoint + geofence نرم‌افزاری + پرواز/فرود | ✅ | ۴ نقطه، ۳۰m |
| GPS-denied با EV از Gazebo truth | ✅ | arm → takeoff → waypoint → land |
| GPS-denied با EV از map-match | ✅ | با baro height (HGT_REF=0 + reboot): کل مسیر، error ۰.۲–۲.۹m؛ فرود/overshoot کنترل باقی است |
| HITL / SIH | ⬜ | فاز بعد |
| سرویس داخلی 4090 | ⬜ | قرارداد API آماده است |
| Excel BOM | ✅ | `outputs/navlab/hardware_bom.xlsx` |

## کارهای باقی‌مانده

### A) Localization / Mapping (اولویت ۱)
- [x] جستجوی localizer بر اساس **ترتیب keyframe** (پنجرهٔ ±N حول آخرین kf).
- [x] prior متحرک + hold-last-EV + آستانهٔ coverage.
- [ ] پیش‌بینی سرعت + آستانهٔ adaptive.
- [ ] نقشه با چند ارتفاع / keyframe بیشتر / فیلتر کیفیت.
- [ ] UI ضبط نقشه (start/stop + تعداد فریم + نقشهٔ پوشش).
- [x] ارزیابی آفلاین localizer: ۹۸.۹٪ match، mean≈۰.۶m، p95≈۱.۸m، latency≈۱۷ms.
- [x] نمایش خطای EV نسبت به truth (`truth_error_m`) و تست staged با GPS روشن.
- [ ] نمایش «تخمینی vs واقعی» و معیار خطا در UI.
- [ ] سناریوی از دست دادن GPS در میانهٔ پرواز + بازیابی.

### B) Control / Failsafe (اولویت ۱)
- [ ] offboard local NED پیاده شد ولی به‌خاطر fly-away در SITL **گیت شده** (`use_offboard=False`) تا bench-test.
- [ ] نگه‌داشتن آخرین EV هنگام نبود match (جلوگیری از blind land).
- [x] **EKF2_HGT_REF/baro**: `HGT_REF=0` قبل از بوت + restart؛ EV افقی؛ GPS off تا پایان مسیر تست شد.
- [ ] failsafe/RTL/land دقیق + geofence واقعی PX4؛ overshoot/attitude هنگام فرود.
- [ ] تست پرواز مسیر با باد/نویز/قطع EV.

### C) Fusion (اولویت ۲)
- [ ] EKF/UKF برای IMU + baro + optical flow + map-match.
- [ ] covariance و outlier gating برای map-match.
- [ ] همگام‌سازی زمان دوربین/MAVLink/IMU.
- [ ] مدیریت قطعی match و dead-reckoning.

### D) سخت‌افزار و HITL (اولویت ۲)
- [ ] Pixhawk 6C + Raspberry Pi 5/4 + دوربین global shutter.
- [ ] کالیبراسیون intrinsics/extrinsics + زمان‌بندی.
- [ ] HITL با Gazebo Classic یا SIH؛ firmware با `pwm_out_sim`.
- [ ] تغذیه/حرارت/لرزش و تست bench.

### E) سرویس تولید 4090 (اولویت ۲)
- [ ] سرویس `/v1/generate` + `/v1/jobs/{id}` + `/v1/health`.
- [ ] صف، cache، auth، نسخهٔ مدل، quality gate.
- [ ] اتصال NavLab با بک‌اند `internal`.
- [ ] تولید نقشهٔ هوایی برای مسیر واقعی + ارزیابی fidelity.

### F) داده و ارزیابی (اولویت ۲)
- [ ] دیتاست واقعی near-nadir با جفت‌های هم‌مقیاس.
- [ ] baseline چهارم: semantic/edge matching.
- [ ] benchmark چند seed / چند مسیر + گزارش خودکار.
- [ ] لاگ ULog + تحلیل drift/خطا.

### G) UI / عملیات (اولویت ۳)
- [ ] صفحهٔ نقشه/ثبت نقشه.
- [ ] احراز هویت و محدودسازی LAN.
- [ ] export گزارش/ویدیو/لاگ.
- [ ] نمایش سلامت EKF و innovationها.

### H) ایمنی و تست میدانی (اولویت ۳)
- [ ] props-off، kill switch، geofence، failsafe.
- [ ] مجوز/ایمنی پرواز.
- [ ] تست میدانی با مسیر کوتاه و سپس بلند.

## ۶ قدم بعدی (پیشنهاد اجرا)

1. **کنترل/فرود**: رفع overshoot و attitude هنگام land (bench offboard / velocity control).
2. **Bench-test offboard** (setpoint ثابت، بدون پرواز) قبل از فعال‌سازی.
3. **UI ثبت نقشه** + نقشهٔ چندارتفاعی.
4. **Fusion** (EKF/UKF) + covariance.
5. **HITL/SIH** + سخت‌افزار.
6. **سرویس 4090** + دیتاست واقعی.

## برآورد حجم/زمان

- انجام‌شده: ~۸٬۵۰۰ خط (Python/HTML/JS/CSS)، ۲۳ endpoint، ۹ template، ۱۶ تست،
  نقشهٔ ۳۶۰ keyframe، ۱۲ تایل تولیدشده، ۱۱GB مدل، ۲.۸GB داده.
- باقی‌مانده برای یک پایلوت قابل‌اعتماد: **~۵–۹ هفته** (بسته به HITL و سخت‌افزار).
  - localization/control/fusion: ۲–۳ هفته
  - HITL + bring-up: ۲–۴ هفته
  - 4090/data/field: ۱–۲ هفته

## معیار پذیرش پایلوت

- GPS-denied از شروع یا میانهٔ مسیر، بدون blind land.
- خطای موقعیت افقی < ۱۰m (۹۵٪) در مسیر ۲۰۰m.
- geofence و failsafe فعال.
- HITL با Pixhawk و دوربین واقعی.
- گزارش تکرارپذیر با seed/مسیر/مدل.
