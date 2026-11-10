# GEPRC F405 + ArduPilot — سند جامع وضعیت و ادامهٔ کار

آخرین به‌روزرسانی: **2026-11-07**
این سند طوری نوشته شده که یک نشست تازه (یا یک نفر دیگر) بدون هیچ زمینهٔ قبلی بتواند ادامه دهد.

---

## ۰) خلاصهٔ وضعیت (TL;DR)

| بخش | وضعیت |
|---|---|
| فریم‌ور ArduPilot Copter روی برد | ✅ کار می‌کند (MAVLink روی USB = COM3) |
| کالیبراسیون IMU / جهت | ✅ انجام و تأیید شد |
| **موتورها** | ✅ کار می‌کنند — **فقط با DShot300** (`MOT_PWM_TYPE=5`) |
| نگاشت موتورها (Quad-X) | ✅ با تست فیزیکی تأیید شد |
| جهت چرخش هر ۴ موتور | ✅ مطابق استاندارد ArduPilot |
| باتری / جریان | ✅ کالیبره |
| **پرواز واقعی** | ❌ هنوز نه — نیاز به **لینک بی‌سیم** (و برای autonomy: GPS/Flow) |
| مشکل حل‌نشده | ❌ بوت سرد USB (باید با DFU بالا آورد) |

**قدم بعدی منطقی:** تهیهٔ لینک تله‌متری (SiK یا ESP32 یا Raspberry Pi) و سپس اولین هوور در حالت STABILIZE.

---

## ۱) چک‌لیست سریع برای شروع هر نشست

```powershell
# 1) برد را در DFU ببرید: باتری قطع → دکمه BOOT را نگه دارید → USB را وصل کنید → ۳ ثانیه → رها
# 2) فریم‌ور را بنویسید (پارامترها حفظ می‌شوند):
E:\drone-vla\ardupilot-build\tools\dfu-util-static.exe -a 0 -s 0x08010000:leave -D "E:\drone-vla\ardupilot-build\arducopter_v3.bin"

# 3) بعد از فلش، COM3 بالا می‌آید. حالا باتری را وصل کنید.
# 4) تست سلامت:
C:\Users\BMarz\AppData\Local\Temp\opencode\mav-venv\Scripts\python.exe C:\Users\BMarz\AppData\Local\Temp\opencode\one_motor.py COM3 1 20 2
```

> اگر COM3 نیامد، دوباره DFU کنید. **هرگز باتری را وقتی USB وصل است قطع/وصل نکنید** (پورت می‌پرد).

---

## ۲) سخت‌افزار

| مورد | مقدار |
|---|---|
| برد | GEPRC F405 (`GEPRCF405`)، STM32F405، ۱MB فلش |
| ژیرو | **ICM42688P** روی SPI3 (CS = PA15) |
| بارومتر | **ندارد** (`HAL_BARO_ALLOW_INIT_NO_BARO 1`) |
| قطب‌نما | **ندارد** (`COMPASS_ENABLE=0`) |
| GPS | **ندارد** |
| باتری | **6S2P Li-ion 22.2V 9000mAh** (INR21700) |
| ریموت/ریسیور | **ندارد** |
| دوربین | آنالوگ FPV ~1200TVL (PAL/NTSC → حداکثر ~720×576) |
| ورود به DFU | دکمهٔ سفید BOOT کنار USB-C + وصل USB |

### شناسهٔ USB
* ArduPilot: `1209:5741` (COM3)
* Betaflight: `0483:5740` (STM32 VCP)
* DFU (ROM): `0483:df11`

---

## ۳) نقشهٔ پین‌ها

### خروجی موتورها (از Betaflight: `resource MOTOR`)

| پد | چانل ArduPilot | برچسب برد (BF) | موقعیت فیزیکی | جهت لازم |
|---|---|---|---|---|
| **PA8** | servo1 / PWM(1) | M1 | عقب-راست | CW |
| **PA9** | servo2 / PWM(2) | M2 | جلو-راست | CCW |
| **PA10** | servo3 / PWM(3) | M3 | عقب-چپ | CCW |
| **PC8** | servo4 / PWM(4) | M4 | جلو-چپ | CW |
| PC9 | PWM(5) | M5 | — | — |
| PB0 | PWM(6) | M6 | — | — |
| PB1 | PWM(7) | M7 | — | — |

تایمرها: PA8/PA9/PA10 = **TIM1** (advanced)، PC8/PC9 = **TIM8**، PB0/PB1 = **TIM3**.

### پورت‌های سریال

| ArduPilot | پین | استفاده |
|---|---|---|
| SERIAL0 | USB (PA11/PA12) | MAVLink |
| SERIAL1 | USART1 PB6/PB7 | **آزاد — بهترین گزینه برای تله‌متری** |
| SERIAL2 | USART2 PA2/PA3 | RCIN (ریسیور در Betaflight اینجا بود) |
| SERIAL3 | USART3 PC10/PC11 | آزاد / GPS |
| SERIAL4 | UART4 PA0/PA1 | آزاد |
| SERIAL5 | UART5 PC12/PD2 | آزاد |
| SERIAL6 | USART6 PC6/PC7 | آزاد / GPS |

> شناسه‌های `serial` در Betaflight **از صفر** شروع می‌شوند (0=USART1 و ...).

### سایر
* SPI1: OSD (CS = PA4) — SPI2: DataFlash (CS = PB12) — SPI3: IMU (CS = PA15)
* I2C1: PB8/PB9 — I2C2: PB10/PB11
* ADC: PC1 = ولتاژ باتری، PC0 = جریان، PC2 = RSSI
* LED: PC14 / PC15

---

## ۴) ⭐ خروجی موتور — مهم‌ترین یافتهٔ پروژه

### ESCهای این پرنده با PWM آنالوگ کار نمی‌کنند — فقط DShot300

```
MOT_PWM_TYPE = 5      ← DShot300   ✅ کار می‌کند
```

| مقدار | پروتکل | نتیجه |
|---|---|---|
| `0` | Normal PWM | ❌ موتور نمی‌چرخد (ESC ساکت) |
| `6` | DShot600 | ❌ بوق تکرارشونده (سیگنال نامعتبر) |
| **`5`** | **DShot300** | ✅ **کار می‌کند** |

شواهد اندازه‌گیری‌شده (موتور ۱، DShot300):
```
baseline     vib 0.12   gyro   0.0
50%          vib 2.30   gyro  32.5   (×20)
70%          vib 2.83   gyro  51.4
100%         vib 1.44   gyro 105.4
```

> **`MOT_PWM_TYPE` فقط با ری‌بوت اعمال می‌شود.** (در `AP_MotorsMatrix::init()` خوانده می‌شود)

### نگاشت نهایی سرووها (با تست فیزیکی تأیید شد)

```
SERVO1_FUNCTION = 35   # motor3 → PA8  = عقب-راست  (CW)
SERVO2_FUNCTION = 33   # motor1 → PA9  = جلو-راست  (CCW)
SERVO3_FUNCTION = 34   # motor2 → PA10 = عقب-چپ   (CCW)
SERVO4_FUNCTION = 36   # motor4 → PC8  = جلو-چپ   (CW)
```

✅ تغییر `SERVO*_FUNCTION` **بدون ری‌بوت** اعمال می‌شود
(چون `AP_Motors::rc_write` بر اساس «تابع» می‌نویسد، نه شمارهٔ کانال).

### ⚠️ ترتیب تست با شمارهٔ موتور یکی نیست

`MAV_CMD_DO_MOTOR_TEST` با «ترتیب تست» کار می‌کند:

| تست (param1) | موقعیت | تابع |
|---|---|---|
| 1 | جلو-راست | motor1 (33) |
| 2 | جلو-چپ | motor4 (36) |
| 3 | عقب-چپ | motor2 (34) |
| 4 | عقب-راست | motor3 (35) |

### تفاوت نام‌گذاری Betaflight ↔ ArduPilot

| موقعیت | برچسب برد (BF) | شماره ArduPilot |
|---|---|---|
| جلو-راست | M2 | M1 |
| عقب-چپ | M3 | M2 |
| عقب-راست | M1 | M3 |
| جلو-چپ | M4 | M4 |

جهت‌های چرخش در هر دو استاندارد **یکی** است؛ فقط شماره‌ها ۹۰ درجه جابه‌جا شده‌اند.
(تأیید شده با برچسب‌های «Propeller direction» روی خود موتورها: M1=CW، M2=CCW، M3=CCW، M4=CW)

---

## ۵) پارامترهای کلیدی فعلی

```
# ---- فریم و جهت ----
FRAME_CLASS            1        (Quad)
FRAME_TYPE             1        (X)
AHRS_ORIENTATION       8        (ROLL_180 — سنسور روی برد وارونه است)
COMPASS_ENABLE         0

# ---- موتور ----
MOT_PWM_TYPE           5        (DShot300)  ← حیاتی
SERVO1_FUNCTION        35
SERVO2_FUNCTION        33
SERVO3_FUNCTION        34
SERVO4_FUNCTION        36
MOT_PWM_MIN            1000
MOT_PWM_MAX            2000
MOT_SPIN_ARM           0.10
MOT_SPIN_MIN           0.15
MOT_SPIN_MAX           0.95
MOT_SAFE_DISARM        0
RC_SPEED               490      (نرخ آپدیت موتور)
SERVO_DSHOT_RATE       0        (0 = DShot با ۱ کیلوهرتز ثابت)
SERVO_DSHOT_ESC        0        (0 = هیچ فرمان DShot اجرا نمی‌شود)

# ---- باتری (6S Li-ion 9000mAh) ----
BATT_MONITOR           4
BATT_CAPACITY          9000
BATT_VOLT_MULT         11.0
BATT_AMP_PERVLT        17.9     ← از ibata_scale=179 بتافلایت (واحد 0.1 A/V)
BATT_AMP_OFFSET        0.764    ← بایاس اندازه‌گیری‌شده
BATT_LOW_VOLT          19.5
BATT_CRT_VOLT          18.0
BATT_ARM_VOLT          21.0
BATT_LOW_MAH           1500
BATT_CRT_MAH           800
MOT_BAT_VOLT_MAX       25.2
MOT_BAT_VOLT_MIN       19.8

# ---- ایمنی / آرمینگ ----
ARMING_SKIPCHK         0        (همهٔ چک‌ها فعال)
ARMING_NEED_LOC        0
ARMING_RUDDER          2
BRD_SAFETY_DEFLT       0
BRD_SAFETY_MASK        16368
BRD_SAFETYOPTION       3
BRD_BOOT_DELAY         2000

# ---- حالت‌های پرواز ----
FLTMODE1..6            0,2,5,6,4,9   (Stabilize, AltHold, Loiter, RTL, Guided, Land)

# ---- سریال ----
SERIAL2_PROTOCOL       23       (RCIN)
```

> **واحد `load` در MAVLink دهم‌درصد است.** مقدار ۳۰۰ = **۳۰٪** بار پردازنده (سالم)، نه ۳۰۰٪.

پشتیبان کامل: `params_backup_ap.txt` (۱۰۹۹ پارامتر) — با اسکریپت `dump_params.py` و `restore_params.py`.

---

## ۶) فریم‌ور و فلش

* ArduCopter **V4.8.0-dev** (master، commit 178c2ef)
* `APJ_BOARD_ID 41778` (غیررسمی)، بوت‌لودر `0x08000000`، برنامه `0x08010000`
* پارامترها: `STORAGE_FLASH_PAGE 1` → آدرس `0x08004000` (با فلش برنامه پاک نمی‌شود)

```powershell
# بوت‌لودر (فقط یک بار، بعد از بازگشت از Betaflight)
dfu-util-static.exe -a 0 -s 0x08000000:0x4000 -D GEPRCF405_bl_v4.bin

# برنامه
dfu-util-static.exe -a 0 -s 0x08010000:leave -D arducopter_v3.bin

# بازگشت کامل به Betaflight
dfu-util-static.exe -a 0 -s 0x08000000:leave -D betaflight_backup\betaflight_full_flash_08000000.bin
```

### محدودیت‌های 4.8-dev
* **HITL حذف شده** → از SITL استفاده کنید
* نام‌ها تغییر کرده: `ARMING_SKIPCHK` (نه `ARMING_CHECK`)، `RTL_ALT_M`، `WP_SPEED_MS`، `GPS1_TYPE`، `MAV_SYSID`
* `ARMING_SKIPCHK = -1` = رد کردن همهٔ چک‌ها

---

## ۷) مشکل USB (حل‌نشده)

| سناریو | نتیجه |
|---|---|
| بعد از `dfu-util ... :leave -D <fw>` | ✅ COM3 فوراً ظاهر می‌شود |
| بوت سرد از باتری، بعد وصل USB | ⚠️ گاهی کار می‌کند، اغلب نه |
| برد با باتری روشن + قطع/وصل باتری | ❌ پورت می‌پرد |
| در ویندوز | گاهی `Unknown USB Device (Device Descriptor Request Failed)` |

تلاش‌های بی‌اثر: `BRD_BOOT_DELAY`، بوت‌لودر بدون CRC، `try_boot=true`.

**راه‌حل عملی:** DFU + نوشتن فریم‌ور (۲۰ ثانیه). برای پرواز واقعی مهم نیست، چون تله‌متری از UART می‌آید نه USB.

### ⚠️ بوق‌های متوالی ESC
اگر بعد از تست‌های پی‌درپی، ESCها بوق پشت‌سرهم زدند و جریان بی‌کار بالا رفت
(۲-۱۰ آمپر نوسانی)، **باتری را قطع/وصل کنید**. حالت گذراست، ربطی به داغ شدن ندارد.

---

## ۸) روش‌های تست (اسکریپت‌های آماده)

همه در `C:\Users\BMarz\AppData\Local\Temp\opencode\` و با venv:

```powershell
$PY = "C:\Users\BMarz\AppData\Local\Temp\opencode\mav-venv\Scripts\python.exe"
$T  = "C:\Users\BMarz\AppData\Local\Temp\opencode"

& $PY "$T\one_motor.py"      COM3 1 20 2    # یک موتور، ۲۰٪، ۲ ثانیه
& $PY "$T\map_motors2.py"    COM3           # تست ۱..۴ با فاصلهٔ ۵ ثانیه
& $PY "$T\dir_motors.py"     COM3 10        # جهت چرخش، ۱۰٪ (آرام)
& $PY "$T\ap_read.py"        COM3 "PARAM1,PARAM2"
& $PY "$T\dump_params.py"    COM3 "E:\drone-vla\ardupilot-build\params_backup_ap.txt"
& $PY "$T\restore_params.py" COM3 "E:\drone-vla\ardupilot-build\params_backup_ap.txt"
```

### روش صحیح تست موتور
1. پروانه‌ها **نصب نباشند**
2. پرنده **مهار** شود
3. ترتیب/جهت را با نوار کاغذی روی بدنهٔ موتور ببینید
4. **هیچ‌وقت** برای مدت طولانی با گاز بالا روی میز تست نکنید (داغ شدن ESC)

---

## ۹) آرمینگ بدون ریموت (برای پرواز)

بدون RC هم می‌شود آرم کرد، ولی باید:

```
ARMING_SKIPCHK  = 32     # بیت ۶ = RC Channels  (بیت‌ها ۱-پایه‌اند در مستندات)
FS_GCS_ENABLE   = 1      # اگر لینک GCS قطع شد → Land
```

بیت‌ماسک `ARMING_SKIPCHK` (شماره‌ها در مستندات ۱-پایه‌اند، مقدار بیت = `1<<(n-1)`):

```
1  Barometer        2  Compass          3  GPS lock        4  INS
5  Parameters       6  RC Channels      7  Board voltage   8  Battery Level
10 Logging          11 Safety switch    12 GPS Config      13 System
14 Mission          15 Rangefinder      16 Camera          17 AuxAuth
18 VisualOdometry   19 FFT
```

⚠️ بدون ریموت، **قطع اضطراری فقط از طریق همان لینک بی‌سیم ممکن است** — پس لینک باید پایدار باشد.

بدون GPS/قطب‌نما فقط حالت‌های **STABILIZE، ALT_HOLD، SPORT** معنی دارند
(Loiter/RTL/Guided به تخمین موقعیت نیاز دارند).

---

## ۱۰) لینک بی‌سیم — گزینه‌ها

| گزینه | قیمت | برد | توضیح |
|---|---|---|---|
| **رادیو SiK** (915/433MHz) | ~۲۵-۳۵$ | ۱-۱۰km | استاندارد، پایدار، توصیه‌شده برای اولین پرواز |
| **ESP8266/ESP32 WiFi** | ~۵-۱۰$ | ۵۰-۲۰۰m | ارزان، latency بیشتر |
| **Raspberry Pi Zero 2 W** | ~۲۰$ | WiFi | **برای پروژهٔ VLA لازم است** (محاسبات روی پرنده) |
| **Jetson Nano/Orin** | ۲۰۰$+ | WiFi | VLA سنگین |

اتصال: به `SERIAL1` (USART1، PB6/PB7) → `SERIAL1_PROTOCOL=2` (MAVLink2)، `SERIAL1_BAUD=57`.

---

## ۱۱) SITL (شبیه‌سازی) — کار می‌کند ✅

```bash
cd /home/drone/ardupilot && ./waf configure --board sitl && ./waf copter

setsid nohup bash -c "sleep infinity | build/sitl/bin/arducopter \
  -w --model + --speedup 1 \
  --home -35.363261,149.165230,584,353 \
  --defaults Tools/autotest/default_params/copter.parm \
  --serial0 tcp:5780 --serial1 tcp:5781 --serial2 tcp:5782" \
  > /tmp/sitl5780.log 2>&1 < /dev/null &
```

نکات:
* SITL با قطع‌شدن کلاینت TCP خودش را می‌بندد → قبل از هر تست دوباره اجرا کنید
* `eeprom.bin` باید مال کاربر `drone` باشد، وگرنه `Param storage failed`
* نمونهٔ Gazebo آزمایشگاه روی 5760 است — تداخل نکنید

دموی پرواز خودکار: `sitl_autoflight.py`
(Guided → arm → takeoff 10m → goto 20m north → land → disarm) ✅

---

## ۱۲) MAVSDK

* `mavsdk 4.1.0` (بایندینگ native) وصل می‌شود ولی **پلاگین‌ها در دسترس نیستند**
  (`System` هیچ `telemetry`/`action` ندارد)
* فعلاً از **pymavlink** استفاده کنید (تست‌شده و پایدار)
* گزینهٔ بعدی: `mavsdk<4` یا `mavsdk-grpc`

---

## ۱۳) فایل‌ها

| مسیر | کاربرد |
|---|---|
| `betaflight_backup\betaflight_full_flash_08000000.bin` | بکاپ کامل ۱MB — بازگشت به Betaflight |
| `betaflight_backup\cli_dump.txt` | تنظیمات Betaflight (منبع `resource`ها) |
| `arducopter_v3.bin` | فریم‌ور فعلی |
| `GEPRCF405_bl_v4.bin` | بوت‌لودر (try_boot=true) |
| `params_backup_ap.txt` | پشتیبان ۱۰۹۹ پارامتر (وضعیت فعلی) |
| `flash_backup_before_bf_restore.bin` | بکاپ ۱MB فلش ArduPilot قبل از تست Betaflight |
| `tools\dfu-util-static.exe` | dfu-util 0.11 |
| `tools\lsusb-static.exe` | لیست دستگاه‌های USB |
| `C:\...\Temp\opencode\GEPRCF405_hwdef.dat` | hwdef فریم‌ور |
| `C:\...\Temp\opencode\GEPRCF405_hwdef-bl.dat` | hwdef بوت‌لودر |
| `C:\...\Temp\opencode\mav-venv` | venv: pymavlink, pyserial, mavsdk |
| `/home/drone/ardupilot` (WSL) | درخت ساخت ArduPilot |

---

## ۱۴) درس‌های آموخته‌شده (برای جلوگیری از تکرار)

1. **`SERVO_OUTPUT_RAW` اثبات خروجی فیزیکی نیست** — از `hal.rcout->read()` می‌آید
   (یعنی فقط ثابت می‌کند `write()` صدا زده شده).
2. **صدای ESC شاهد قابل اعتماد نیست** — BLHeli ملودی راه‌اندازی را مستقل از سیگنال می‌زند.
3. **با PWM آنالوگ وقت تلف نکنید** — این ESC فقط DShot300 می‌خواهد.
4. **ترتیب تست موتور ≠ شمارهٔ موتور** در ArduPilot.
5. **`ARMING_SKIPCHK` بیت‌ماسک ۱-پایه است** (بیت ۶ = RC = مقدار ۳۲).
6. **قبل از هر تغییر بزرگ، بکاپ پارامتر بگیرید** (`dump_params.py`).
7. **باتری را با USB وصل، قطع/وصل نکنید** — پورت USB می‌پرد.
8. برای تست سخت‌افزار، Betaflight با DShot300 ابزار تشخیص عالی بود
   (ایمیج کاملش داریم و برمی‌گردانیم).

---

## ۱۵) کارهای باقی‌مانده (به ترتیب اولویت)

1. **لینک بی‌سیم** (SiK یا ESP32 یا Pi) — پیش‌نیاز هر پروازی
2. تنظیم `ARMING_SKIPCHK=32` + `FS_GCS_ENABLE=1`
3. **اولین هوور** در STABILIZE با پروانه، مهار/فضای باز، آماده برای disarm فوری
4. تنظیم دقیق `MOT_SPIN_ARM` / `MOT_SPIN_MIN` بعد از دیدن رفتار واقعی
5. (برای autonomy) **Optical Flow + Rangefinder** یا GPS+قطب‌نما
6. (اختیاری) حل دائمی مشکل USB
7. (اختیاری) بارومتر BMP280 روی I2C2
