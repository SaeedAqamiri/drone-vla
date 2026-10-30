# drone-vla — یادداشت‌های فنی (چیزهایی که یاد گرفتیم)

> هر چیزی که موقع راه‌اندازی و تست این بستر یاد گرفتیم. قبل از هر کاری اینجا رو چک کن.

## ۱. نقشه ورک‌اسپیس

| مسیر | چی هست |
|---|---|
| `env_isaac45/` | venv پایتون 3.10 — Isaac Sim 4.5 (pip) + Isaac Lab 2.1 + rsl_rl + torch 2.5.1+cu124 |
| `env_flight/` | venv پایتون 3.11 — mavsdk 3.17.4 (`mavsdk<4`!) + pymavlink + pyulog + pandas + lerobot + huggingface_hub |
| `IsaacLab-2.1/` | کلون تگ v2.1.0 — نصب editable روی `env_isaac45` |
| `PX4-Autopilot/` | PX4 main — SITL بیلدشده (`make px4_sitl gz_x500`) |
| `flux-action/` | کلون black-forest-labs/flux-action + venv پایتون 3.12 خودش (torch 2.10.0+cu128, natten 0.21.6+torch2100cu128, bitsandbytes) |
| `models/flux-3-action-base/` | دانلود HF — trunk 13.97GB + video_vae 2.5GB + text_encoder (Qwen3-VL-4B) 8.9GB |
| `scripts/` | `flight_log.py`، `isaac_quad_logger.py`، `flux3_quant_test.py` |
| `logs/` | mission1-3.csv (تله‌متری پرواز) + quad_*.npz (اپیزود Isaac) + ulog_local_position.csv |

## ۲. دستورهای روزمره

```bash
# ── PX4 SITL (بدون HEADLESS برای دیدن گرافیک Gazebo) ──
cd ~/drone-vla/PX4-Autopilot && HEADLESS=1 make px4_sitl gz_x500
# بوت کامل = این خط در لاگ: "Startup script returned successfully"

# ── پرواز + لاگ (در ترمینال دوم، بعد از بوت) ──
~/drone-vla/env_flight/bin/python ~/drone-vla/scripts/flight_log.py \
  --fly --altitude 5 --hold 8 --duration 95 --out ~/drone-vla/logs/flight.csv

# ── لاگ اپیزود RL پهپاد در Isaac Lab ──
env_isaac45/bin/python ~/drone-vla/scripts/isaac_quad_logger.py --headless --num_envs 64 --steps 5000
# (بدون --headless = پنجره گرافیکی Isaac Sim)

# ── GUI خالی Isaac Sim ──
env_isaac45/bin/isaacsim

# ── تبدیل ulog خود PX4 ──
~/drone-vla/env_flight/bin/python -c "from pyulog import ULog; ..."
# فایل‌ها: PX4-Autopilot/build/px4_sitl_default/rootfs/fs/log/<date>/*.ulg
```

## ۳. تله‌های محیط این ماشین (مهم!)

- **پروسه‌های بک‌گراند وقتی یک tool/shell تایم‌اوت می‌خوره کشته می‌شن** → برای سیم‌هایی که باید بمونن از `setsid nohup ... &` استفاده کن و با PID بکش (نه pkill -f؛ الگوی pkill می‌تونه شل خودت رو match کنه و بکشته! از `pkill -f "[p]attern"` استفاده کن).
- **RAM کل 16GB است** → هرگز تنسور 14GB (trunk) را روی CPU materialize نکن؛ meta device + streaming از safetensors mmap.
- کلید دانلودhf: گاهی `--include` را نادیده می‌گیرد و "complete" گزارش می‌دهد — بعد از دانلود `ls` کن و فایل‌های بزرگ را چک کن.
- کش uv را با hardlink به venvها می‌بندد → `uv cache prune` چیزی آزاد نمی‌کند؛ برای فضای واقعی فایل‌های یتیم را بگیر: `find ~/.cache/uv/archive-v0 -type f -links 1 -delete` (فقط بعد از حذف venvها).
- GPU: RTX 4060 Laptop **8GB** — سقف اجرا؛ مدل‌های 14GB+ نیاز به GPU کرایه‌ای 24GB+ دارند.
- cmake را داخل venv نصب کردیم (`uv pip install cmake`) تا `isaaclab.sh` sudo نخواهد.

## ۴. PX4 + Gazebo

- `Tools/setup/ubuntu.sh` با sudo می‌شکند (می‌نویسد به `/home/root/.bashrc`) → فقط Gazebo را دستی از ریپو OSRF نصب کن: `gz-harmonic libunwind-dev` + `libopencv-dev` + gstreamer plugins.
- دیپ‌های بیلد PX4 با pip: `kconfiglib empy==3.3.4 jinja2 jsonschema pyros-genmsg` (`--user`).
- هارت‌بیت‌های MAVLink به 127.0.0.1:**14540** می‌آیند؛ `ss -lun` لیسنر نشان نمی‌دهد — طبیعی است.
- بوت سالم = "Startup script returned successfully". تست سریع شبکه: bind روی 14540 و recvfrom.
- **آرم فقط روی سیمِ تازه‌بوت‌شده راحت جواب می‌دهد**؛ بعد از پرواز/فرود، آرم با pymavlink بارها TEMPORARILY_REJECTED شد ("Resolve system health failures first") — با MAVSDK روی بوت تازه روی همان سیم آرم شد. ترتیب امتحان‌شده: بوت تازه → MAVSDK arm → takeoff → hold → land → disarm ✓ (ارتفاع 5.03m ثبت شد).
- ulog سشن‌ها: `vehicle_local_position` z از -5.02 تا 0 = همان صعود ۵ متری.

## ۵. MAVSDK (نسخه 3.17.4 gRPC)

1. حتماً `mavsdk<4` پین شود (v4 = API جدید با نام mavsdk-grpc).
2. آدرس اتصال: `udpin://0.0.0.0:14540` (`udp://:14540` خطای "No network interface" می‌دهد).
3. `hold()` زیر **`drone.action.hold()`** است نه `drone.hold()`.
4. `asyncio.wait_for(gen.__anext__(), t)` هنگ می‌کند — async-for را داخل یک coroutine بپیچ و روی آن wait_for بزن.
5. sampler تله‌متری باید **قبل از** دستورات پرواز استارت شود وگرنه CSV پرواز را از دست می‌دهد.
6. `mavsdk_server` یتیم از ران قبلی می‌تواند پورت‌ها را بگیرد → قبل از ران مجدد بکشش.

## ۶. Isaac Lab + پهپاد

- محیط ثبت‌شده: **`Isaac-Quadcopter-Direct-v0`** (direct workflow: obs 12 بعدی، action 4 بعدی روتور، rsl_rl).
- `gym.make` wrapper برمی‌گرداند → `env.unwrapped.device` و `env.unwrapped.step_dt` (نه `env.device`).
- `env.step()` خودش envهای terminated را reset می‌کند.
- بار اول اجرا ~۱۰ دقیقه pull اکستنشن؛ بعدی‌ها سریع.
- 8GB VRAM برای اجرای headless با چند env کافی است (تست شد).

## ۷. FLUX 3 Action (مدل World Action از BFL)

- **معماری:** trunk DiT با 6.95B پارامتر (345 لایه Linear) + KinoVAE (2.5GB) + انکودر متن Qwen3-VL-4B (8.9GB). bf16 روی GPU رسماً ~32GB می‌خواهد.
- **وزن‌ها:** `black-forest-labs/flux-3-action-base` روی HF — **گیت نیست**؛ revision پین‌شده: `62878e2925e59b7a89ec14463ce89932624c490d`.
- **کانفیگ پهپاد/گیم:** canvas 512×512، دوربین تکی `images.game`، اکشن ۴ بعدی `[forward, lateral, up, yaw]` در [-1,1] با 15Hz، chunk=32، سمپلر `cosmos_unipc` با `num_inference_steps=1` (اینفرنس سبک)، `attn_mode='torch'` (NATTEN موقع اجرا لازم نیست).
- **دیتاست fine-tune پهپاد (adapter اسمش `rotor`):** اپیزودهای NPZ با `frames (T,256,256,3) uint8` + `action (T,4) float32` + `index.json` که هر مدخلش `task` (دستور پروازی) دارد؛ adapter خودش پیشوند `fly the drone: ` را می‌گذارد. **فرمت NPZ لاگر Isaac ما دقیقاً همین است — فقط سنسور دوربین باید اضافه شود.** خروجی policy: `plan (1, 32, 4)`؛ برای closed-loop هر بار آخرین اکشن اجراشده را به‌عنوان state بده.
- **offload داخلی:** `policy.set_text_encoder_offload(True)` انکودر متن را روی CPU نگه می‌دارد (8.9GB آزاد می‌شود؛ فقط یک‌بار به‌ازای هر instruction به GPU می‌آید).
- **پیش‌بینی حافظه روی 8GB (تست با `scripts/flux3_quant_test.py`):**
  | حالت | trunk | + VAE | جمع | نتیجه |
  |---|---|---|---|---|
  | bf16 | 14GB | 2.5 | ~16.5+ | ❌ (عملاً محال؛ رسماً ~32GB) |
  | int8 | ~7.4GB | 2.5 | ~9.9 | ❌ **تأیید شد: OOM در 7.46GB** |
  | nf4 (4-bit) | ~3.7GB | 2.5 | **6.80 peak** | ✅ **تأیید شد: اجرا شد!** |

- **نتیجهٔ عملی nf4 روی 4060 Laptop 8GB** (`--quant nf4`):
  - لود کامل: 37 ثانیه → 6.25GB پایدار، peak در اینفرنس **6.80GB**
  - خروجی: `plan (1, 32, 4)` — اکشن پهپاد آمادهٔ closed-loop
  - زمان هر پلن: ~1.1 ثانیه (پس پلنِ 32 اکشنی = 2.13s پرواز در 15Hz → با execution horizon=8 یعنی 533ms اکشن به‌ازای 1.1s محاسبه؛ ~2x کندتر از real-time ولی قابل استفاده برای تست، نه کنترل لحظه‌ای)
  - انکودر متن در تست = mock؛ در استفادهٔ واقعی `set_text_encoder_offload(True)` (CPU، ~0.5s به‌ازای هر instruction جدید فقط یک‌بار)
- **باگ‌هایی که هنگام کوانتیزیشن bnb پیدا شد (برای مراجعهٔ بعدی):**
  1. `embed_timesteps` از `weight.dtype` برای dtype خروجی استفاده می‌کند → وزنِ packed uint8 باعث ورودی uint8 به SiLU می‌شود → پچ شد به bf16
  2. `FinalLayer.forward` روی **وزن خام** `chunk(2)` می‌زند (`adaLN_modulation[1].weight.chunk(2)`) → لایه‌های `.adaLN_modulation` باید bf16 بمانند (فقط ~150MB)
  3. هدهای fresh (`emb_in.<mod>`, `final_layer.<mod>`) باید از کوانتیزیشن مستثنی باشند وگرنه `fix_4bit_weight_quant_state_from_module` assert می‌زند
  4. کلیدهای trunk با `remap_shared_action_keys` از `action_prediction` به نام modality (مثلاً `game`) remap می‌شوند — هنگام لود تن‌به‌تن باید (new_name ← orig_key) را از همین تابع بگیری
- ترفند اسکریپت تست: DiT روی `meta` ساخته می‌شود، trunk تن‌به‌تن از safetensors (mmap) به GPU استریم و Linearها با bitsandbytes کوانتیزه می‌شوند → RAM سیستم همیشه پایین می‌ماند.
- fine-tune واقعی (طبق داکیومنت BFL) با ۸ GPU انجام شده — این بخش مال GPU کرایه‌ای است.

## ۹. پایپ‌لاین دیتای rotor برای FLUX 3 Action (پاث Gazebo — تست‌شده ✓)

فرمت اپیزود (دقیقاً همان adapterِ `rotor` در flux-action):
`frames (T,256,256,3) uint8` + `action (T,4) float32 در [-1,1]` + `index.json` با `task` (بدون پیشوند — adapter خودش `fly the drone: ` اضافه می‌کند)

**قرارداد اکشن دیتاست ما:** `[pitch_cmd, roll_cmd, thrust_cmd, yaw_cmd]` در 15Hz — hover برای Crazyflie = 0.526

سه اسکریپت (دو پروسهٔ همزمان + merge):
```bash
# ۱) سیم با دوربین: مدل mono_cam به 256x256 ویرایش شده (Tools/simulation/gz/models/mono_cam/model.sdf)
make px4_sitl gz_x500_mono_cam          # در PX4-Autopilot
# ۲) کپچر فریم (python سیستمی 3.10 — بایندهای apt: python3-gz-transport13, python3-gz-msgs10)
/usr/bin/python3 scripts/gz_frame_capture.py \
  --topic /world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image \
  --frames 5000 --trigger outputs/data/rotor_gz/run1/START \
  --done outputs/data/rotor_gz/run1/START.done --outdir outputs/data/rotor_gz/run1
# ۳) پرواز + لاگ اکشن (لاگر خودش START را می‌سازد و در پایان START.done)
env_flight/bin/python scripts/gz_flight_logger.py --steps 300 \
  --trigger outputs/data/rotor_gz/run1/START --out outputs/data/rotor_gz/run1/actions.csv
# ۴) merge با همترازی timestamp
env_flight/bin/python scripts/merge_rotor_episode.py --frames-dir ... --actions ... \
  --steps 300 --task "..." --out ep_0000.npz
```
- پروتکل trigger/done حیاتی است: کپشن فقط mtime جدیدتر از استارت خودش را قبول می‌کند و تا START.done ضبط می‌کند (وگرنه فریم‌های قبل/بعد از پرواز ثبت می‌شد)
- خروجی تأییدشده: `ep_0000.npz` (300 گام، motion واقعی در فریم‌ها، فازهای اکشن متمایز)
- نمونهٔ خروجی: `outputs/data/rotor_gz/ep_0000.npz` ✓

**نکتهٔ مهم Isaac Sim — حل شد (2026-09-28):** با درایور 595 رندر RTX هر دو نسخه 5.1 و 4.5 سگ‌فلت می‌داد. راه‌حل نهایی:
1. **دان‌گرید درایور به 580.178.04** (`nvidia-driver-580-open`) — باگِ کوئریِ «max instances» که 1 برمی‌گرداند در 595 بود
2. **Isaac Sim 4.5 در venv جدا** (`env_isaac45`، py3.10، `isaacsim[all,extscache]==4.5.0.0` از pypi.nvidia.com) — توجه: pip خودش torch 2.14 نصب می‌کند؛ باید `torch==2.5.1+cu124` دانگرید شود
3. **تنظیمات حیاتی kit** (در sys.argv قبل از SimulationApp):
   - `--/rtx-transient/scenedb/maxInstancesLimit=4000000` + `forceMaxTLASInstancesLimit=true` (پیش‌فرض نامحدود = بافر 7.5GB → کرش)
   - `--/renderer/activeGpu=0` + `--/renderer/multiGpu/enabled=False` (لپ‌تاپ hybrid)
4. **swap موقت 32GB** (`/swapfile2`) — بوت رندر 12GB+ RAM می‌خواهد؛ بدون آن OOM kill می‌شود
5. تجربهٔ `isaacsim.exp.base.kit` (نه full که viewport می‌خواهد)
6. کپچر از طریق `isaacsim.sensors.camera.Camera` (مسیر native) + `rep.orchestrator.step(rt_subframes=32)` + ذخیرهٔ فوری خام + `os._exit(0)` (teardown کیت گاهی سگ‌فلت رقابتی دارد)
7. نکتهٔ writers 4.5: basic_writer حذف شده؛ `isaacsim.replicator.writers` بعد از core استارت می‌شود و رجیستری را می‌شوید → `register_writers()` را دستی صدا بزنید؛ PytorchWriter (بدون param rgb، با listener) و DOPE/Pose/YCB موجودهند
نتیجه: `scripts/render_sensor45.py` فریم RGB واقعی می‌گیرد → `outputs/isaac45_sensor_frame.png`. مسیر Gazebo برای دیتا همچنان ساده‌تر است ولی حالا Isaac هم برای فوتورئال باز است. اسکریپت `isaac_rotor_logger.py` (Isaac Lab) هنوز به Isaac Lab 2.1 روی 4.5 نیاز دارد — نصب نشده.

## ۱۰. وضعیت کارهای ناتمام

- [x] اجرای `flux3_quant_test.py --quant nf4` → **تأیید: FITS، peak 6.80GB، پلن (1,32,4) در ~1.1s**
- [x] تست int8 → **تأیید: OOM (7.46GB)**
- [x] تولید اپیزود rotor از شبیه‌ساز → **پاث Gazebo تست شد و کار می‌کند**
- [x] پورت لاگر rotor به Isaac 4.5 + Isaac Lab 2.1 ✓ (تگ v2.1.0 در `IsaacLab-2.1/`؛ نکات نصب: `setuptools==75.8.0` لازم است چون 84 pkg_resources ندارد؛ `pip install -e source/isaaclab --no-build-isolation`؛ خطای `Unknown asset config type for clone_in_fabric` یعنی آن فیلد در 2.1 وجود ندارد — حذفش کن؛ خوشبختانه `scene.sensors` به‌عنوان فیلد cfg در 2.1 موجود است (برخلاف main) پس TiledCameraCfg به‌عنوان فیلد + انتساب در `_setup_scene` کار می‌کند)
- [ ] چند اپیزود متنوع (تغییر task/seed) + index.json نهایی
- [ ] closed-loop: اتصال خروجی policy به کنترلر پهپاد
- [ ] تست GUI هر دو شبیه‌ساز روی دسکتاپ

## ۱۱. پاک‌سازی سیستم (2026-09-30)

- **حذف شد:** `env_isaaclab/` (Isaac Sim 5.1، 18G) + `IsaacLab/` (کلون main) — رندر 5.1 هرگز روی این سیستم کار نکرد؛ همه‌چیز روی 4.5 + IsaacLab-2.1 است. `isaac_quad_logger.py` حالا باید با `env_isaac45` اجرا شود (تست نشده).
- **حذف شد:** `/swapfile2` (32G سواپ موقت) — ⚠ قبل از رندر Isaac 4.5 باید دوباره ساخته شود:
  ```bash
  sudo fallocate -l 32G /swapfile2 && sudo chmod 600 /swapfile2 && sudo mkswap /swapfile2 && sudo swapon /swapfile2
  ```
- **کش‌ها:** `~/.cache/ov` (Omniverse — خودش دوباره می‌سازد)، pip cache، کش یتیم uv (28G — فایل‌های nlink=1 بعد از حذف env_isaaclab)، `~/knime-setup` (نصاب قدیمی)، deb کروم در Downloads.
- **درس uv:** prune فایل‌های hardlink-شده را آزاد نمی‌کند؛ بعد از حذف هر venv بزرگ، `find ~/.cache/uv/archive-v0 -type f -links 1 -delete` بزن.
- نگه داشته شدند: استک Whisper (`~/models/whisper-*`، `~/pyenvs/whisper`، `~/opt/whisper*`)، `~/opt/cuda-*`، `~/miniforge3`، `~/opencode`.

## ۱۲. sat2aerial — بیک‌آف ماهواره ← هوایی (2026-10-21)

بستهٔ `sat2aerial/` برای سنجش ایدهٔ «تولید تصویر هوایی از ماهواره و بعد matching» ساخته شد.

- **متری که مهم است:** `inliers` بین تصویر تولیدشده و فریم واقعی پهپاد، در مقایسه با baseline ماهوارهٔ خام.
  `cons_inliers` نشان می‌دهد مدل هندسهٔ ماهواره را حفظ کرده یا ساختمان/جاده جعل کرده.
- **بک‌اندها:** `dummy` (بدون مدل، فقط اعتبارسنجی harness)، `bfl` (API رسمی FLUX 3 Image Klein، ~$0.009/عکس)،
  `flux3` (لوکال با diffusers؛ روی 8GB عملاً دردناک).
- **مدل گیت‌شده است:** `black-forest-labs/flux-3-image-klein` با `gated: auto`؛ باید `hf auth login` +
  پذیرش لایسنس FLUX Kommunity روی صفحهٔ مدل. بدون آن دانلود 401 می‌دهد.
- **واقعیت حافظه:** transformer fp8r = 6.27GB، bf16 = 11.1GB؛ text encoder (Qwen3-VL) = 8.9GB؛
  benchmark خود HF برای fp8r در 1024px حدود 16.3GB peak بدون offload. ادعای 8GB مال ComfyUI با block-swap است،
  نه diffusers. برای بیک‌آف از API یا GPU کرایه‌ای استفاده کن؛ روی لپ‌تاپ 512px + fp8r را امتحان کن.
- **دستورها:**
  ```bash
  env_flight/bin/python -m sat2aerial selfcheck --out /tmp/sat2aerial_selfcheck
  env_flight/bin/python -m sat2aerial demo --out outputs/sat2aerial/demo --sites 4 --clean
  env_flight/bin/python -m sat2aerial bakeoff --pairs data/pairs --out outputs/sat2aerial/bakeoff \
    --backend bfl --variants nadir,oblique30,nadir_style,oblique30_style --size 768
  env_flight/bin/python -m unittest discover -s tests -v
  ```
- **تست‌شده:** selfcheck (inliers=147، ratio=0.54) و demo روی ۴ صحنهٔ مصنوعی (۱۹ ثانیه؛ گزارش در
  `outputs/sat2aerial/demo/bakeoff/report.md`). بک‌اند BFL و flux3 هنوز اجرای واقعی نشده‌اند (کلید/مدل نداریم).
- **گام بعد:** یک دیتاست paired واقعی (UAV-VisLoc / SUES-200 / DenseUAV) + اجرای bakeoff با BFL،
  و بعد تصمیم بین semantic matching و sim2real augmentation.

### ۱۲.۱ — ComfyUI int8 روی 8GB (تست‌شده ✓)

- ریپوی مدل: `black-forest-labs/flux-3-image-klein-comfy` (گیت `auto` جداست؛ لایسنسش را جدا بپذیر).
- فایل‌ها: DiT int8 = 5.5GB، text encoder Qwen3-VL int8 = 4.6GB، VAE = 321MB.
- نصب: `ComfyUI/` + venv `env_comfy` (torch 2.10 cu128). فقط نود اختیاری OpenGL لود نمی‌شود (بی‌اهمیت).
- اجرا:
  ```bash
  cd ~/drone-vla/ComfyUI
  ../env_comfy/bin/python main.py --lowvram --disable-auto-launch \
    --listen 127.0.0.1 --port 8188 --preview-method none
  ```
- بک‌اند جدید `--backend comfy` در `sat2aerial` به `/prompt` گراف می‌فرستد
  (UNETLoader → CLIPLoader type=flux3 → CLIPTextEncode → VAEEncode → ReferenceLatent → KSampler euler/simple 16 steps cfg=1 → VAEDecode).
- زمان: ~۵۶s برای تک‌رفرنس و ~۸۴s برای دو رفرنس در ۷۶۸px روی RTX 4060 8GB؛ لود اولیهٔ ~۱۰GB از دیسک طولانی‌تر است.
- prompt upsampler (۹B) استفاده نمی‌شود؛ پرامپت‌های `build_prompt` خودشان مفصل‌اند.

**یافتهٔ مهم — واریانس seed:** روی صحنهٔ مصنوعی site_00، با seed=7 هندسهٔ شهر تقریباً کامل حفظ شد
(`cons_inliers=147`)؛ با seed=0 مدل یک شهر مشابه ولی جابه‌جاشده ساخت (`cons_inliers=9.5`, `success=0.25`).
پس bakeoff حالا `--seeds N` دارد و جدول seed variance هم در گزارش می‌آید؛ نتیجهٔ تک‌seed معتبر نیست.

**دیتاست واقعی UAV-VisLoc:** با `remotezip` فقط `satellite10.tif` (۱۰۲MB) و ۱۲ فریم drone از zip ۱۷GB استخراج شد
(بدون دانلود کل فایل). ولی مسیر پرواز loc10 نزدیک لبهٔ چپ نقشه است و crop ساده baseline را صفر inlier کرد.
بهترین crop با grid-search فقط ۲۵ inlier داد؛ مشکل georeferencing/لبهٔ نقشه/اختلاف فصل است. قبل از نتیجه‌گیری
واقعی باید offset و footprint دوربین کالیبره شود یا از دیتاستی با جفت‌های مرکزی (SUES-200/DenseUAV) استفاده شود.

**درس uv:** کش `~/.cache/uv/archive-v0` خراب بود (۱۸۴ archive بدون `RECORD`). اسکریپت پاک‌سازی و بعد نصب
بدون `--no-cache` جواب داد؛ `--no-cache` برای torch باعث دانلود مجدد ۲.۵GB و تایم‌اوت می‌شود.

### ۱۲.۲ — نتیجهٔ multi-seed و نمونهٔ واقعی (2026-10-21)

**Multi-seed روی صحنهٔ مصنوعی** (۲ سایت × ۲ variant × ۲ seed، FLUX 3 int8):
- `nadir` بدون style ref: success=0.25، میانگین ۱۳ inlier، بازهٔ seedها ۷–۲۲.
- `nadir_style`: success=1.00، میانگین ۱۴۲ inlier، بازهٔ ۱۰۰–۱۹۰.
- baseline ماهوارهٔ خام: ۱۷۵ inlier، success=1.00.
یعنی style reference برای این مدل حیاتی است؛ بدون آن مدل چیدمان را دوباره «می‌سازد».

**University-1652 (نمونهٔ واقعی):** از zip عمومی ۹.۲GB با `remotezip` فقط چند جفت استخراج شد
(`test/gallery_satellite/0000/0000.jpg` + `test/gallery_drone/0000/image-01.jpeg`؛ هر دو 512×512).
- baseline: ۹ inlier، ratio=0.129، coverage=0.136 (fail با آستانهٔ ۱۵).
- تصویر تولیدشده با پرامپت oblique30: **۷ inlier**، ratio=0.10، coverage=0.087؛ یعنی بهتر نشد.
- ولی consistency با ماهواره: ۵۱۹ inlier، coverage=0.938، edge IoU=0.92، SSIM≈0.96.
- مونتاژ `outputs/sat2aerial/uni0000_montage.png` نشان می‌دهد خروجی **همان نمای top-down ماهواره** است،
  نه نمای oblique. مدل هندسهٔ رفرنس را قربانی پرامپت نمی‌کند.

**UAV-VisLoc loc03:** cropهای geo ساده با صحنهٔ drone جور نبودند؛ ۴۳ inlier اولیه همگی روی یک خط
(رودخانه/جاده) بودند و coverage=0.0. برای همین `min_coverage=0.03` به معیار success اضافه شد تا
false positiveهای هم‌خط رد شوند.

**نتیجه:** translation دوبعدی هندسه و ظاهر را حفظ می‌کند ولی **تغییر viewpoint سه‌بعدی** (parallax/occlusion)
را نمی‌سازد. برای فریم‌های oblique پهپاد باید از DSM/3DGS رندر گرفت یا مأموریت near-nadir باشد.

### ۱۲.۳ — bakeoff واقعی University-1652 (۵ جفت، 512px)

`data/pairs/uni1652/` از zip عمومی `haiduonghuynhle/University1652` (۹.۲GB) با `remotezip` ساخته شد
(satellite 512×512 + gallery_drone 512×512، همان ساختمان، نمای drone oblique).

| variant | success | inliers | cons_inliers | edge IoU | SSIM |
|---|---|---|---|---|---|
| baseline | 0.00 | 9.4 | - | - | - |
| nadir (بدون style) | 0.00 | 7.4 | 437.8 | 0.885 | 0.879 |
| oblique30_style | 0.00 | 8.6 | 141.0 | 0.344 | 0.292 |

هیچ‌کدام baseline را بهتر نکرد و هیچ‌کدام success نشد. خروجی همچنان top-down است؛ پرامپت oblique
viewpoint سه‌بعدی نمی‌سازد و style ref در این حالت هندسه را هم خراب‌تر می‌کند.
گزارش: `outputs/sat2aerial/uni1652_bakeoff/report.md`؛ مونتاژ: `sites/<id>/..._montage.png`.

## ۱۳. NavLab — پلتفرم UI (2026-10-28)

- **اجرا:** `env_flight/bin/python -m navlab` روی پورت 8080؛ UI فارسی/RTL با FastAPI + Jinja2 + HTMX + Leaflet.
- **دیتابیس:** SQLite در `outputs/navlab/navlab.db`؛ مدل‌ها Mission/MapSource/Job/GeneratedTile/Run.
- **صفحه‌ها:** داشبورد، مأموریت، آزمایش سریع، SITL، سخت‌افزار، تنظیمات.
- **تست‌شده:** ساخت مأموریت، آپلود PNG، تولید dummy، تطبیق، Run با ۵۰۳ inlier و coverage=0.81،
  آزمایش سریع با ۵۴۴ inlier. خطای قبلی coverage=0 روی UAV-VisLoc به‌درستی reject شد.
- **SITL:** PX4 + Gazebo Harmonic، home روی نقطهٔ شروع مأموریت، MAVSDK روی 14540.
  takeoff → goto → land تست شد (۳۰m، ~۱۰۰m جابه‌جایی). ارتفاع دمو حداکثر ۵۰m.

**تله‌های حل‌شده:**
1. SQLModel/SQLAlchemy 2.1 `datetime.utcnow()` را رد می‌کند → `datetime.now(timezone.utc)`.
2. PX4 با stdin=DEVNULL در حلقه `pxh>` می‌افتد و لاگ ۳GB تولید می‌کند → `stdin=PIPE` باز + reader thread
   و ring buffer (بدون فایل لاگ).
3. `mavsdk_server` یتیم پورت 14540 را می‌گیرد → قبل از start/stop با pkill الگوی محدود پاک می‌شود.
4. `goto_location` ارتفاع **AMSL** می‌خواهد؛ `home_alt + target` استفاده شد. قبل از takeoff باید
   `health.is_armable` و `set_takeoff_altitude` را ست کرد، وگرنه PX4 به‌خاطر attitude failure فرود می‌آید.
5. FastAPI: به‌خاطر مشکل ریداکشن `@app...` از `app.add_api_route` استفاده شد (بدون decorator).

**باقی‌مانده:** استریم دوربین Gazebo در UI، waypoint کامل + geofence، نقشهٔ آنلاین XYZ،
GPS-denied با تزریق EV، و HITL.

### ۱۳.۱ — دوربین و waypoint/geofence (2026-10-28)

- **دوربین زنده:** `scripts/gz_latest_frame.py` با system Python + gz، فریم‌ها را به `latest.jpg` می‌نویسد؛
  `CameraStream` آن را با SITL start/stop می‌کند و UI از `/api/sim/camera.jpg` می‌خواند.
- **تله:** protobuf سیستم جدید با protoهای gz سازگار نیست → `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`
  قبل از import تنظیم شد.
- **مدل/دنیا:** `x500_mono_cam` دوربینش رو به جلو بود و تصویر افق می‌داد؛ `x500_mono_cam_down` + `baylands`
  تصویر رو به پایین و بافت‌دار می‌دهد. topic auto-discover می‌شود.
- **waypoint/geofence:** نقاط میانی در `mission.meta["waypoints"]`؛ مسیر چندقطعه‌ای با
  `route_summary_multi`/`tile_bounds_multi`؛ geofence = بافر کوریدور (polygon).
- **پرواز:** `fly_to_end` حالا روی همهٔ نقاط می‌رود، هر leg تا فاصلهٔ <20m صبر می‌کند، و اگر فاصله از
  polyline از `corridor_km/2 + 150m` بیشتر شود land + خطا می‌دهد. تست: ۴ نقطه، ۳۰m، فرود نزدیک endpoint.
- تست‌ها: ۱۶/۱۶.

### ۱۳.۲ — BOM اکسل

- `scripts/make_hardware_xlsx.py` فایل `outputs/navlab/hardware_bom.xlsx` را می‌سازد
  (شیت‌های Summary/Drone_BOM/GPU_Station/Available/Notes، RTL، فرمول‌های USD/تومان).
- نرخ ارز در `Summary!B10` قابل ویرایش است؛ فرمول‌های تومان خودکار به‌روز می‌شوند.
- در `/hardware` دکمهٔ دانلود به `/media/hardware_bom.xlsx` وصل شد.

### ۱۳.۳ — GPS-denied EV (مرحلهٔ ۱) — کار کرد

- **منبع EV:** `scripts/gz_truth_bridge.py` از topic `/world/baylands/pose/info` موقعیت مدل را در
  `outputs/navlab/truth/truth.json` می‌نویسد (۲۰Hz). قبلاً `gz model -p` هر ۵Hz timeout می‌داد و
  EV قطع/وصل می‌شد.
- **تزریق:** `VISION_POSITION_ESTIMATE` با `MavlinkDirect` (fields_json، covariance=0.5، usec=0).
- **پارامترها:** `EKF2_EV_CTRL=11` (horizontal+vertical+yaw)، `EKF2_HGT_REF=3`، `EKF2_GPS_CTRL=0`،
  `COM_ARM_WO_GPS=1`، `EKF2_EV_NOISE_MD=1`، `EKF2_EVP_NOISE=0.5`. Global origin روی start مأموریت.
- **نتیجه:** arm شد، takeoff تا ۳۰m، ۴ waypoint طی شد، land شد. `lat/lon` در EKF معتبر شد
  (35.6892, 51.389) و EV فعال ماند.
- **هشدارها/باقی‌مانده:** «no heading reference» و «horizontal position unstable» هنوز در preflight
  دیده می‌شوند (ولی arm/پرواز انجام شد). در فرود overshoot ~۳۰۰m داشت (کنترل offboard دقیق‌تر لازم است).
- **مهم:** EV فعلاً از **Gazebo truth** است، نه از دوربین/map-matching. مرحلهٔ بعد جایگزینی منبع EV با
  خروجی تطبیق تصویری و سپس offboard local NED است.

### ۱۳.۴ — Map-based EV (نیمه‌کاره)

- نقشهٔ بصری از یک پرواز GPS ساخته شد: ۳۶۰ کی‌فریم ۲۵۶×۲۵۶ + truth در
  `outputs/navlab/maps/3/` (`record_map.py`). localizer با SIFT + prior مکانی.
- نتیجهٔ تست: در ابتدا درست match کرد (kf95 → kf107 x=5.6,y=4.9 → kf118 x=15,y=10.8)
  با inliers بالا و latency ~۱۹ms؛ ولی وسط مسیر match قطع شد (hold last) و PX4
  `Failsafe: blind land` زد. `goto_location` هم `COMMAND_DENIED`/invalid setpoints داد.
- جمع‌بندی: localizer در نزدیکی شروع خوب است؛ برای کل مسیر باید:
  1. جستجو بر اساس **ترتیب keyframe** (پنجرهٔ ±N حول آخرین kf) به‌جای فقط radius مکانی.
  2. prior متحرک + سرعت (پیش‌بینی) و آستانهٔ adaptive.
  3. کنترل با **offboard local NED** به‌جای `goto_location`.
  4. نقشه با ارتفاع‌های چندگانه/کی‌فریم بیشتر و کیفیت بهتر.
- سرور بالاست، SITL خاموش. truth-EV (۱۳.۳) کامل کار می‌کند.

### ۱۳.۵ — Offboard local NED: ریسک و گیت

- تلاش برای offboard `set_position_ned` در SITL منجر به setpoint نامعتبر، RTL/blind land و
  پرواز به خارج از مسیر شد (`offboard.stop` هم COMMAND_DENIED). بنابراین `use_offboard` پیش‌فرض
  **False** شد؛ مسیر اثبات‌شده همان `goto_location` + EV است.
- قبل از فعال‌سازی مجدد: تست bench با setpoint ثابت، بررسی فریم NED و `position_velocity_ned`،
  و محدودکردن حداکثر فاصله/سرعت.
- localizer حالا پنجرهٔ keyframe و hold-last-EV دارد؛ ولی offboard تا اعتبارسنجی فعال نمی‌شود.

### ۱۳.۶ — اعتبارسنجی آفلاین localizer (نتیجهٔ مهم)

- نقشه: ۳۶۰ keyframe؛ برای جلوگیری از self-match، فریم‌های فرد به‌عنوان map و زوج‌ها به‌عنوان live.
- نتیجه با radius 10–35m:
  - match rate: **۹۸.۳–۹۸.۹٪**
  - mean error: **~۰.۵–۰.۶m**، median ~۰.۰۱m، p95 **۱.۵–۱.۸m**
  - latency: **~۱۷ms**
- یعنی **map matching خودش خوب است**؛ قطع‌شدن وسط پرواز از کنترل/ارتفاع/failsafe بود، نه الگوریتم تطبیق.
- offboard فعلاً گیت شده؛ مسیر کنترل اثبات‌شده `goto_location` + EV است. قدم بعد: map EV را در ارتفاع ۳۰m
  با prior اولیه از INS فعال کن و ارتفاع را ثابت نگه دار.

### ۱۳.۷ — Staged map EV (GPS روشن) — نتیجهٔ خیلی خوب

- مسیر: GPS روشن، پرواز تا ۳۰m، سپس map EV با `disable_gps=0` و `init_prior=1`.
- نتیجه: localizer در کل مسیر match کرد؛ `truth_error_m` بین **۰.۲۸ تا ۱.۵۶m**، ارتفاع ۳۰m ثابت.
  یعنی map EV به‌عنوان منبع موقعیت افقی دقیق است.
- بعد از `gps/disable`: دو نمونه اول match/دقیق، بعد ارتفاع افت کرد و PX4 `Landing at current position` /
  `invalid setpoints` / `blind land` داد.
- علت اصلی: `EKF2_HGT_REF` **reboot-required** است؛ با خاموش‌کردن GPS، مرجع ارتفاع (که روی GPS مانده)
  نامعتبر می‌شود. راه‌حل: قبل از بوت `EKF2_HGT_REF=0` (baro) یا `3` (vision) را ست و SITL را ری‌استارت کن،
  یا از baro به‌عنوان height استفاده کن. سپس EV افقی (`EKF2_EV_CTRL=1`) کافی است.
- گیت offboard همچنان فعال است؛ کنترل با `goto_location` انجام شد.

### ۱۳.۸ — GPS-denied با map-EV + baro height (موفق)

- `EKF2_HGT_REF=0` (baro) قبل از بوت ست شد و SITL ری‌استارت شد (چون reboot-required).
- پرواز: GPS روشن → map EV با `EKF2_EV_CTRL=1` (افقی) + prior از INS → `gps/disable`.
- نتیجه: ارتفاع ۳۰m تا نزدیک end حفظ شد؛ map match تا kf≈۲۶۵ ادامه داشت؛
  `truth_error_m` حدود **۰.۲–۲.۹m**؛ فرود/landing نرمال (بدون blind land).
- نقص باقی‌مانده: overshoot افقی/attitude هنگام فرود (`roll` نهایی عجیب) و کنترل offboard گیت‌شده.
- `position_velocity_ned` rate=10Hz اضافه شد؛ prior اگر خالی باشد از telemetry lat/lon محاسبه می‌شود.

### ۱۳.۹ — موارد ۱–۴ (2026-10-30)

- **Landing:** `_settle` (`action.hold()` 2s) قبل از land؛ پرواز GPS endpoint را در ~۲m گرفت،
  roll نهایی ~۰ و بدون overshoot/failsafe.
- **Bench offboard:** `bench_offboard` setpoint ثابت را روی حالت disarmed می‌فرستد و `offboard.start`
  را پذیرفت (`ok:true`). پرواز offboard همچنان `use_offboard=False`.
- **Fusion:** `Fusion2D` (KF سرعت ثابت، process/measurement noise، gate=9.21) در `_ev_loop` map
  استفاده می‌شود؛ `fusion.update/predict_only` + status؛ ۳ تست.
- **Map UI:** `MapRecorder` (`navlab/mapping.py`) + APIهای `/api/sim/map/start|stop|status`؛
  تست ۲۳ فریم/۵s در `maps/mission3_alt30`؛ altitude در manifest؛ UI فریم/مسیر/وضعیت.
- **HITL:** `navlab/hardware.py` سریال‌ها را detect و چک‌لیست HITL را در `/hardware` نشان می‌دهد.
- تست‌ها: **۱۹/۱۹**.
