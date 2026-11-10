# Visual Localization Lab — راهنمای کامل اجرا

این ریپو یک آزمایشگاه قابل‌تکرار برای تست ایده‌های **مکانیابی تصویری پهپاد** است:

- ساخت **tile store متریک north-up** از تصاویر ماهواره‌ای (GeoTIFF / JP2، شامل JP2های چرخیده)
- ساخت **سکانس query** از UAV-VisLoc، پوشهٔ فریم، یا **ویدیوی واقعی + telemetry**
- تولید **رفرنس هوایی شبیه‌سازی‌شده** با `SGLang` (FLUX.3 Klein) یا `ComfyUI`/`BFL`/`dummy`
- تطبیق با `SIFT` / `ORB` / `AKAZE` / `LoFTR` / `LightGlue`
- انتخاب کاندید با **prior سرعت/موقعیت** (به‌جای جستجو در کل دیتابیس)
- گیت‌های `scale` / `jump` / `coverage` / `confirmation`
- گزارش `JSON`/`CSV`/`Markdown` با `fix_rate`، `precision`، `false_fix`، `p95 error`، latency و recovery

---

## ۱) نقشهٔ فایل‌ها

```text
navlab/lab/
  tiles.py        # tile store متریک، WarpedVRT، TileStore/ReferenceStore
  sequences.py    # UAV-VisLoc، frame-folder، video+telemetry
  matchers.py     # رجیستری SIFT/ORB/AKAZE + dispatch به deep
  deep.py         # LoFTR و DISK+LightGlue (kornia)
  candidates.py   # Prior + CandidateSelector
  runner.py       # LabRunner/RunnerConfig، gateها، refine موقعیت
  metrics.py      # محاسبهٔ متریک‌ها
  report.py       # summary/results/frames/report
  generate.py     # تولید رفرنس برای هر tile
  references.py   # ensemble ماهواره + تولیدشده
  bench.py        # اجرای ماتریسی benchmark
  synthetic.py    # صحنه/query مصنوعی برای validate کردن harness
  cli.py          # python -m navlab.lab
experiments/lab/   # YAML configهای benchmark
tests/             # ۲۶ تست (lab + sat2aerial + navlab)
```

---

## ۲) پیش‌نیازها

- Linux، Python **3.11**
- `uv` (توصیه‌شده) یا `venv` + `pip`
- برای deep matcherها: GPU با CUDA (روی 4060 8GB تست شد؛ LoFTR/LightGlue در 768px جا می‌شوند)
- برای SGLang: دسترسی به سرور GPU؛ سرویس پیشنهادی روی `172.16.20.117:30010`

### نصب پایه

```bash
cd ~/drone-vla

# اگر env_flight را از قبل داری:
uv pip install --python env_flight/bin/python \
  numpy opencv-python-headless rasterio pyproj pillow pandas pyyaml \
  requests fastapi "uvicorn[standard]" sqlmodel shapely python-multipart aiofiles

# اگر از صفر می‌سازی:
uv venv env_flight --python 3.11
uv pip install --python env_flight/bin/python \
  numpy opencv-python-headless rasterio pyproj pillow pandas pyyaml \
  requests fastapi "uvicorn[standard]" sqlmodel shapely python-multipart aiofiles
```

برای deep matcherها:

```bash
# حالت عادی (اینترنت سالم)
uv pip install --python env_flight/bin/python kornia

# اگر PyPI کند/بلاک بود، از mirror:
uv pip install --python env_flight/bin/python --no-deps \
  --index-url https://mirrors.aliyun.com/pypi/simple kornia
```

نسخهٔ تست‌شده: `kornia==0.8.3` + `kornia_rs==0.2.0`.
اگر `uv` به‌خاطر شبکه timeout داد، می‌توانی از کش `~/.cache/uv/archive-v0/` هم استفاده کنی
(پکیج استخراج‌شده را به `env_flight/lib/python3.11/site-packages/` کپی کن).

### وزن‌های deep matcher

kornia معمولاً خودش دانلود می‌کند. اگر دانلود بلاک بود، این‌ها را در
`~/.cache/torch/hub/checkpoints/` بگذار:

| فایل | منبع |
|---|---|
| `loftr_outdoor.ckpt` | `http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt` |
| `depth-save.pth` | `https://gitee.com/xueyoo/disk/raw/master/depth-save.pth` |
| `disk_lightglue_v0-1_arxiv-pth` | release رسمی LightGlue (`disk_lightglue.pth`) |

نمونهٔ دانلود با mirror:

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
curl -L -o ~/.cache/torch/hub/checkpoints/depth-save.pth \
  https://gitee.com/xueyoo/disk/raw/master/depth-save.pth
```

---

## ۳) آدرس‌ها و APIها

### SGLang / FLUX.3 Klein (توصیه‌شده)

| مورد | مقدار پیش‌فرض |
|---|---|
| سرویس | `http://172.16.20.117:30010` |
| health | `GET /health` |
| مدل‌ها | `GET /v1/models` |
| تولید تصویر | `POST /v1/images/edits` |
| env override | `NAVLAB_SGLANG_URL` |

فرمت درخواست `POST /v1/images/edits` (multipart):

```text
image                    فایل ماهواره (PNG)
prompt                   متن prompt
size                     "768x768"
num_inference_steps      "16"
seed                     "0"
response_format          "b64_json"
output_format            "png"
```

پاسخ: `{"data":[{"b64_json":"..."}]}`. پیاده‌سازی کلاینت: `sat2aerial/backends/sglang.py`.

**شروع/توقف سرویس روی سرور GPU** (نام کاربری و host را خودت بگذار):

```bash
export SGLANG_HOST=172.16.20.117
export SGLANG_PORT=30010
export FLUX3_USER=<remote-user>
export FLUX3_ROOT=/home/ava/flux3-klein

# وضعیت
ssh -l "$FLUX3_USER" "$SGLANG_HOST" "cd '$FLUX3_ROOT' && bash api/status.sh"
# شروع
ssh -l "$FLUX3_USER" "$SGLANG_HOST" "cd '$FLUX3_ROOT' && bash api/sglang_start.sh"
# توقف
ssh -l "$FLUX3_USER" "$SGLANG_HOST" "cd '$FLUX3_ROOT' && bash api/sglang_stop.sh"
# تست
curl -s "http://$SGLANG_HOST:$SGLANG_PORT/v1/models"
```

سرویس با `--model-variant fp8r --batching-max-size 2 --dit-cpu-offload=false` بالا می‌آید.
روی H100 80GB تست شد: ~۳.۵s برای یک تصویر 768px و ~۱۳s برای چهار تصویر.

#### اتصال مستقیم از یک سرور/کلاینت دیگر به FLUX.3 روی ۱۱۷

```bash
export NAVLAB_SGLANG_URL=http://172.16.20.117:30010

# سلامت سرویس
curl -s "$NAVLAB_SGLANG_URL/health"
# مدل و وضعیت
curl -s "$NAVLAB_SGLANG_URL/v1/models"
```

نمونهٔ Python برای تولید یک تصویر از tile ماهواره:

```python
import base64
import os
import requests

url = os.environ.get("NAVLAB_SGLANG_URL", "http://172.16.20.117:30010")
with open("tile.png", "rb") as handle:
    response = requests.post(
        f"{url}/v1/images/edits",
        data={
            "prompt": "top-down aerial view of the same area, nadir, realistic",
            "size": "768x768",
            "num_inference_steps": "16",
            "seed": "0",
            "response_format": "b64_json",
            "output_format": "png",
        },
        files={"image": ("tile.png", handle, "image/png")},
        timeout=1800,
    )
response.raise_for_status()
image_b64 = response.json()["data"][0]["b64_json"]
with open("generated.png", "wb") as handle:
    handle.write(base64.b64decode(image_b64))
```

استفاده از wrapper آماده (همان API، بدون نوشتن requests):

```python
from pathlib import Path

from sat2aerial.backends import make_backend

backend = make_backend("sglang", server="http://172.16.20.117:30010", seed=0)
meta = backend.generate(
    satellite=Path("tile.png"),
    prompt="top-down aerial view, nadir",
    out_path=Path("generated.png"),
    size=(768, 768),
    seed=0,
)
```

یا از CLI خود آزمایشگاه:

```bash
export NAVLAB_SGLANG_URL=http://172.16.20.117:30010
env_flight/bin/python -m navlab.lab generate \
  --store outputs/navlab/lab/uav03_tiles \
  --out outputs/navlab/lab/uav03_tiles/generated_sglang \
  --backend sglang --view nadir --size 768x768 --altitude 466
```

نکته‌ها:
- `NAVLAB_SGLANG_URL` را در `env_flight`/shell خود ست کن؛ `navlab.config.SGLANG_URL` و `navlab.generation.get_backend("sglang")` از همین استفاده می‌کنند.
- سرویس فعلاً **بدون authentication** است؛ فقط از شبکهٔ مطمئن/LAN استفاده کن و پورت `30010` را عمومی باز نکن.
- اگر از سرور دیگری وصل می‌شوی، مطمئن شو firewall اجازهٔ TCP به `172.16.20.117:30010` را می‌دهد.
- اگر `Connection refused` گرفتی یعنی سرویس پایین است؛ با `api/sglang_start.sh` روی همان سرور بالا بیاور.

### ComfyUI (جایگزین لوکال)

| مورد | مقدار پیش‌فرض |
|---|---|
| سرویس | `http://127.0.0.1:8188` |
| health | `GET /system_stats` |
| تولید | `POST /prompt` (گراف ComfyUI) |
| env override | `NAVLAB_COMFY_URL` |

```bash
cd ComfyUI
../env_comfy/bin/python main.py --lowvram --disable-auto-launch \
  --listen 127.0.0.1 --port 8188 --preview-method none
```

### BFL API

```bash
export BFL_API_KEY=<key>
```

### NavLab web

```bash
env_flight/bin/python -m navlab
# http://127.0.0.1:8080
```

پورت‌های پیشنهادی: `30010` (SGLang)، `8080` (NavLab)، `8188` (ComfyUI)، `8090` (ComfyUI gateway).

---

## ۴) داده‌ها

```text
data/raw/uav_visloc/UAV_VisLoc_dataset/
  03/ satellite03.tif  drone/*.JPG  03.csv
  06/ ...
  07/ ...
  10/ ...
data/pairs/                 # cropهای قدیمی UAV-VisLoc (smoke test)
navlab/final_Request points/*.jp2
```

`data/` و `outputs/` در gitignore هستند. روی سرور جدید باید داده‌ها را جدا کپی کنی.
UAV-VisLoc محلی فقط زیرمجموعهٔ maps 03/06/07/10 است (۱۰/۸/۱/۱۲ فریم).
map 07 ستون‌های `Omega/Kappa` ندارد؛ فیلتر near-nadir برای آن پیش‌فرض صفر می‌شود.

---

## ۵) اجرای end-to-end

همهٔ دستورها از ریشهٔ ریپو اجرا می‌شوند.

### ۵.۱ ساخت query از UAV-VisLoc

```bash
env_flight/bin/python -m navlab.lab sequence \
  --root data/raw/uav_visloc/UAV_VisLoc_dataset \
  --map 03 --near-nadir 15 --max-frames 50 \
  --out outputs/navlab/lab/uav03_queries.json
```

### ۵.۲ ساخت tile store متریک north-up

```bash
env_flight/bin/python -m navlab.lab tile-store \
  --source data/raw/uav_visloc/UAV_VisLoc_dataset/03/satellite03.tif \
  --queries outputs/navlab/lab/uav03_queries.json \
  --tile-ground-m 700 --tile-px 768 \
  --out outputs/navlab/lab/uav03_tiles
```

- `--tile-ground-m` را نزدیک footprint واقعی دوربین بگذار (مثلاً `2*altitude*tan(FOV/2)`).
- اگر GSD هدف می‌خواهی: `--gsd 0.5` به‌جای `--tile-ground-m`.
- برای JP2 با bbox: `--bbox min_lon,min_lat,max_lon,max_lat`.
- برای تصاویر چرخیده: نیازی به کار خاصی نیست؛ `local_crs` + `calculate_default_transform` خروجی north-up می‌سازد.

### ۵.۳ تولید رفرنس با SGLang

```bash
export NAVLAB_SGLANG_URL=http://172.16.20.117:30010

env_flight/bin/python -m navlab.lab generate \
  --store outputs/navlab/lab/uav03_tiles \
  --out outputs/navlab/lab/uav03_tiles/generated_sglang \
  --backend sglang --view nadir --size 768x768 --altitude 466 --seeds 1
```

خروجی: `references_nadir.json` + تصاویر در `generated/nadir/`.

### ۵.۴ اجرای ensemble (ماهواره + تولیدشده) و benchmark

```bash
env_flight/bin/python -m navlab.lab combine \
  --store outputs/navlab/lab/uav03_tiles \
  --references outputs/navlab/lab/uav03_tiles/generated_sglang/references_nadir.json \
  --out outputs/navlab/lab/uav03_tiles/ensemble.json \
  --name uav03_ensemble

env_flight/bin/python -m navlab.lab run \
  --store outputs/navlab/lab/uav03_tiles \
  --queries outputs/navlab/lab/uav03_queries.json \
  --references outputs/navlab/lab/uav03_tiles/ensemble.json \
  --matcher lightglue --mode gray --max-dim 768 \
  --confirmation 1 --scale-tolerance 1.0 \
  --out outputs/navlab/lab/run_uav03_ensemble

env_flight/bin/python -m navlab.lab bench \
  --config experiments/lab/uav03_bench.yaml \
  --out outputs/navlab/lab/bench_uav03
```

### ۵.۵ ویدیوی واقعی + telemetry

```bash
env_flight/bin/python -m navlab.lab video \
  --video outputs/navlab/lab/real_flight/flight.mp4 \
  --telemetry outputs/navlab/lab/real_flight/telemetry.csv \
  --fps 2 \
  --out outputs/navlab/lab/real_flight/queries
```

سپس همان `run` را با `--queries .../queries/queries.json` اجرا کن.
فرمت telemetry:

```csv
timestamp,lat,lon,alt,yaw
0.0,35.6892,51.3890,120.0,90.0
```

اگر telemetry نداری، `has_truth=False` می‌شود و فقط matching/prior سنجیده می‌شود
(خطای موقعیت نسبت به truth گزارش نمی‌شود).

---

## ۶) مرجع CLI

| دستور | کار |
|---|---|
| `tile-store` | ساخت tile store از TIF/JP2 با route/queries/bbox |
| `sequence` | استخراج query از UAV-VisLoc |
| `video` | استخراج فریم از ویدیو + همگام‌سازی telemetry |
| `generate` | تولید رفرنس با dummy/sglang/comfy/bfl |
| `combine` | ساخت manifest ensemble |
| `run` | یک اجرای localization با matcher/gateهای دلخواه |
| `bench` | اجرای ماتریسی چند reference/matcher/mode |

نمونهٔ `bench` (YAML):

```yaml
store: outputs/navlab/lab/real_scene_probe/tiles
queries: outputs/navlab/lab/real_scene_probe/queries.json
reference_sets:
  - name: satellite
  - name: generated_sglang
    references: outputs/navlab/lab/real_scene_probe/tiles/generated_sglang/references_nadir.json
    reference_kind: generated
matchers: [sift, orb, akaze, loftr, lightglue]
modes: [gray]
runner:
  max_dim: 768
  base_radius: 150
  max_radius: 1200
  max_jump_m: 300
  scale_tolerance: 1.0
  max_candidates: 8
  bootstrap_candidates: 15
  confirmation: 1
  rng_seed: 0
```

---

## ۷) خروجی‌ها و متریک‌ها

هر run یک پوشه می‌سازد:

```text
outputs/navlab/lab/<run>/
  summary.json   # config + summary + store/reference info
  results.csv    # یک ردیف خلاصه
  frames.csv     # per-frame: truth/prior/error/latency/reason/match metrics
  report.md      # جدول خوانا
```

متریک‌های مهم:

- `match_rate`: نسبت فریم‌هایی که matcher موفق شد (آستانه‌های inlier/coverage)
- `fix_rate`: accepted / کل فریم‌ها
- `precision`: accepted fixهایی که خطایشان ≤ `max_error_m` است
- `false_fix_rate`: accepted fixهای بد
- `p95_error_m`: خطای افقی ۹۵٪
- `time_to_first_fix_s`، `recovery_s` (بعد از dropout)
- `mean_latency_ms`, `mean_candidates`

`--confirmation 2` یک fix را فقط بعد از دو match پیوسته و نزدیک قبول می‌کند
(کاهش false positive، اما نصف شدن fix_rate در سکانس‌های کوتاه).

---

## ۸) نتایج مرجع (صحنهٔ مصنوعی، ۱۰ فریم)

| matcher | reference | accepted | precision | p95 error | latency |
|---|---|---|---|---|---|
| SIFT | ماهوارهٔ خام | 10/10 | 1.0 | 9.74m | ~420ms |
| LoFTR | ماهوارهٔ خام | 10/10 | 1.0 | 9.73m | ~407ms |
| LightGlue | ماهوارهٔ خام | 10/10 | 1.0 | 9.86m | ~214ms |
| SIFT | SGLang-generated | 10/10 | 1.0 | 9.69m | ~350ms |
| LoFTR | SGLang-generated | 10/10 | 1.0 | 9.69m | ~403ms |
| LightGlue | SGLang-generated | 10/10 | 1.0 | 9.74m | ~209ms |

نتیجه: در این صحنه رفرنس تولیدشده با SGLang **هم‌ارز ماهوارهٔ خام** است و `gray` بهتر از `edges` کار می‌کند.

UAV-VisLoc واقعی (03/06/07/10) با وجود pose دقیق، تطبیق ضعیفی دارد
(در بهترین حالت ۲/۱۰ fix با precision 1.0). علت اصلی georeference/زاویهٔ دوربین خود دیتاست است،
نه harness. برای benchmark واقعی، ویدیوی خودت با telemetry دقیق را ترجیح بده.

---

## ۹) تست‌ها

```bash
env_flight/bin/python -m unittest discover -s tests -q
# 26 tests, OK
```

---

## ۱۰) عیب‌یابی

| مشکل | راه‌حل |
|---|---|
| `SGLang is not reachable` | سرویس را با `api/sglang_start.sh` بالا بیاور؛ `curl http://HOST:30010/v1/models` |
| `kornia` نصب نمی‌شود | از mirror (`mirrors.aliyun.com`, `pypi.tuna.tsinghua.edu.cn`) یا کش `~/.cache/uv` استفاده کن |
| دانلود وزن LoFTR/DISK کند است | از Gitee/ghproxy دانلود و در `~/.cache/torch/hub/checkpoints/` بگذار |
| `WarpedVRT` با `resolution` خطا/رفتار عجیب | `resolution` را به `WarpedVRT` نده؛ `calculate_default_transform` + `transform/width/height` استفاده کن (در `tiles.py` پیاده شده) |
| `boundless reads` روی VRT | `_read_tile` پنجره را به extent کلیپ می‌کند؛ همان را نگه دار |
| همهٔ matchها با `scale` رد می‌شوند | `--scale-tolerance 1.0` یا `0`؛ `tile_ground_m` را نزدیک footprint بگذار |
| JP2 چرخیده/مقیاس اشتباه | `inspect_image` حالا `gsd_xy_m` و `rotated` می‌دهد؛ `crop_bounds` north-up است |
| UAV-VisLoc match نمی‌شود | محدودیت دیتاست؛ از صحنهٔ مصنوعی یا ویدیوی واقعی با georeference خوب استفاده کن |
| GPU کم است | `--max-dim 768` برای deep؛ SGLang روی سرور جدا اجرا شود |

---

## ۱۱) نکات عملیاتی

- هیچ کلید/رمز در ریپو نگذار؛ از `BFL_API_KEY`, `NAVLAB_SGLANG_URL`, `NAVLAB_COMFY_URL` استفاده کن.
- `data/`, `outputs/`, `*.jp2`, `*.ckpt`, `*.pth` در gitignore هستند.
- برای دسترسی از سرور دیگر، فقط پورت `30010` (SGLang) و در صورت نیاز `8080` (NavLab) را در firewall باز کن.
- SGLang با batch=2 موازی کار می‌کند؛ برای تولید انبوه، چند `generate` موازی یا `jobs.py` با `concurrency=2` بزن.
- برای تکرارپذیری، `--rng-seed 0` و `--seeds` ثابت را نگه دار؛ کانفیگ و git commit هر run در `summary.json`/`report.md` ثبت می‌شود.
