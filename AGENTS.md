# drone-vla — راهنمای عامل

این مخزن شامل چند بخش است (VLA، SITL، Gazebo/NavLab، و پیکربندی ناوبری).

## 🚁 کار جاری: هواپیمای GEPRC F405 + ArduPilot

**قبل از هر اقدام روی سخت‌افزار، این سند را کامل بخوانید:**

```
ardupilot-build/HANDOFF_AP.md
```

سند بالا شامل: نقشهٔ پین‌ها، پارامترهای فعلی، یافتهٔ DShot300،
نگاشت موتورها، مشکل USB، روش تست، و کارهای باقی‌مانده است.

### نکات کلیدی
* **موتورها فقط با `MOT_PWM_TYPE=5` (DShot300) کار می‌کنند.**
* بوت سرد USB کار نمی‌کند → همیشه با **DFU** بالا بیاورید.
* قطب‌نما/GPS/بارومتر/ریموت ندارد.
* پشتیبان پارامترها: `ardupilot-build/params_backup_ap.txt`
* بکاپ بازگشت به Betaflight: `ardupilot-build/betaflight_backup/betaflight_full_flash_08000000.bin`

## محیط
* WSL: `wsl -d Ubuntu-24.04 -u drone` — درخت ساخت ArduPilot در `/home/drone/ardupilot`
* ویندوز venv: `C:\Users\BMarz\AppData\Local\Temp\opencode\mav-venv`
* SITL مستقل روی `tcp:127.0.0.1:5780` — نمونهٔ Gazebo آزمایشگاه روی 5760 است، تداخل نکنید.
