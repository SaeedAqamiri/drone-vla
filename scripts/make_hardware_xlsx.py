from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

OUT = Path("outputs/navlab/hardware_bom.xlsx")
RATE = 1_000_000

NAVY = "1F3864"
BLUE = "2F5597"
LIGHT = "EEF3FB"
WHITE = "FFFFFF"
BORDER = Side(style="thin", color="B4C6E7")
THIN = Border(left=BORDER, right=BORDER, top=BORDER, bottom=BORDER)

TITLE_FONT = Font(name="Tahoma", size=16, bold=True, color=WHITE)
SUB_FONT = Font(name="Tahoma", size=10, color="D9E2F3")
HEAD_FONT = Font(name="Tahoma", size=10, bold=True, color=WHITE)
BODY = Font(name="Tahoma", size=10, color="1F2937")
BOLD = Font(name="Tahoma", size=10, bold=True)

USD = '"$"#,##0'
TOMAN = '#,##0'
PCT = "0%"

DRONE = [
    ("Core", "Flight controller", "Pixhawk 6C (یا CUAV V5+)", 1, 150, 220, "خوب", "الزامی",
     "PX4 native، dual IMU، پشتیبانی HITL، قطعهٔ رایج در ایران",
     "برای HITL باید ماژول pwm_out_sim در firmware باشد"),
    ("Core", "Companion computer", "Raspberry Pi 5 8GB (یا Pi 4B 8GB)", 1, 75, 110, "متوسط / خوب", "الزامی",
     "اجرای SIFT/ORB/optical flow، MAVLink و ناوبری؛ بدون نیاز به GPU",
     "Pi 4B در ایران راحت‌تر و ارزان‌تر است؛ Pi 5 سریع‌تر"),
    ("Core", "Storage", "NVMe HAT + SSD 256GB", 1, 30, 50, "متوسط", "توصیه‌شده",
     "سیستم‌عامل، دیتاست، لاگ و کش تطبیق",
     "در صورت نبود، microSD صنعتی هم جواب می‌دهد"),
    ("Core", "Navigation camera", "Arducam OV9281 global shutter mono + لنز ۶mm", 1, 40, 65, "متوسط", "الزامی",
     "دوربین رو به پایین؛ global shutter برای کاهش motion blur در ۱۸۰km/h",
     "رزولوشن ۱MP؛ برای جزئیات بیشتر IMX477 رنگ هم اضافه کن"),
    ("Core", "GNSS (ground truth)", "u-blox M9N", 1, 25, 45, "خوب", "الزامی",
     "home/arming و ground truth در SITL/HITL؛ در سناریوی GPS-denied غیرفعال می‌شود",
     "برای مقایسهٔ مسیر واقعی و تخمینی لازم است"),
    ("Core", "Telemetry", "SiK 433MHz (جفت)", 1, 30, 60, "خوب", "الزامی",
     "لینک GCS؛ ۴۳۳MHz در ایران مجازتر از ۹۱۵ است",
     "برای HITL می‌توان از USB هم استفاده کرد"),
    ("Core", "Power", "Holybro PM02", 1, 20, 30, "خوب", "الزامی",
     "اندازه‌گیری ولتاژ/جریان و تغذیهٔ FC", ""),
    ("Core", "Power", "BEC 5V/5A", 1, 10, 15, "خوب", "الزامی",
     "تغذیهٔ پایدار Raspberry Pi", "Pi 5 به ۵V/۵A نیاز دارد"),
    ("Core", "Mechanical", "Vibration mount", 1, 5, 10, "خوب", "الزامی",
     "کاهش لرزش دوربین و IMU", ""),
    ("Core", "Thermal", "Heatsink + fan برای Pi", 1, 10, 15, "خوب", "الزامی",
     "Pi 5 زیر بار پردازش تصویر گرم می‌شود", ""),
    ("Core", "Mechanical", "Enclosure، standoff، کابل", 1, 20, 40, "خوب", "الزامی",
     "نصب، سیم‌کشی و محافظت", ""),
    ("Core", "Storage", "microSD صنعتی ۳۲GB", 1, 10, 15, "خوب", "توصیه‌شده",
     "بکاپ OS / ریکاوری", ""),
    ("Core", "HITL", "USB hub + کابل", 1, 10, 20, "خوب", "توصیه‌شده",
     "اتصال Pixhawk/Pi/لپ‌تاپ در HITL", ""),
    ("Optional", "Altitude", "TFmini-S rangefinder", 1, 40, 60, "متوسط", "توصیه‌شده",
     "ارتفاع AGL برای مقیاس optical flow و فرود", "بهبود پایداری ارتفاع"),
    ("Optional", "Optical flow", "PMW3901", 1, 20, 30, "متوسط", "اختیاری",
     "سرعت کمکی در بافت ضعیف / backup", ""),
    ("Optional", "Color camera", "Arducam IMX477 HQ", 1, 50, 70, "خوب", "اختیاری",
     "ضبط و بازبینی انسانی؛ rolling shutter", "برای ناوبری، OV9281 بهتر است"),
    ("Optional", "Edge AI", "Jetson Orin Nano", 1, 250, 500, "کم", "بعداً",
     "فقط اگر matcher یادگیری‌محور روی پهپاد خواستی", "فعلاً لازم نیست"),
]

GPU = [
    ("Core", "GPU", "RTX 4090 24GB", 1, 1600, 2500, "کم", "الزامی (برای تولید)",
     "FLUX 3 Klein BF16 بدون کوانتیزاسیون؛ 24GB برای 1024px تنگ است",
     "با text encoder offload و 64GB RAM قابل استفاده است"),
    ("Core", "Host", "CPU + 64GB RAM + 2TB NVMe + PSU 1000W", 1, 1000, 1500, "متوسط", "الزامی",
     "offload کردن text encoder ۸.۹GB و نگهداری مدل‌ها",
     "اگر فقط API/اجاره استفاده می‌کنی، فعلاً لازم نیست"),
    ("Optional", "Power", "UPS 1500VA", 1, 150, 250, "متوسط", "توصیه‌شده",
     "پایداری برق و جلوگیری از خرابی در قطعی", ""),
]

AVAILABLE = [
    ("Laptop", "RTX 4060 Laptop 8GB / 15GB RAM", 1, "موجود", "فقط feasibility؛ تولید FLUX روی آن توصیه نمی‌شود"),
    ("Software", "PX4-Autopilot + Gazebo Harmonic", 1, "موجود", "SITL و دوربین Gazebo"),
    ("Software", "ComfyUI + FLUX 3 Klein int8", 1, "موجود", "قابل اجرا روی 8GB؛ bf16 نه"),
    ("Software", "Isaac Sim 4.5 + Isaac Lab 2.1", 1, "موجود", "برای رندر فوتورئال اختیاری"),
    ("Data", "UAV-VisLoc / University-1652 samples", 1, "موجود", "برای ارزیابی تطبیق"),
    ("Service", "BFL API key / 4090", 0, "ندارد", "فعلاً dummy/ComfyUI؛ بعداً API یا سرویس داخلی"),
]


def style_sheet(ws, title: str, subtitle: str, headers: list[str], widths: list[int], header_row: int = 4):
    ws.sheet_view.rightToLeft = True
    ws.sheet_view.showGridLines = False
    last_col = len(headers)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=last_col)
    cell = ws.cell(row=1, column=1, value=title)
    cell.font = TITLE_FONT
    cell.fill = PatternFill("solid", fgColor=NAVY)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 30
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=last_col)
    cell = ws.cell(row=2, column=1, value=subtitle)
    cell.font = SUB_FONT
    cell.fill = PatternFill("solid", fgColor=NAVY)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[2].height = 22
    for index, header in enumerate(headers, start=1):
        cell = ws.cell(row=header_row, column=index, value=header)
        cell.font = HEAD_FONT
        cell.fill = PatternFill("solid", fgColor=BLUE)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = THIN
    ws.row_dimensions[header_row].height = 34
    for index, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(index)].width = width
    ws.freeze_panes = f"A{header_row + 1}"


def fill_rows(ws, start_row: int, rows: list[tuple], priority_col: int = 13):
    fills = {
        "الزامی": "FCE4D6",
        "توصیه‌شده": "FFF2CC",
        "اختیاری": "E7E6E6",
        "بعداً": "E7E6E6",
        "الزامی (برای تولید)": "FCE4D6",
    }
    for offset, row in enumerate(rows):
        excel_row = start_row + offset
        for col, value in enumerate(row, start=1):
            cell = ws.cell(row=excel_row, column=col, value=value)
            cell.font = BODY
            cell.border = THIN
            cell.alignment = Alignment(
                horizontal="center" if col in {1, 2, 5, 6, 7, 8, 9, 10, 11, 12} else "right",
                vertical="center",
                wrap_text=col in {3, 4, 14, 15},
            )
            if col in {6, 7, 8, 9}:
                cell.number_format = USD
            if col in {10, 11}:
                cell.number_format = TOMAN
            if col == priority_col:
                cell.fill = PatternFill("solid", fgColor=fills.get(str(value), LIGHT))
                cell.font = BOLD
            elif offset % 2 == 1:
                cell.fill = PatternFill("solid", fgColor=LIGHT)


def main() -> None:
    wb = Workbook()
    wb.calculation.fullCalcOnLoad = True

    # Summary
    ws = wb.active
    ws.title = "Summary"
    style_sheet(
        ws,
        "NavLab — لیست نیازهای سخت‌افزاری",
        "پهپاد ناوبری بدون GPS — بدون بدنه، باتری و موتور",
        ["بخش", "حداقل USD", "حداکثر USD", "حداقل تومان", "حداکثر تومان", "توضیح"],
        [30, 16, 16, 20, 20, 70],
    )
    ws["A5"] = "فرضیات"
    ws["A5"].font = Font(name="Tahoma", size=12, bold=True, color=NAVY)
    assumptions = [
        "مأموریت: ۲۰۰ متر AGL، دوربین تقریباً رو به پایین، مسیر ۵۰۰۰km (قابل تغییر)",
        "پردازش تصویر/Matching روی Raspberry Pi؛ تولید تصویر روی 4090 یا API",
        "قیمت‌ها دلاری و تقریبی‌اند؛ در ایران با نرخ آزاد و واردات معمولاً ۱.۵–۲ برابر می‌شوند.",
        "نرخ دلار را در سلول زرد B10 وارد کن؛ ستون‌های تومان خودکار به‌روز می‌شوند.",
    ]
    for index, text in enumerate(assumptions, start=6):
        ws.merge_cells(start_row=index, start_column=1, end_row=index, end_column=6)
        cell = ws.cell(row=index, column=1, value=text)
        cell.font = BODY
        cell.alignment = Alignment(horizontal="right", vertical="center", wrap_text=True)
    ws["A10"] = "نرخ دلار (تومان)"
    ws["A10"].font = BOLD
    ws["B10"] = RATE
    ws["B10"].number_format = TOMAN
    ws["B10"].fill = PatternFill("solid", fgColor="FFF2CC")
    ws["B10"].border = THIN
    ws.merge_cells("C10:F10")
    ws["C10"] = "این عدد را با نرخ آزاد روز جایگزین کن (فقط برای برآورد)."
    ws["C10"].font = Font(name="Tahoma", size=9, italic=True, color="7F6000")

    summary_headers = ["بخش", "حداقل USD", "حداکثر USD", "حداقل تومان", "حداکثر تومان", "توضیح"]
    for index, header in enumerate(summary_headers, start=1):
        cell = ws.cell(row=12, column=index, value=header)
        cell.font = HEAD_FONT
        cell.fill = PatternFill("solid", fgColor=BLUE)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = THIN
    summary_rows = [
        ("پهپاد — Core", "=Drone_BOM!H18", "=Drone_BOM!I18", "بدون اختیاری‌ها"),
        ("پهپاد — اختیاری", "=Drone_BOM!H23", "=Drone_BOM!I23", "Rangefinder/Flow/دوربین رنگ/Jetson"),
        ("ایستگاه 4090", "=GPU_Station!H8", "=GPU_Station!I8", "GPU + Host + UPS (UPS اختیاری)"),
        ("جمع کل", "=B13+B14+B15", "=C13+C14+C15", "شامل اختیاری‌ها؛ قبل از ضریب واردات/نرخ ارز"),
        ("موجود (هزینهٔ صفر)", 0, 0, "لپ‌تاپ، PX4/Gazebo، ComfyUI int8، Isaac"),
    ]
    for offset, row in enumerate(summary_rows):
        excel_row = 13 + offset
        for col, value in enumerate(row, start=1):
            cell = ws.cell(row=excel_row, column=col, value=value)
            cell.font = BOLD if excel_row == 16 else BODY
            cell.border = THIN
            cell.alignment = Alignment(horizontal="center" if col in {2, 3, 4, 5} else "right", vertical="center")
            if col in {2, 3}:
                cell.number_format = USD
            if col in {4, 5}:
                cell.number_format = TOMAN
            if offset % 2 == 1:
                cell.fill = PatternFill("solid", fgColor=LIGHT)
    for row in range(13, 18):
        ws.cell(row=row, column=4, value=f"=B{row}*$B$10").number_format = TOMAN
        ws.cell(row=row, column=5, value=f"=C{row}*$B$10").number_format = TOMAN

    ws["A19"] = "نکته‌ها"
    ws["A19"].font = Font(name="Tahoma", size=12, bold=True, color=NAVY)
    notes = [
        "برای یک مأموریت یک‌بارهٔ ۵۰۰۰km، API ارزان‌تر از خرید 4090 است (~$۳۴۳ برای ۳۸k تصویر).",
        "خرید 4090 وقتی منطقی است که تولید تکراری/سرویس داخلی بخواهی.",
        "HITL با Gazebo Classic یا SIH؛ Gazebo Harmonic پشتیبانی رسمی HITL ندارد.",
        "دوربین ناوبری باید global shutter باشد؛ IMX477 فقط برای ضبط/بازبینی.",
        "در ۲۰۰m و ۱۸۰km/h، motion blur و همگام‌سازی زمان مهم‌اند.",
    ]
    for index, text in enumerate(notes, start=20):
        ws.merge_cells(start_row=index, start_column=1, end_row=index, end_column=6)
        cell = ws.cell(row=index, column=1, value="• " + text)
        cell.font = BODY
        cell.alignment = Alignment(horizontal="right", vertical="center", wrap_text=True)

    # Drone BOM
    ws = wb.create_sheet("Drone_BOM")
    headers = [
        "ردیف", "دسته", "قطعه", "مدل/پیشنهاد", "تعداد",
        "قیمت واحد حداقل USD", "قیمت واحد حداکثر USD",
        "جمع حداقل USD", "جمع حداکثر USD",
        "جمع حداقل تومان", "جمع حداکثر تومان",
        "موجودی ایران", "اولویت", "توجیه", "ملاحظات",
    ]
    style_sheet(
        ws,
        "لیست سخت‌افزار پهپاد",
        "Core: لازم برای ناوبری و تست | Optional: بهبود یا فاز بعد",
        headers,
        [5, 10, 24, 38, 7, 13, 13, 13, 13, 16, 16, 12, 12, 60, 42],
    )
    rows = []
    for index, item in enumerate(DRONE, start=1):
        category, part, model, qty, low, high, availability, priority, reason, note = item
        rows.append((index, category, part, model, qty, low, high, None, None, None, None, availability, priority, reason, note))
    core_rows = rows[:13]
    optional_rows = rows[13:]
    fill_rows(ws, 5, core_rows)
    fill_rows(ws, 19, optional_rows)
    core_end = 17
    optional_start = 19
    optional_end = 22
    for row in list(range(5, core_end + 1)) + list(range(optional_start, optional_end + 1)):
        ws.cell(row=row, column=8, value=f"=E{row}*F{row}").number_format = USD
        ws.cell(row=row, column=9, value=f"=E{row}*G{row}").number_format = USD
        ws.cell(row=row, column=10, value=f"=H{row}*Summary!$B$10").number_format = TOMAN
        ws.cell(row=row, column=11, value=f"=I{row}*Summary!$B$10").number_format = TOMAN

    def total_row(row: int, label: str, start: int, end: int, fill: str) -> None:
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
        cell = ws.cell(row=row, column=1, value=label)
        cell.font = Font(name="Tahoma", size=10, bold=True, color=NAVY)
        cell.fill = PatternFill("solid", fgColor=fill)
        cell.alignment = Alignment(horizontal="right", vertical="center")
        cell.border = THIN
        ws.cell(row=row, column=8, value=f"=SUM(H{start}:H{end})").number_format = USD
        ws.cell(row=row, column=9, value=f"=SUM(I{start}:I{end})").number_format = USD
        ws.cell(row=row, column=10, value=f"=SUM(J{start}:J{end})").number_format = TOMAN
        ws.cell(row=row, column=11, value=f"=SUM(K{start}:K{end})").number_format = TOMAN
        for col in range(8, 12):
            c = ws.cell(row=row, column=col)
            c.font = BOLD
            c.fill = PatternFill("solid", fgColor=fill)
            c.border = THIN
            c.alignment = Alignment(horizontal="center", vertical="center")

    total_row(18, "جمع Core", 5, core_end, "D9E2F3")
    total_row(23, "جمع اختیاری", optional_start, optional_end, "FFF2CC")
    ws.merge_cells("A24:G24")
    cell = ws["A24"]
    cell.value = "جمع کل پهپاد"
    cell.font = Font(name="Tahoma", size=10, bold=True, color=NAVY)
    cell.fill = PatternFill("solid", fgColor="C6E0B4")
    cell.alignment = Alignment(horizontal="right", vertical="center")
    cell.border = THIN
    ws["H24"] = "=H18+H23"
    ws["I24"] = "=I18+I23"
    ws["J24"] = "=J18+J23"
    ws["K24"] = "=K18+K23"
    for col in range(8, 12):
        c = ws.cell(row=24, column=col)
        c.font = BOLD
        c.fill = PatternFill("solid", fgColor="C6E0B4")
        c.border = THIN
        c.number_format = USD if col in {8, 9} else TOMAN
        c.alignment = Alignment(horizontal="center", vertical="center")
    ws.auto_filter.ref = f"A4:O{optional_end}"

    # GPU
    ws = wb.create_sheet("GPU_Station")
    style_sheet(
        ws,
        "ایستگاه 4090 — سرویس تولید تصویر",
        "FLUX 3 Klein BF16 بدون کوانتیزاسیون؛ 24GB VRAM و 64GB RAM",
        headers,
        [5, 10, 24, 38, 7, 13, 13, 13, 13, 16, 16, 12, 12, 60, 42],
    )
    gpu_rows = []
    for index, item in enumerate(GPU, start=1):
        category, part, model, qty, low, high, availability, priority, reason, note = item
        gpu_rows.append((index, category, part, model, qty, low, high, None, None, None, None, availability, priority, reason, note))
    fill_rows(ws, 5, gpu_rows)
    for row in range(5, 8):
        ws.cell(row=row, column=8, value=f"=E{row}*F{row}").number_format = USD
        ws.cell(row=row, column=9, value=f"=E{row}*G{row}").number_format = USD
        ws.cell(row=row, column=10, value=f"=H{row}*Summary!$B$10").number_format = TOMAN
        ws.cell(row=row, column=11, value=f"=I{row}*Summary!$B$10").number_format = TOMAN
    ws.merge_cells("A8:G8")
    cell = ws["A8"]
    cell.value = "جمع ایستگاه 4090"
    cell.font = Font(name="Tahoma", size=10, bold=True, color=NAVY)
    cell.fill = PatternFill("solid", fgColor="C6E0B4")
    cell.alignment = Alignment(horizontal="right", vertical="center")
    cell.border = THIN
    ws["H8"] = "=SUM(H5:H7)"
    ws["I8"] = "=SUM(I5:I7)"
    ws["J8"] = "=SUM(J5:J7)"
    ws["K8"] = "=SUM(K5:K7)"
    for col in range(8, 12):
        c = ws.cell(row=8, column=col)
        c.font = BOLD
        c.fill = PatternFill("solid", fgColor="C6E0B4")
        c.border = THIN
        c.number_format = USD if col in {8, 9} else TOMAN
        c.alignment = Alignment(horizontal="center", vertical="center")

    # Available
    ws = wb.create_sheet("Available")
    style_sheet(
        ws,
        "موجود — هزینهٔ صفر",
        "چیزهایی که همین حالا در اختیار داریم",
        ["ردیف", "دسته", "قطعه", "تعداد", "وضعیت", "توضیح"],
        [5, 14, 48, 8, 12, 70],
    )
    for index, (category, item, qty, status, note) in enumerate(AVAILABLE, start=1):
        row = 4 + index
        values = [index, category, item, qty, status, note]
        for col, value in enumerate(values, start=1):
            cell = ws.cell(row=row, column=col, value=value)
            cell.font = BODY
            cell.border = THIN
            cell.alignment = Alignment(horizontal="center" if col in {1, 4, 5} else "right", vertical="center", wrap_text=col in {3, 6})
            if index % 2 == 0:
                cell.fill = PatternFill("solid", fgColor=LIGHT)

    # Notes
    ws = wb.create_sheet("Notes")
    style_sheet(
        ws,
        "راهنما و فرضیات",
        "قبل از خرید بخوان",
        ["موضوع", "توضیح"],
        [28, 110],
    )
    notes = [
        ("منبع قیمت", "قیمت‌ها تقریبی و دلاری‌اند (۲۰۲۶). در ایران نرخ آزاد + هزینهٔ واردات معمولاً ۱.۵–۲ برابر می‌کند."),
        ("نرخ ارز", "در Summary سلول B10 را با نرخ روز جایگزین کن؛ ستون‌های تومان خودکار محاسبه می‌شوند."),
        ("4090 و BF16", "وزن‌ها: transformer ۱۱.۱GB + text encoder ۸.۹GB + VAE ۰.۳GB ≈ ۲۰.۳GB. 24GB برای 1024px تنگ است؛ ۷۶۸px یا offload لازم است."),
        ("API vs خرید", "برای ۵۰۰۰km و ۰.۵m GSD: ~۳۸k تصویر؛ API ~$۳۴۳، 4090 ~۵۳h. خرید 4090 برای تولید تکراری/داخلی منطقی است."),
        ("HITL", "PX4 HITL رسمی با Gazebo Classic/jMAVSim است؛ Gazebo Harmonic پشتیبانی رسمی ندارد. SIH گزینهٔ ساده‌تر است."),
        ("دوربین", "برای ۲۰۰m/۱۸۰km/h، global shutter مهم‌تر از رزولوشن است. OV9281 mono برای ناوبری؛ IMX477 برای ضبط."),
        ("Pi", "Pi 5 8GB سریع‌تر است؛ Pi 4B 8GB در ایران راحت‌تر. مدل‌های learned سنگین را روی Pi اجرا نکن."),
        ("ایران", "Pixhawk clones، Pi، Arducam، SiK و PM02 معمولاً موجودند؛ Jetson و global shutter ممکن است وارداتی باشند."),
        ("Safety", "HITL/پرواز واقعی: پروانه‌ها را جدا کن، kill switch و geofence داشته باش، و از باتری/ESCs مناسب استفاده کن."),
    ]
    for index, (topic, text) in enumerate(notes, start=5):
        a = ws.cell(row=index, column=1, value=topic)
        b = ws.cell(row=index, column=2, value=text)
        a.font = BOLD
        b.font = BODY
        a.border = THIN
        b.border = THIN
        a.alignment = Alignment(horizontal="right", vertical="center")
        b.alignment = Alignment(horizontal="right", vertical="center", wrap_text=True)
        ws.row_dimensions[index].height = 38
        if index % 2 == 0:
            a.fill = PatternFill("solid", fgColor=LIGHT)
            b.fill = PatternFill("solid", fgColor=LIGHT)

    for sheet in wb.worksheets:
        sheet.page_setup.orientation = "landscape"
        sheet.page_setup.fitToWidth = 1
        sheet.sheet_properties.pageSetUpPr.fitToPage = True
        sheet.print_options.horizontalCentered = True

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
