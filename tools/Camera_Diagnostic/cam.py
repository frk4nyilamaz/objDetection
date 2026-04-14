#!/usr/bin/env python3
import argparse
import csv
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Yaygın V4L2 pixel formatları için nominal bpp.
# Sıkıştırılmış formatlarda bitrate değişken olduğu için None bırakıldı.
BPP_MAP = {
    "YUYV": 16,
    "UYVY": 16,
    "YVYU": 16,
    "VYUY": 16,
    "NV12": 12,
    "NV21": 12,
    "YU12": 12,   # I420 / YUV420 planar
    "YV12": 12,
    "GREY": 8,
    "Y8I": 16,
    "Z16 ": 16,
    "RGB3": 24,
    "BGR3": 24,
    "RGBP": 16,   # RGB565
    "BA81": 8,    # Bayer BGGR8
    "RGGB": 8,
    "GBRG": 8,
    "GRBG": 8,
    "MJPG": None,
    "JPEG": None,
    "H264": None,
    "HEVC": None,
    "MPG2": None,
}

COMPRESSED_FORMATS = {"MJPG", "JPEG", "H264", "HEVC", "MPG2"}


@dataclass
class ModeRow:
    fourcc: str
    description: str
    compressed: str
    width: int
    height: int
    megapixels: float
    nominal_bpp: str
    fps: str
    frame_interval_s: str
    estimated_raw_bw_mbps: str
    estimated_raw_bw_MBps: str
    notes: str


def run_cmd(cmd: List[str], check: bool = True) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Komut hata verdi: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout


def require_cmd(name: str) -> None:
    if shutil.which(name) is None:
        print(
            f"Hata: '{name}' bulunamadı.\n"
            f"Kurulum örneği:\n"
            f"  sudo apt install v4l-utils",
            file=sys.stderr,
        )
        sys.exit(1)


def read_text_file(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return None


def safe_float_text(value: float, ndigits: int = 3) -> str:
    return f"{value:.{ndigits}f}"


def pretty_bool(flag: bool) -> str:
    return "yes" if flag else "no"


def device_name_only(device: str) -> str:
    return Path(device).name


def find_usb_root_from_video_device(video_device: str) -> Optional[Path]:
    sys_path = Path("/sys/class/video4linux") / device_name_only(video_device) / "device"
    if not sys_path.exists():
        return None

    try:
        cur = sys_path.resolve()
    except Exception:
        cur = sys_path

    visited = set()
    while True:
        if cur in visited:
            return None
        visited.add(cur)

        if (cur / "idVendor").exists() and (cur / "idProduct").exists():
            return cur

        if cur.parent == cur:
            return None
        cur = cur.parent


def get_usb_info(video_device: str) -> Dict[str, str]:
    usb_root = find_usb_root_from_video_device(video_device)
    info = {
        "usb_vendor_id": "",
        "usb_product_id": "",
        "usb_manufacturer": "",
        "usb_product": "",
        "usb_serial": "",
        "usb_speed_mbps": "",
        "usb_version": "",
        "usb_busnum": "",
        "usb_devnum": "",
    }
    if not usb_root:
        return info

    mapping = {
        "usb_vendor_id": "idVendor",
        "usb_product_id": "idProduct",
        "usb_manufacturer": "manufacturer",
        "usb_product": "product",
        "usb_serial": "serial",
        "usb_speed_mbps": "speed",
        "usb_version": "version",
        "usb_busnum": "busnum",
        "usb_devnum": "devnum",
    }
    for dst, src in mapping.items():
        val = read_text_file(usb_root / src)
        info[dst] = val or ""
    return info


def parse_key_value_block(text: str) -> Dict[str, str]:
    out = {}
    for line in text.splitlines():
        if ":" in line:
            left, right = line.split(":", 1)
            key = left.strip().lower().replace(" ", "_").replace("/", "_")
            out[key] = right.strip()
    return out


def get_driver_info(video_device: str) -> Dict[str, str]:
    txt = run_cmd(["v4l2-ctl", "-D", "-d", video_device])
    info = parse_key_value_block(txt)
    return {
        "driver_name": info.get("driver_name", ""),
        "card_type": info.get("card_type", ""),
        "bus_info": info.get("bus_info", ""),
        "driver_version": info.get("driver_version", ""),
        "capabilities": info.get("capabilities", ""),
        "device_caps": info.get("device_caps", ""),
    }


def get_current_format(video_device: str) -> Dict[str, str]:
    info = {}
    try:
        txt = run_cmd(["v4l2-ctl", "--get-fmt-video", "-d", video_device])
        info.update(parse_key_value_block(txt))
    except Exception:
        pass

    try:
        txt = run_cmd(["v4l2-ctl", "--get-parm", "-d", video_device])
        m = re.search(r"Frames per second:\s*([0-9.]+)", txt)
        if m:
            info["frames_per_second"] = m.group(1)
    except Exception:
        pass

    return info


def parse_list_formats_ext(text: str) -> List[ModeRow]:
    fmt_re = re.compile(r"^\s*\[(\d+)\]:\s*'([^']+)'\s*\((.+)\)\s*$")
    size_re = re.compile(r"^\s*Size:\s*(Discrete|Stepwise|Continuous)\s+(\d+)x(\d+)(.*)$")
    interval_discrete_re = re.compile(
        r"^\s*Interval:\s*Discrete\s*([0-9.]+)s\s*\(([0-9.]+)\s*fps\)\s*$"
    )
    interval_generic_re = re.compile(r"^\s*Interval:\s*(.+)\s*$")

    rows: List[ModeRow] = []
    current_fmt = None
    current_desc = None
    current_w = None
    current_h = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")

        m = fmt_re.match(line)
        if m:
            current_fmt = m.group(2)
            current_desc = m.group(3)
            current_w = None
            current_h = None
            continue

        m = size_re.match(line)
        if m and current_fmt:
            current_w = int(m.group(2))
            current_h = int(m.group(3))
            continue

        m = interval_discrete_re.match(line)
        if m and current_fmt and current_w and current_h:
            interval_s = float(m.group(1))
            fps = float(m.group(2))
            bpp = BPP_MAP.get(current_fmt)
            mp = (current_w * current_h) / 1_000_000.0

            if bpp is not None:
                raw_bw_mbps = (current_w * current_h * fps * bpp) / 1_000_000.0
                raw_bw_MBps = raw_bw_mbps / 8.0
                bw_mbps_text = safe_float_text(raw_bw_mbps, 3)
                bw_MBps_text = safe_float_text(raw_bw_MBps, 3)
                bpp_text = str(bpp)
                notes = "raw estimate"
            else:
                bw_mbps_text = "variable"
                bw_MBps_text = "variable"
                bpp_text = "variable"
                notes = "compressed or driver-specific"

            rows.append(
                ModeRow(
                    fourcc=current_fmt,
                    description=current_desc or "",
                    compressed=pretty_bool(current_fmt in COMPRESSED_FORMATS),
                    width=current_w,
                    height=current_h,
                    megapixels=round(mp, 3),
                    nominal_bpp=bpp_text,
                    fps=safe_float_text(fps, 3),
                    frame_interval_s=safe_float_text(interval_s, 6),
                    estimated_raw_bw_mbps=bw_mbps_text,
                    estimated_raw_bw_MBps=bw_MBps_text,
                    notes=notes,
                )
            )
            continue

        m = interval_generic_re.match(line)
        if m and current_fmt and current_w and current_h:
            spec = m.group(1).strip()
            bpp = BPP_MAP.get(current_fmt)
            mp = (current_w * current_h) / 1_000_000.0

            rows.append(
                ModeRow(
                    fourcc=current_fmt,
                    description=current_desc or "",
                    compressed=pretty_bool(current_fmt in COMPRESSED_FORMATS),
                    width=current_w,
                    height=current_h,
                    megapixels=round(mp, 3),
                    nominal_bpp=str(bpp) if bpp is not None else "variable",
                    fps="see_spec",
                    frame_interval_s=spec,
                    estimated_raw_bw_mbps="n/a",
                    estimated_raw_bw_MBps="n/a",
                    notes="non-discrete interval spec",
                )
            )

    return rows


def get_mode_rows(video_device: str) -> List[ModeRow]:
    txt = run_cmd(["v4l2-ctl", "--list-formats-ext", "-d", video_device])
    rows = parse_list_formats_ext(txt)
    if not rows:
        raise RuntimeError("Mod listesi parse edilemedi. v4l2-ctl çıktısını kontrol et.")
    return rows


def print_table(headers: List[str], rows: List[List[object]]) -> None:
    str_rows = [[("" if v is None else str(v)) for v in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in str_rows:
        print(fmt(row))


def write_csv(path: Path, headers: List[str], rows: List[List[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def write_markdown_report(
    path: Path,
    device: str,
    summary_items: List[Tuple[str, str]],
    mode_headers: List[str],
    mode_rows: List[List[object]],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# USB Camera Diagnostic Report\n\n")
        f.write(f"- Device: `{device}`\n")
        f.write(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`\n\n")
        f.write("## Summary\n\n")
        for k, v in summary_items:
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Modes\n\n")
        f.write("| " + " | ".join(mode_headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(mode_headers)) + " |\n")
        for row in mode_rows:
            vals = [str(x).replace("|", "/") for x in row]
            f.write("| " + " | ".join(vals) + " |\n")


def main():
    parser = argparse.ArgumentParser(
        description="USB kamera diagnostik aracı: format, çözünürlük, FPS, MP, bpp ve bandwidth özeti üretir."
    )
    parser.add_argument(
        "-d", "--device", default="/dev/video0", help="Kamera cihazı, örn: /dev/video0"
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        default=None,
        help="Çıktı dosyaları için prefix. Verilmezse zaman damgalı otomatik isim kullanılır.",
    )
    args = parser.parse_args()

    require_cmd("v4l2-ctl")

    if not Path(args.device).exists():
        print(f"Hata: cihaz bulunamadı: {args.device}", file=sys.stderr)
        sys.exit(1)

    prefix = args.output_prefix
    if not prefix:
        prefix = f"camera_diag_{device_name_only(args.device)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    driver = get_driver_info(args.device)
    usb = get_usb_info(args.device)
    current = get_current_format(args.device)
    mode_rows_objs = get_mode_rows(args.device)

    unique_formats = sorted({r.fourcc for r in mode_rows_objs})
    max_mode = max(
        mode_rows_objs,
        key=lambda r: (
            r.width * r.height,
            float(r.fps) if r.fps.replace(".", "", 1).isdigit() else 0,
        ),
    )
    max_res_text = f"{max_mode.width}x{max_mode.height}"
    max_mp_text = safe_float_text(max_mode.megapixels, 3)

    current_fmt = current.get("pixel_format", "")
    current_wh = ""
    if current.get("width_height"):
        current_wh = current["width_height"].replace("/", "x")
    elif current.get("width") and current.get("height"):
        current_wh = f"{current['width']}x{current['height']}"
    current_fps = current.get("frames_per_second", "")

    summary = [
        ("device", args.device),
        ("driver_name", driver.get("driver_name", "")),
        ("card_type", driver.get("card_type", "")),
        ("bus_info", driver.get("bus_info", "")),
        ("driver_version", driver.get("driver_version", "")),
        ("capabilities", driver.get("capabilities", "")),
        ("device_caps", driver.get("device_caps", "")),
        ("usb_vendor_id", usb.get("usb_vendor_id", "")),
        ("usb_product_id", usb.get("usb_product_id", "")),
        ("usb_manufacturer", usb.get("usb_manufacturer", "")),
        ("usb_product", usb.get("usb_product", "")),
        ("usb_serial", usb.get("usb_serial", "")),
        ("usb_speed_mbps", usb.get("usb_speed_mbps", "")),
        ("usb_version", usb.get("usb_version", "")),
        ("current_resolution", current_wh),
        ("current_pixel_format", current_fmt),
        ("current_fps", current_fps),
        ("advertised_max_resolution", max_res_text),
        ("advertised_max_megapixels", max_mp_text),
        ("unique_pixel_formats", ", ".join(unique_formats)),
        (
            "notes",
            "Advertised max megapixels = listed mode resolution area / 1e6; this is not always the physical sensor MP.",
        ),
    ]

    mode_headers = [
        "fourcc",
        "description",
        "compressed",
        "width",
        "height",
        "megapixels",
        "nominal_bpp",
        "fps",
        "frame_interval_s",
        "estimated_raw_bw_mbps",
        "estimated_raw_bw_MBps",
        "notes",
    ]
    mode_rows = [
        [
            r.fourcc,
            r.description,
            r.compressed,
            r.width,
            r.height,
            safe_float_text(r.megapixels, 3),
            r.nominal_bpp,
            r.fps,
            r.frame_interval_s,
            r.estimated_raw_bw_mbps,
            r.estimated_raw_bw_MBps,
            r.notes,
        ]
        for r in mode_rows_objs
    ]

    summary_headers = ["field", "value"]
    summary_rows = [[k, v] for k, v in summary]

    summary_csv = Path(f"{prefix}_summary.csv")
    modes_csv = Path(f"{prefix}_modes.csv")
    report_md = Path(f"{prefix}_report.md")

    write_csv(summary_csv, summary_headers, summary_rows)
    write_csv(modes_csv, mode_headers, mode_rows)
    write_markdown_report(report_md, args.device, summary, mode_headers, mode_rows)

    print("\n=== CAMERA SUMMARY ===")
    print_table(summary_headers, summary_rows)

    print("\n=== CAMERA MODES ===")
    print_table(mode_headers, mode_rows)

    print("\nYazılan dosyalar:")
    print(f"  - {summary_csv}")
    print(f"  - {modes_csv}")
    print(f"  - {report_md}")


if __name__ == "__main__":
    main()