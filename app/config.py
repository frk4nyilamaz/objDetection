from pathlib import Path

PROJECT_ROOT = Path("/mnt/sda1/furkan/objectDetectionProject")
SNAPSHOT_DIR = PROJECT_ROOT / "outputs" / "snapshots"
RECORDINGS_DIR = PROJECT_ROOT / "outputs" / "recordings"
SCRCPY_BINARY_PATH = Path.home() / "tools" / "scrcpy" / "scrcpy-linux-x86_64-v3.3.4" / "scrcpy"

MODEL_DIR = PROJECT_ROOT / "models"
YOLOV8N_PATH = MODEL_DIR / "yolov8n.pt"
YOLOV8S_PATH = MODEL_DIR / "yolov8s.pt"
FONT_PATH = PROJECT_ROOT / "assets" / "fonts" / "FiraSans-Black.ttf"
DEFAULT_USB_CAMERA_DEVICE = "/dev/video0"
DEFAULT_ANDROID_V4L2_DEVICE = "/dev/video2"

CAMERA_SOURCE_TYPES = [
    {"id": "1", "key": "usb", "label": "USB Camera"},
    {"id": "2", "key": "android", "label": "Android Phone Camera"},
]

CAMERA_MODES = [
    {"id": "info", "label" : "For best, choose 3. Because our camera's output shows that at 1024x768, both formats support 30 FPS. However, MJPG is much \"lighter\" on the USB bus "},
    {"id": "1", "label": "640x480 @30 MJPG", "width": 640, "height": 480, "fps": 30, "pixel_format": "MJPG"},
    {"id": "2", "label": "800x600 @30 MJPG", "width": 800, "height": 600, "fps": 30, "pixel_format": "MJPG"},
    {"id": "3", "label": "1024x768 @30 MJPG", "width": 1024, "height": 768, "fps": 30, "pixel_format": "MJPG"},
    {"id": "4", "label": "640x480 @30 YUYV", "width": 640, "height": 480, "fps": 30, "pixel_format": "YUYV"},
    {"id": "5", "label": "800x600 @30 YUYV", "width": 800, "height": 600, "fps": 30, "pixel_format": "YUYV"},
    {"id": "6", "label": "1024x768 @10 YUYV", "width": 1024, "height": 768, "fps": 10, "pixel_format": "YUYV"},
]

ANDROID_CAMERA_PROFILES = [
    {"id": "1", "label": "Back 1920x1080 @30", "device": "/dev/video2", "camera_index": 2, "camera_id": 0, "width": 1920, "height": 1080, "fps": 30},
    {"id": "2", "label": "Back 1280x720 @30", "device": "/dev/video2", "camera_index": 2, "camera_id": 0, "width": 1280, "height": 720, "fps": 30},
    {"id": "3", "label": "Front 1920x1080 @30", "device": "/dev/video2", "camera_index": 2, "camera_id": 1, "width": 1920, "height": 1080, "fps": 30},
    {"id": "4", "label": "Front 1280x720 @30", "device": "/dev/video2", "camera_index": 2, "camera_id": 1, "width": 1280, "height": 720, "fps": 30},
]


def get_camera_source_type_by_id(source_id: str) -> dict:
    for source in CAMERA_SOURCE_TYPES:
        if source["id"] == source_id:
            return source
    raise ValueError(f"Gecersiz kamera kaynagi id: {source_id}")


def get_mode_by_id(mode_id: str) -> dict:
    for mode in CAMERA_MODES:
        if mode["id"] == mode_id:
            return mode
    raise ValueError(f"Gecersiz mode id: {mode_id}")


def get_android_profile_by_id(profile_id: str) -> dict:
    for profile in ANDROID_CAMERA_PROFILES:
        if profile["id"] == profile_id:
            return profile
    raise ValueError(f"Gecersiz Android kamera profili id: {profile_id}")


def build_gst_pipeline(device: str, mode: dict) -> str:
    width = int(mode["width"])
    height = int(mode["height"])
    fps = int(mode["fps"])
    pixel_format = str(mode["pixel_format"]).upper()

    if pixel_format == "MJPG":
        return (
            f"v4l2src device={device} ! "
            f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
            "jpegdec ! "
            "videoconvert ! "
            "appsink drop=true sync=false"
        )

    if pixel_format == "YUYV":
        return (
            f"v4l2src device={device} ! "
            f"video/x-raw,format=YUY2,width={width},height={height},framerate={fps}/1 ! "
            "videoconvert ! "
            "appsink drop=true sync=false"
        )

    raise ValueError(f"Desteklenmeyen pixel format: {pixel_format}")


def build_android_gst_pipeline(device: str) -> str:
    return (
        f"v4l2src device={device} ! "
        "videoconvert ! "
        "appsink drop=true sync=false"
    )

def build_android_scrcpy_command(mode: dict, sink_device: str) -> list[str]:
    width = int(mode["width"])
    height = int(mode["height"])
    fps = int(mode["fps"])
    camera_id = int(mode["camera_id"])

    return [
        str(SCRCPY_BINARY_PATH),
        "--video-source=camera",
        f"--camera-id={camera_id}",
        f"--camera-size={width}x{height}",
        f"--camera-fps={fps}",
        f"--v4l2-sink={sink_device}",
        "--no-playback",
        "--no-audio",
    ]

#yapılacaklar: aynısnı birde ffmpeg ile yapacağız. Seçeneğe göre ffmpeg veya gstreamer.
"""

640*480*24; bandwidth 
bu kameranın tablosunı, bandwith. Mp, Mp/s gibi değerleri. 

colormode flag'i bak. Bir piksel kaç bit tutuluyor ? 

kameranın resolutionlarına göre verebielceği fps değerlerini veren -> hem ffmpeg hem gstreamer ...
...bunu yapabiliyor 

İleriye dönük: Kullanıcının GUI üzerinden seçim uyapabileceği alanlar eklenebilir. 
GUI'ye bulaşma, txt tabanlı. 




Kameranın teknik verilerine dayanarak yapılan temel hesaplamalar aşağıdadır:
## 1. Kare Başına Veri (Frame Size)
Bir adet görüntünün ham veri boyutudur.

* Formül: Genişlik × Yükseklik × Bit Derinliği
* Hesap: $640 \times 480 \times 24 \text{ bit} = 7.372.800 \text{ bit}$
* Byte Cinsinden: $921.600 \text{ Byte}$ (~900 KB)

## 2. Veri Akış Hızı (Bitrate)
Saniyede taşınan veya kaydedilen veri miktarıdır.

* Formül: Kare Başına Veri × FPS
* Hesap: $7.372.800 \times 30 = 221.184.000 \text{ bps}$
* Megabit Cinsinden: 221,18 Mbps
* Megabyte Cinsinden: 27,65 MB/s

## 3. Depolama Gereksinimi
Kayıt süresine göre ihtiyaç duyulan alan (Örn: 1 Dakika).

* Dakikalık Hesap: $27,65 \text{ MB/s} \times 60 \text{ saniye} = 1.659 \text{ MB}$
* Sonuç: Dakikada yaklaşık 1,62 GB yer kaplar.

## 4. Piksel Sayısı (Resolution)

* Hesap: $640 \times 480 = 307.200 \text{ piksel}$
* Sonuç: 0,3 Megapiksel (VGA Çözünürlük)

## 5. Bant Genişliği İhtiyacı

* İletim: Sıkıştırma olmadığı için veri hattının (USB, Ethernet vb.) net 221 Mbps hızını kayıpsız desteklemesi gerekir.

Bu hesaplamaları belirli bir kayıt süresi veya depolama kapasitesi (örneğin 1 TB diske ne kadar kayıt sığar) üzerinden detaylandırmamı ister misiniz?





"""
