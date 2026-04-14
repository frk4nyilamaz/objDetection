from pathlib import Path

PROJECT_ROOT = Path("/mnt/sda1/furkan/objectDetectionProject")
SNAPSHOT_DIR = PROJECT_ROOT / "outputs" / "snapshots"

MODEL_DIR = PROJECT_ROOT / "models"
YOLOV8N_PATH = MODEL_DIR / "yolov8n.pt"
YOLOV8S_PATH = MODEL_DIR / "yolov8s.pt"

CAMERA_MODES = [
    {"id": "info", "label" : "For best, choose 3. Because our camera's output shows that at 1024x768, both formats support 30 FPS. However, MJPG is much \"lighter\" on the USB bus "},
    {"id": "1", "label": "640x480 @30 MJPG", "width": 640, "height": 480, "fps": 30, "pixel_format": "MJPG"},
    {"id": "2", "label": "800x600 @30 MJPG", "width": 800, "height": 600, "fps": 30, "pixel_format": "MJPG"},
    {"id": "3", "label": "1024x768 @30 MJPG", "width": 1024, "height": 768, "fps": 30, "pixel_format": "MJPG"},
    {"id": "4", "label": "640x480 @30 YUYV", "width": 640, "height": 480, "fps": 30, "pixel_format": "YUYV"},
    {"id": "5", "label": "800x600 @30 YUYV", "width": 800, "height": 600, "fps": 30, "pixel_format": "YUYV"},
    {"id": "6", "label": "1024x768 @10 YUYV", "width": 1024, "height": 768, "fps": 10, "pixel_format": "YUYV"},
]


def get_mode_by_id(mode_id: str) -> dict:
    for mode in CAMERA_MODES:
        if mode["id"] == mode_id:
            return mode
    raise ValueError(f"Gecersiz mode id: {mode_id}")


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