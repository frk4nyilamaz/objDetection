V1.0
# Object Detection Project

Bu proje, harici USB kamera üzerinden canlı görüntü alıp nesne tespiti yapmak için hazırlanıyor.

## Kullanılan Temel Bileşenler
- Python
- OpenCV
- YOLOv8
- CUDA
- GStreamer / alternatif kamera açılış yöntemi

## Mevcut Durum
Şu anda proje başlangıç aşamasında.
İlk hedef, kamerayı farklı backend seçenekleriyle açıp görüntü akışını doğrulamaktır.

 
source .venv/bin/activate ile sanala gir


V1.1
# Object Detection Project

Bu proje, harici USB kamera ile canlı görüntü alıp YOLOv8 modeli üzerinden nesne tespiti yapmak için hazırlanıyor.

## Kullanılan Yapı
- Python
- OpenCV
- YOLOv8
- CUDA
- GStreamer / direkt device açılışı

## Mevcut Özellikler
- Kamera backend seçimi
- Model seçimi (`YOLOv8n`, `YOLOv8s`, compare mode)
- Kamera mode seçimi (çözünürlük / fps / format)
- Canlı detection
- Ekranda ortalama FPS gösterimi


V1.2
# Object Detection Project

Bu proje, harici USB kamera ile canlı görüntü alıp YOLOv8 modeli üzerinden nesne tespiti yapmak için hazırlanıyor.

## Kullanılan Yapı
- Python
- OpenCV
- YOLOv8
- CUDA / CPU
- GStreamer / direkt device açılışı

## Mevcut Özellikler
- Kamera backend seçimi
- Model seçimi (`YOLOv8n`, `YOLOv8s`, compare mode)
- Kamera mode seçimi (çözünürlük / fps / format)
- Canlı detection
- Ortalama FPS gösterimi
- Engine değiştirme (`CPU / GPU`)
- Bilgi panelini açıp kapatma
- Tuş yardım menüsünü açıp kapatma
- Snapshot alma
- Detection kutularını açıp kapatma

## Çalıştırma

Sanal ortamı aktif ettikten sonra:

```bash
python app/mainCamTest.py


09/04/2026 Yeni Görev: 
-Dil Seçeneği Eklenecek
    -> Labellar, Menu seçenekleri(ops) vs.


V 1.3 
# Object Detection Project

Bu proje, harici USB kamera ile canlı görüntü alıp YOLOv8 modeli üzerinden gerçek zamanlı nesne tespiti yapmak için hazırlanıyor.

## Kullanılan Yapı
- Python
- OpenCV
- YOLOv8
- CUDA / CPU
- GStreamer / direkt device açılışı
- FreeType tabanlı metin gösterimi

## Mevcut Özellikler
- Kamera backend seçimi
- Model seçimi (`YOLOv8n`, `YOLOv8s`, compare mode)
- Kamera mode seçimi (çözünürlük / fps / format)
- Canlı detection
- Ortalama FPS gösterimi
- Engine değiştirme (`CPU / GPU`)
- Bilgi panelini açıp kapatma
- Tuş yardım menüsünü açıp kapatma
- Snapshot alma
- Detection kutularını açıp kapatma
- Türkçe karakter desteği
- Tespit edilen objeleri `labels/source.json` içinde kayıt altında tutma
- Mevcut Türkçe karşılıkları `labels/tr.json` üzerinden gösterme

## Proje Yapısı
- `app/` -> ana uygulama dosyaları
- `tools/` -> yardımcı kayıt / locale araçları
- `models/` -> model dosyaları
- `labels/` -> obje kayıtları ve dil dosyaları
- `outputs/` -> çıktı dosyaları ve snapshotlar

## Çalıştırma

Önce sanal ortamı aktif et:

```bash
source .venv/bin/activate

Sonra proje kök dizininden çalıştır:
python -m app.mainCamTest

## Mevcut Dil Mantığı

Varsayılan dil İngilizce
İngilizce modda YOLO’dan gelen etiket doğrudan gösterilir
Türkçe modda labels/tr.json içindeki kayıtlı karşılıklar kullanılır
Türkçe karşılığı olmayan yeni objeler labels/source.json içinde kayıt altına alınır
Çeviri tarafı şimdilik manuel yönetilmektedir