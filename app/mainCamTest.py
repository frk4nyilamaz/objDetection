import time
from collections import deque
from datetime import datetime

import cv2
import torch
from ultralytics import YOLO

from app.config import (
    CAMERA_SOURCE_TYPES,
    CAMERA_MODES,
    ANDROID_CAMERA_PROFILES,
    SNAPSHOT_DIR,
    RECORDINGS_DIR,
    PROJECT_ROOT,
    YOLOV8N_PATH,
    FONT_PATH,
    YOLOV8S_PATH,
    DEFAULT_USB_CAMERA_DEVICE,
    get_camera_source_type_by_id,
    get_android_profile_by_id,
    get_mode_by_id,
    build_android_gst_pipeline,
    build_gst_pipeline,
)

from app.camera import build_android_camera, build_camera, CameraError
from app.text_renderer import TextRenderer
from tools.label_registry import LabelRegistry
from tools.label_locale_store import LabelLocaleStore


renderer = TextRenderer(str(FONT_PATH))


def get_user_camera_source_choice() -> dict:
    print("\nKamera kaynagi secimi yapin:")
    for source in CAMERA_SOURCE_TYPES:
        print(f"{source['id']} - {source['label']}")

    user_input = input("Seciminiz (1/2): ").strip()

    try:
        return get_camera_source_type_by_id(user_input)
    except ValueError:
        print("Gecersiz secim yapildi. Varsayilan olarak USB Camera secildi.")
        return get_camera_source_type_by_id("1")


def get_user_backend_choice() -> str:
    print("\nKamera backend: ")
    print("1 - GStreamer")
    print("2 - FFmpeg benzeri direkt device acilisi")
    user_input = input("Seciminiz (1/2): ").strip()

    if user_input == "1":
        return "gstreamer"
    if user_input == "2":
        return "ffmpeg"

    print("Gecersiz secim yapildi. Varsayilan olarak GStreamer secildi.")
    return "gstreamer"


def get_user_model_choice() -> str:
    print("\nModel secimi yapin:")
    print("1 - YOLOv8n")
    print("2 - YOLOv8s")
    print("3 - Compare mode (n ve s birlikte)")
    user_input = input("Seciminiz (1/2/3): ").strip()

    if user_input == "1":
        return "n"
    if user_input == "2":
        return "s"
    if user_input == "3":
        return "compare"

    print("Gecersiz secim yapildi. Varsayilan olarak YOLOv8s secildi.")
    return "s"


def get_user_mode_choice() -> dict:
    print("\nKamera mode secimi yapin:")
    for mode in CAMERA_MODES:
        print(f"{mode['id']} - {mode['label']}")

    user_input = input("Seciminiz: ").strip()

    try:
        return get_mode_by_id(user_input)
    except ValueError:
        print("Gecersiz secim yapildi. Varsayilan olarak 640x480 @30 MJPG secildi.")
        return get_mode_by_id("1")


def get_user_android_profile_choice() -> dict:
    print("\nAndroid kamera profili secimi yapin:")
    for profile in ANDROID_CAMERA_PROFILES:
        print(f"{profile['id']} - {profile['label']}")

    user_input = input("Seciminiz: ").strip()

    try:
        return get_android_profile_by_id(user_input)
    except ValueError:
        print("Gecersiz secim yapildi. Varsayilan olarak Back 1920x1080 @30 secildi.")
        return get_android_profile_by_id("1")


def draw_text_block(frame, lines, origin_x, origin_y, font_height=22):
    thickness = 1
    line_height = font_height + 8
    padding = 10

    max_width = 0
    for line in lines:
        (w, _), _ = renderer.get_text_size(line, font_height, thickness)
        max_width = max(max_width, w)

    block_width = max_width + padding * 2
    block_height = len(lines) * line_height + padding * 2

    x1 = origin_x
    y1 = origin_y
    x2 = origin_x + block_width
    y2 = origin_y + block_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    y = origin_y + padding + font_height
    for line in lines:
        renderer.put_text(
            frame,
            line,
            (origin_x + padding, y),
            font_height,
            (255, 255, 255),
            thickness,
        )
        y += line_height

    return frame


def draw_info_panel(frame, source_label, backend_choice, model_name, selected_mode, current_engine, language_mode, avg_fps):
    info_lines = [
        f"Source    : {source_label}",
        f"Model     : {model_name}",
        f"Backend   : {backend_choice}",
        f"Mode      : {selected_mode['label']}",
        f"Engine    : {current_engine.upper()}",
        f"Language  : {language_mode.upper()}",
        f"FPS(avg)  : {avg_fps:.2f}",
    ]
    return draw_text_block(frame, info_lines, 10, 10, font_height=22)


def draw_menu_panel(frame, menu_visible):
    if menu_visible:
        menu_lines = [
            "Keys",
            "i : info on/off",
            "m : menu on/off",
            "e : engine cpu/gpu",
            "l : language en/tr",
            "p : snapshot",
            "x : boxes on/off",
            "r : record video",
            "q : quit",
        ]
        box_height = len(menu_lines) * (20 + 8) + 20
        y = frame.shape[0] - box_height - 10
        return draw_text_block(frame, menu_lines, 10, y, font_height=20)

    hint_lines = ["m: menu"]
    y = frame.shape[0] - 50
    return draw_text_block(frame, hint_lines, 10, y, font_height=18)


def resolve_display_label(
    raw_label: str,
    language_mode: str,
    registry: LabelRegistry,
    tr_store: LabelLocaleStore,
) -> str:
    entry = registry.get_or_create_label(raw_label)
    label_id = entry["id"]

    if language_mode == "en":
        return raw_label

    if language_mode == "tr":
        tr_value = tr_store.get_translation(label_id)
        if tr_value:
            return tr_value

        tr_store.mark_pending(label_id, raw_label)
        return "..."

    return raw_label


def register_detected_labels(results, registry: LabelRegistry) -> None:
    result = results[0]
    names = result.names

    if result.boxes is None or len(result.boxes) == 0:
        return

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        raw_label = names[cls_id]
        registry.get_or_create_label(raw_label)


def draw_detections(frame, results, language_mode, registry, tr_store):
    result = results[0]
    names = result.names

    if result.boxes is None or len(result.boxes) == 0:
        return frame

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf_percent = float(box.conf[0].item()) * 100.0
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        raw_label = names[cls_id]
        display_label = resolve_display_label(raw_label, language_mode, registry, tr_store)

        label = f"{display_label} {conf_percent:.1f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        font_height = 20
        text_thickness = 1
        (text_w, text_h), _ = renderer.get_text_size(label, font_height, text_thickness)

        text_x1 = x1
        text_y1 = max(y1 - text_h - 14, 0)
        text_x2 = x1 + text_w + 12
        text_y2 = max(y1, text_h + 14)

        cv2.rectangle(frame, (text_x1, text_y1), (text_x2, text_y2), (0, 255, 0), -1)

        renderer.put_text(
            frame,
            label,
            (x1 + 6, text_y2 - 6),
            font_height,
            (0, 0, 0),
            text_thickness,
        )

    return frame


def save_snapshot(frame):
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = SNAPSHOT_DIR / f"snapshot_{timestamp}.jpg"
    success = cv2.imwrite(str(filename), frame)

    if success:
        print(f"[SNAPSHOT] Kaydedildi: {filename}")
    else:
        print("[SNAPSHOT] Kaydetme basarisiz.")

def draw_recording_badge(frame):
    cv2.circle(frame, (25, 25), 8, (0, 0, 255), -1)
    renderer.put_text(frame, "REC", (40, 32), 20, (0, 0, 255), 1)
    return frame

def create_video_writer(record_frame, selected_mode):
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RECORDINGS_DIR / f"record_{timestamp}.avi"

    height, width = record_frame.shape[:2]

    # Düşük CPU yükü için pratik seçim: MJPG + AVI
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    output_fps = float(selected_mode["fps"]) if selected_mode.get("fps") else 20.0
    output_fps = max(1.0, min(output_fps, 30.0))

    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError("VideoWriter acilamadi.")

    print(f"[RECORD] Kayit basladi: {output_path}")
    print(f"[RECORD] Boyut: {width}x{height}, FPS: {output_fps:.1f}")
    return writer, output_path


def stop_video_writer(video_writer, recording_path):
    if video_writer is not None:
        video_writer.release()
        print(f"[RECORD] Kayit durduruldu: {recording_path}")


def predict_and_render(model, frame, engine, boxes_visible, language_mode, registry, tr_store):
    results = model.predict(
        source=frame,
        device=engine,
        conf=0.35,
        verbose=False,
    )

    register_detected_labels(results, registry)

    rendered = frame.copy()

    if boxes_visible:
        rendered = draw_detections(rendered, results, language_mode, registry, tr_store)

    return rendered


def open_camera_with_retry(
    camera,
    attempts=15,
    delay_seconds=0.5,
    validate_frame=False,
):
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            camera.open()

            if validate_frame:
                for _ in range(5):
                    try:
                        camera.read()
                        return
                    except CameraError as exc:
                        last_error = exc
                        time.sleep(0.1)

                camera.release()
                continue

            return
        except CameraError as exc:
            last_error = exc
            camera.release()
            if attempt < attempts:
                time.sleep(delay_seconds)

    if last_error is not None:
        raise last_error


def main():
    camera_source = get_user_camera_source_choice()
    backend_choice = ""
    selected_mode = None
    device_name = DEFAULT_USB_CAMERA_DEVICE
    base_window_name = "CAM-TEST"

    current_engine = "cuda" if torch.cuda.is_available() else "cpu"
    language_mode = "en"
    info_visible = True
    menu_visible = True
    boxes_visible = True
    recording = False
    video_writer = None
    recording_path = None
    camera = None
    source_label = camera_source["key"].upper()

    registry = LabelRegistry(PROJECT_ROOT)
    tr_store = LabelLocaleStore(PROJECT_ROOT, "tr")

    if camera_source["key"] == "usb":
        backend_choice = get_user_backend_choice()
        selected_mode = get_user_mode_choice()
        device_name = DEFAULT_USB_CAMERA_DEVICE
        camera = build_camera(
            backend_choice=backend_choice,
            device=device_name,
            mode=selected_mode,
        )
    else:
        backend_choice = "gstreamer"
        selected_profile = get_user_android_profile_choice()
        device_name = selected_profile["device"]
        selected_mode = {
            "label": selected_profile["label"],
            "width": selected_profile["width"],
            "height": selected_profile["height"],
            "fps": selected_profile["fps"],
            "pixel_format": "YUYV",
        }
        camera = build_android_camera(
            device=device_name,
            mode=selected_mode,
        )

    model_choice = get_user_model_choice()

    model_n = None
    model_s = None

    if model_choice == "n":
        model_n = YOLO(str(YOLOV8N_PATH))
        print(f"YOLOv8n modeli yuklendi: {YOLOV8N_PATH}")
    elif model_choice == "s":
        model_s = YOLO(str(YOLOV8S_PATH))
        print(f"YOLOv8s modeli yuklendi: {YOLOV8S_PATH}")
    else:
        model_n = YOLO(str(YOLOV8N_PATH))
        model_s = YOLO(str(YOLOV8S_PATH))
        print(f"YOLOv8n modeli yuklendi: {YOLOV8N_PATH}")
        print(f"YOLOv8s modeli yuklendi: {YOLOV8S_PATH}")

    try:
        print("\nSecilen ayarlar:")
        print(f"Kaynak       : {camera_source['label']}")
        print(f"Backend      : {backend_choice}")
        print(f"Model modu   : {model_choice}")
        if camera_source["key"] == "usb":
            print(f"Kamera mode  : {selected_mode['label']}")
        else:
            print(f"Android mode : {selected_mode['label']}")
        print(f"Device       : {device_name}")
        print(f"Engine       : {current_engine}")
        print(f"Language     : {language_mode}")

        if camera_source["key"] == "usb" and backend_choice == "gstreamer":
            gst_pipeline = build_gst_pipeline(device_name, selected_mode)
            print(f"Pipeline     : {gst_pipeline}")
        elif camera_source["key"] == "usb":
            print("Pipeline     : Direkt device + CAP_V4L2")
        else:
            android_gst_pipeline = build_android_gst_pipeline(device_name)
            print(f"Pipeline     : {android_gst_pipeline}")

        if camera_source["key"] == "android":
            open_camera_with_retry(camera, attempts=40, delay_seconds=0.5, validate_frame=True)
        else:
            camera.open()

        fps_history = deque(maxlen=30)

        while True:
            loop_start = time.perf_counter()
            frame = camera.read()

            snapshot_frame = None

            if model_choice == "n":
                annotated_n = predict_and_render(
                    model_n, frame, current_engine, boxes_visible, language_mode, registry, tr_store
                )

                loop_time = time.perf_counter() - loop_start
                current_fps = 1.0 / max(loop_time, 1e-6)
                fps_history.append(current_fps)
                avg_fps = sum(fps_history) / len(fps_history)

                if info_visible:
                    draw_info_panel(
                        annotated_n,
                        source_label,
                        backend_choice,
                        "YOLOv8n",
                        selected_mode,
                        current_engine,
                        language_mode,
                        avg_fps,
                    )

                draw_menu_panel(annotated_n, menu_visible)

                cv2.imshow(f"{base_window_name} - YOLOv8n", annotated_n)
                snapshot_frame = annotated_n

            elif model_choice == "s":
                annotated_s = predict_and_render(
                    model_s, frame, current_engine, boxes_visible, language_mode, registry, tr_store
                )

                loop_time = time.perf_counter() - loop_start
                current_fps = 1.0 / max(loop_time, 1e-6)
                fps_history.append(current_fps)
                avg_fps = sum(fps_history) / len(fps_history)

                if info_visible:
                    draw_info_panel(
                        annotated_s,
                        source_label,
                        backend_choice,
                        "YOLOv8s",
                        selected_mode,
                        current_engine,
                        language_mode,
                        avg_fps,
                    )

                draw_menu_panel(annotated_s, menu_visible)

                cv2.imshow(f"{base_window_name} - YOLOv8s", annotated_s)
                snapshot_frame = annotated_s

            else:
                annotated_n = predict_and_render(
                    model_n, frame, current_engine, boxes_visible, language_mode, registry, tr_store
                )
                annotated_s = predict_and_render(
                    model_s, frame, current_engine, boxes_visible, language_mode, registry, tr_store
                )

                loop_time = time.perf_counter() - loop_start
                current_fps = 1.0 / max(loop_time, 1e-6)
                fps_history.append(current_fps)
                avg_fps = sum(fps_history) / len(fps_history)

                if info_visible:
                    draw_info_panel(
                        annotated_n,
                        source_label,
                        backend_choice,
                        "YOLOv8n",
                        selected_mode,
                        current_engine,
                        language_mode,
                        avg_fps,
                    )
                    draw_info_panel(
                        annotated_s,
                        source_label,
                        backend_choice,
                        "YOLOv8s",
                        selected_mode,
                        current_engine,
                        language_mode,
                        avg_fps,
                    )

                draw_menu_panel(annotated_n, menu_visible)
                draw_menu_panel(annotated_s, menu_visible)

                cv2.imshow(f"{base_window_name} - YOLOv8n", annotated_n)
                cv2.imshow(f"{base_window_name} - YOLOv8s", annotated_s)

                snapshot_frame = cv2.hconcat([annotated_n, annotated_s])

            if snapshot_frame is not None:
                record_frame = snapshot_frame.copy()

                if recording:
                    draw_recording_badge(record_frame)

                    if video_writer is None:
                        video_writer, recording_path = create_video_writer(
                            record_frame,
                            selected_mode,
                        )

                    video_writer.write(record_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("i"):
                info_visible = not info_visible
            elif key == ord("m"):
                menu_visible = not menu_visible
            elif key == ord("e"):
                if not torch.cuda.is_available():
                    current_engine = "cpu"
                    print("[ENGINE] CUDA yok. CPU kullaniliyor.")
                else:
                    current_engine = "cpu" if current_engine == "cuda" else "cuda"
                    print(f"[ENGINE] Yeni engine: {current_engine.upper()}")
            elif key == ord("l"):
                language_mode = "tr" if language_mode == "en" else "en"
                print(f"[LANGUAGE] Yeni dil: {language_mode.upper()}")
            elif key == ord("p"):
                if snapshot_frame is not None:
                    save_snapshot(snapshot_frame)
            elif key == ord("r"):
                if not recording:
                    recording = True
                    video_writer = None
                    recording_path = None
                    print("[RECORD] Kayit istegi alindi. Sonraki frame ile baslatilacak.")
                else:
                    recording = False
                    stop_video_writer(video_writer, recording_path)
                    video_writer = None
                    recording_path = None
            elif key == ord("x"):
                boxes_visible = not boxes_visible
                print(f"[BOXES] {'ACIK' if boxes_visible else 'KAPALI'}")

    except CameraError as e:
        print(f"[KAMERA HATASI] {e}")
    except Exception as e:
        print(f"[GENEL HATA] {e}")
    finally:
        stop_video_writer(video_writer, recording_path)
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        print("Kamera kapatildi.")


if __name__ == "__main__":
    main()
