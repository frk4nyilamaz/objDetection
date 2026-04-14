import time
from collections import deque
from datetime import datetime

import cv2
import torch
from ultralytics import YOLO

from app.config import (
    CAMERA_MODES,
    SNAPSHOT_DIR,
    PROJECT_ROOT,
    YOLOV8N_PATH,
    FONT_PATH,
    YOLOV8S_PATH,
    get_mode_by_id,
    build_gst_pipeline,
)
from app.camera import build_camera, CameraError
from app.text_renderer import TextRenderer
from tools.label_registry import LabelRegistry
from tools.label_locale_store import LabelLocaleStore


renderer = TextRenderer(str(FONT_PATH))


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

    print("Gecersiz secim yapildi. Varsayilan olarak YOLOv8n secildi.")
    return "n"


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


def draw_info_panel(frame, backend_choice, model_name, selected_mode, current_engine, language_mode, avg_fps):
    info_lines = [
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


def main():
    backend_choice = get_user_backend_choice()
    model_choice = get_user_model_choice()
    selected_mode = get_user_mode_choice()

    device_name = "/dev/video0"
    base_window_name = "CAM-TEST"

    current_engine = "cuda" if torch.cuda.is_available() else "cpu"
    language_mode = "en"
    info_visible = True
    menu_visible = True
    boxes_visible = True

    registry = LabelRegistry(PROJECT_ROOT)
    tr_store = LabelLocaleStore(PROJECT_ROOT, "tr")

    camera = build_camera(
        backend_choice=backend_choice,
        device=device_name,
        mode=selected_mode,
    )

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
        print(f"Backend      : {backend_choice}")
        print(f"Model modu   : {model_choice}")
        print(f"Kamera mode  : {selected_mode['label']}")
        print(f"Device       : {device_name}")
        print(f"Engine       : {current_engine}")
        print(f"Language     : {language_mode}")

        if backend_choice == "gstreamer":
            gst_pipeline = build_gst_pipeline(device_name, selected_mode)
            print(f"Pipeline     : {gst_pipeline}")
        else:
            print("Pipeline     : Direkt device + CAP_V4L2")

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
                        backend_choice,
                        "YOLOv8n",
                        selected_mode,
                        current_engine,
                        language_mode,
                        avg_fps,
                    )
                    draw_info_panel(
                        annotated_s,
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
            elif key == ord("x"):
                boxes_visible = not boxes_visible
                print(f"[BOXES] {'ACIK' if boxes_visible else 'KAPALI'}")

    except CameraError as e:
        print(f"[KAMERA HATASI] {e}")
    except Exception as e:
        print(f"[GENEL HATA] {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Kamera kapatildi.")


if __name__ == "__main__":
    main()