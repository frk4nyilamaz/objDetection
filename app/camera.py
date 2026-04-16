import cv2

from app.config import build_android_gst_pipeline, build_gst_pipeline


class CameraError(Exception):
    pass


class Camera:
    def __init__(self, source, backend, mode: dict, apply_mode: bool = True):
        self.source = source
        self.backend = backend
        self.mode = mode
        self.apply_mode = apply_mode
        self.cap = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.source, self.backend)

        if not self.cap.isOpened():
            raise CameraError("Kamera acilamadi.")

        if self.backend == cv2.CAP_V4L2 and self.apply_mode:
            self._apply_v4l2_mode()

    def _apply_v4l2_mode(self) -> None:
        width = int(self.mode["width"])
        height = int(self.mode["height"])
        fps = int(self.mode["fps"])
        pixel_format = str(self.mode["pixel_format"]).upper()

        if pixel_format == "MJPG":
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        elif pixel_format == "YUYV":
            fourcc = cv2.VideoWriter_fourcc(*"YUYV")
        else:
            raise CameraError(f"Desteklenmeyen pixel format: {pixel_format}")

        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        if self.cap is None:
            raise CameraError("Kamera henuz acilmadi.")

        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise CameraError("Kameradan frame okunamadi.")

        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def _normalize_v4l2_source(device):
    if isinstance(device, int):
        return device

    if isinstance(device, str) and device.startswith("/dev/video"):
        try:
            return int(device.removeprefix("/dev/video"))
        except ValueError as exc:
            raise CameraError(f"V4L2 device index parse edilemedi: {device}") from exc

    return device


def build_camera(backend_choice: str, device: str, mode: dict, apply_mode: bool = True):
    choice = backend_choice.strip().lower()

    if choice == "gstreamer":
        pipeline = build_gst_pipeline(device, mode)
        return Camera(pipeline, cv2.CAP_GSTREAMER, mode, apply_mode=False)

    if choice == "ffmpeg":
        source = _normalize_v4l2_source(device)
        return Camera(source, cv2.CAP_V4L2, mode, apply_mode=apply_mode)

    raise CameraError("Gecersiz backend secimi. 'gstreamer' veya 'ffmpeg' kullanin.")


def build_android_camera(device: str, mode: dict):
    pipeline = build_android_gst_pipeline(device)
    return Camera(pipeline, cv2.CAP_GSTREAMER, mode, apply_mode=False)
