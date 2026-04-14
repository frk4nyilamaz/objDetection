import cv2

from app.config import build_gst_pipeline


class CameraError(Exception):
    pass


class Camera:
    def __init__(self, source, backend, mode: dict):
        self.source = source
        self.backend = backend
        self.mode = mode
        self.cap = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.source, self.backend)

        if not self.cap.isOpened():
            raise CameraError("Kamera acilamadi.")

        if self.backend == cv2.CAP_V4L2:
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


def build_camera(backend_choice: str, device: str, mode: dict):
    choice = backend_choice.strip().lower()

    if choice == "gstreamer":
        pipeline = build_gst_pipeline(device, mode)
        return Camera(pipeline, cv2.CAP_GSTREAMER, mode)

    if choice == "ffmpeg":
        return Camera(device, cv2.CAP_V4L2, mode)

    raise CameraError("Gecersiz backend secimi. 'gstreamer' veya 'ffmpeg' kullanin.")