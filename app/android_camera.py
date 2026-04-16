import subprocess
import time
from pathlib import Path

from app.config import SCRCPY_BINARY_PATH, build_android_scrcpy_command


class AndroidCameraError(Exception):
    pass


class AndroidCameraSession:
    def __init__(self, sink_device: str, mode: dict):
        self.sink_device = sink_device
        self.mode = mode
        self.process = None
        self.log_path = Path("/tmp") / "android_camera_scrcpy.log"
        self.log_handle = None
        self.scrcpy_dir = Path(SCRCPY_BINARY_PATH).parent
        self.adb_binary_path = self.scrcpy_dir / "adb"

    def _validate_scrcpy_binary(self) -> None:
        binary_path = Path(SCRCPY_BINARY_PATH)
        if not binary_path.exists():
            raise AndroidCameraError(f"scrcpy binary bulunamadi: {binary_path}")

    def _ensure_adb_device_visible(self) -> None:
        adb_binary = self.adb_binary_path if self.adb_binary_path.exists() else Path("adb")

        try:
            result = subprocess.run(
                [str(adb_binary), "devices"],
                capture_output=True,
                text=True,
                check=False,
                cwd=str(self.scrcpy_dir),
            )
        except Exception as exc:
            raise AndroidCameraError(f"adb devices calistirilamadi: {exc}") from exc

        if result.returncode != 0:
            error_text = result.stderr.strip() or result.stdout.strip()
            raise AndroidCameraError(f"adb devices basarisiz: {error_text}")

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        connected_devices = []
        for line in lines[1:]:
            if "\tdevice" in line:
                connected_devices.append(line.split("\t")[0].strip())

        if not connected_devices:
            raise AndroidCameraError(
                "Android cihaz gorunmuyor. Once `adb devices` ile telefonu gorunur hale getirin."
            )

        print(f"[ANDROID] ADB cihaz bulundu: {connected_devices[0]}")

    def _ensure_sink_device_ready(self) -> None:
        if not Path(self.sink_device).exists():
            raise AndroidCameraError(
                f"V4L2 sink cihazi bulunamadi: {self.sink_device}. "
                "Once `sudo modprobe v4l2loopback video_nr=2 card_label=\"AndroidCam\" exclusive_caps=1` "
                "calistirin ve sonra `ls -l /dev/video2` ile kontrol edin."
            )

    def start(self, startup_timeout: float = 2.0) -> None:
        if self.process is not None and self.process.poll() is None:
            return

        self._validate_scrcpy_binary()
        self._ensure_adb_device_visible()
        self._ensure_sink_device_ready()

        cmd = build_android_scrcpy_command(self.mode, self.sink_device)
        print(f"[ANDROID] scrcpy baslatiliyor: {' '.join(cmd)}")

        try:
            self.log_handle = self.log_path.open("w", encoding="utf-8")
            self.process = subprocess.Popen(
                cmd,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                cwd=str(self.scrcpy_dir),
            )
        except Exception as exc:
            if self.log_handle is not None:
                self.log_handle.close()
                self.log_handle = None
            raise AndroidCameraError(f"scrcpy baslatilamadi: {exc}") from exc

        time.sleep(startup_timeout)
        if self.process.poll() is not None:
            log_excerpt = self._read_log_excerpt()
            raise AndroidCameraError(
                "scrcpy erken sonlandi. `adb devices`, `ls -l /dev/video2` ve scrcpy logunu kontrol edin. "
                f"Log: {log_excerpt}"
            )

        print(f"[ANDROID] scrcpy subprocess calisiyor: {self.sink_device}")

    def _read_log_excerpt(self) -> str:
        if self.log_handle is not None:
            self.log_handle.flush()

        if not self.log_path.exists():
            return "scrcpy logu olusmadi."

        lines = self.log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if not lines:
            return f"bos log ({self.log_path})"

        excerpt = " | ".join(lines[-6:])
        return f"{excerpt} [{self.log_path}]"

    def ensure_running(self) -> None:
        if self.process is None:
            return

        if self.process.poll() is not None:
            log_excerpt = self._read_log_excerpt()
            raise AndroidCameraError(
                "scrcpy prosesi calisma sirasinda sonlandi. "
                f"Log: {log_excerpt}"
            )

    def stop(self) -> None:
        if self.process is None:
            if self.log_handle is not None:
                self.log_handle.close()
                self.log_handle = None
            return

        if self.process.poll() is None:
            self.process.terminate()

            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)

            print("[ANDROID] scrcpy durduruldu.")

        self.process = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
