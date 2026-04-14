from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


class TextRenderer:
    def __init__(self, font_path: str | None = None):
        self.font_path = font_path or self._find_default_font()

        self.ft2 = None
        self.use_freetype = False
        self.use_pillow = False

        # 1) OpenCV freetype dene
        if hasattr(cv2, "freetype") and self.font_path:
            try:
                self.ft2 = cv2.freetype.createFreeType2()
                self.ft2.loadFontData(str(self.font_path), 0)
                self.use_freetype = True
                print(f"[TEXT] OpenCV FreeType aktif: {self.font_path}")
                return
            except Exception as e:
                print(f"[TEXT] OpenCV FreeType kullanilamadi: {e}")

        # 2) Pillow dene
        if PIL_AVAILABLE and self.font_path:
            try:
                ImageFont.truetype(str(self.font_path), 20)
                self.use_pillow = True
                print(f"[TEXT] Pillow text renderer aktif: {self.font_path}")
                return
            except Exception as e:
                print(f"[TEXT] Pillow renderer kullanilamadi: {e}")

        # 3) En son fallback
        print("[TEXT] Fallback cv2.putText kullanilacak.")

    def _find_default_font(self) -> str | None:
        candidates = [
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]

        for path in candidates:
            if Path(path).exists():
                return path

        return None

    def _fallback_scale(self, font_height: int) -> float:
        return max(font_height / 30.0, 0.5)

    def get_text_size(self, text: str, font_height: int, thickness: int = 1):
        if self.use_freetype and self.ft2 is not None:
            size, baseline = self.ft2.getTextSize(text, font_height, thickness)
            return size, baseline

        if self.use_pillow and self.font_path:
            font = ImageFont.truetype(str(self.font_path), font_height)
            dummy = Image.new("RGB", (1, 1))
            draw = ImageDraw.Draw(dummy)
            bbox = draw.textbbox((0, 0), text, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            return (width, height), 0

        size, baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            self._fallback_scale(font_height),
            thickness,
        )
        return size, baseline

    def put_text(
        self,
        frame,
        text: str,
        org: tuple[int, int],
        font_height: int,
        color: tuple[int, int, int],
        thickness: int = 1,
        line_type=cv2.LINE_AA,
    ):
        if self.use_freetype and self.ft2 is not None:
            self.ft2.putText(
                frame,
                text,
                org,
                font_height,
                color,
                thickness,
                line_type,
                True,
            )
            return frame

        if self.use_pillow and self.font_path:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(str(self.font_path), font_height)

            # OpenCV BGR, Pillow RGB
            rgb_color = (color[2], color[1], color[0])
            draw.text(org, text, font=font, fill=rgb_color)

            result = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            frame[:] = result
            return frame

        cv2.putText(
            frame,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            self._fallback_scale(font_height),
            color,
            thickness,
            line_type,
        )
        return frame