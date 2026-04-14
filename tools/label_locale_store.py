import json
from pathlib import Path
from typing import Optional


class LabelLocaleStore:
    def __init__(self, project_root: str | Path, language_code: str):
        self.project_root = Path(project_root)
        self.labels_dir = self.project_root / "labels"
        self.language_code = language_code.strip().lower()
        self.locale_file = self.labels_dir / f"{self.language_code}.json"

        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.locale_data = self._load_locale()

    def _load_json(self, path: Path, default: dict) -> dict:
        if not path.exists():
            self._save_json(path, default)
            return default.copy()

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else default.copy()
        except Exception:
            return default.copy()

    def _save_json(self, path: Path, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_locale(self) -> dict:
        default = {
            "language": self.language_code,
            "translations": {}
        }
        data = self._load_json(self.locale_file, default)

        if "language" not in data or not isinstance(data["language"], str):
            data["language"] = self.language_code

        if "translations" not in data or not isinstance(data["translations"], dict):
            data["translations"] = {}

        return data

    def save(self) -> None:
        self._save_json(self.locale_file, self.locale_data)

    def _key(self, label_id: int) -> str:
        return str(label_id)

    def get_entry(self, label_id: int) -> Optional[dict]:
        return self.locale_data["translations"].get(self._key(label_id))

    def has_translation(self, label_id: int) -> bool:
        entry = self.get_entry(label_id)
        if not isinstance(entry, dict):
            return False

        translation = entry.get("translation")
        status = entry.get("status")

        return isinstance(translation, str) and bool(translation.strip()) and status == "ready"

    def get_translation(self, label_id: int) -> Optional[str]:
        entry = self.get_entry(label_id)
        if not isinstance(entry, dict):
            return None

        translation = entry.get("translation")
        status = entry.get("status")

        if isinstance(translation, str) and translation.strip() and status == "ready":
            return translation

        return None

    def upsert_translation(
        self,
        label_id: int,
        source_label: str,
        translation: str,
        status: str = "ready"
    ) -> dict:
        key = self._key(label_id)

        entry = {
            "id": label_id,
            "label": source_label,
            "translation": translation,
            "status": status
        }

        self.locale_data["translations"][key] = entry
        self.save()
        return entry

    def mark_pending(self, label_id: int, source_label: str) -> dict:
        key = self._key(label_id)

        existing = self.locale_data["translations"].get(key)
        if isinstance(existing, dict) and existing.get("status") == "ready":
            return existing

        entry = {
            "id": label_id,
            "label": source_label,
            "translation": "",
            "status": "pending"
        }

        self.locale_data["translations"][key] = entry
        self.save()
        return entry