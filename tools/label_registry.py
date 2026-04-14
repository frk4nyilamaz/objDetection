import json
import re
from pathlib import Path
from typing import Optional


class LabelRegistry:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self.labels_dir = self.project_root / "labels"
        self.source_file = self.labels_dir / "source.json"

        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.source_data = self._load_source()

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

    def _load_source(self) -> dict:
        default = {
            "next_id": 1,
            "labels": {}
        }
        data = self._load_json(self.source_file, default)

        if "next_id" not in data or not isinstance(data["next_id"], int):
            data["next_id"] = 1

        if "labels" not in data or not isinstance(data["labels"], dict):
            data["labels"] = {}

        return data

    def normalize_label(self, label: str) -> str:
        value = label.strip().lower()
        value = re.sub(r"\s+", " ", value)
        return value

    def save(self) -> None:
        self._save_json(self.source_file, self.source_data)

    def get_all_labels(self) -> dict:
        return self.source_data["labels"]

    def get_entry_by_label(self, label: str) -> Optional[dict]:
        normalized = self.normalize_label(label)
        return self.source_data["labels"].get(normalized)

    def register_label(self, label: str) -> dict:
        normalized = self.normalize_label(label)
        existing = self.source_data["labels"].get(normalized)

        if existing is not None:
            return existing

        new_id = self.source_data["next_id"]

        entry = {
            "id": new_id,
            "label": label,
            "normalized_label": normalized
        }

        self.source_data["labels"][normalized] = entry
        self.source_data["next_id"] = new_id + 1
        self.save()

        return entry

    def get_or_create_label(self, label: str) -> dict:
        entry = self.get_entry_by_label(label)
        if entry is not None:
            return entry
        return self.register_label(label)