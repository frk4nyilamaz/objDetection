import json
import re
from pathlib import Path


class LabelTranslator:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self.translations_dir = self.project_root / "translations"
        self.cache_file = self.translations_dir / "label_cache.json"
        self.override_file = self.translations_dir / "label_override.json"

        self.translations_dir.mkdir(parents=True, exist_ok=True)

        self.cache = self._load_json(self.cache_file)
        self.override = self._load_json(self.override_file)

    def _load_json(self, path: Path) -> dict:
        if not path.exists():
            path.write_text("{}", encoding="utf-8")
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_json(self, path: Path, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def normalize_label(self, label: str) -> str:
        value = label.strip().lower()
        value = re.sub(r"\s+", " ", value)
        return value

    def get(self, label: str, target_lang: str = "tr") -> str:
        normalized = self.normalize_label(label)

        override_value = self._get_from_store(self.override, normalized, target_lang)
        if override_value:
            return override_value

        cache_value = self._get_from_store(self.cache, normalized, target_lang)
        if cache_value:
            return cache_value

        translated = self._translate_fallback(label, target_lang)
        self._store_translation(normalized, label, target_lang, translated)
        return translated

    def _get_from_store(self, store: dict, normalized_label: str, target_lang: str) -> str | None:
        entry = store.get(normalized_label)
        if not isinstance(entry, dict):
            return None

        translations = entry.get("translations", {})
        if not isinstance(translations, dict):
            return None

        value = translations.get(target_lang)
        return value if isinstance(value, str) and value.strip() else None

    def _store_translation(self, normalized_label: str, source_label: str, target_lang: str, translated: str) -> None:
        entry = self.cache.get(normalized_label, {})
        if not isinstance(entry, dict):
            entry = {}

        entry["source"] = source_label

        translations = entry.get("translations", {})
        if not isinstance(translations, dict):
            translations = {}

        translations["en"] = source_label
        translations[target_lang] = translated
        entry["translations"] = translations

        self.cache[normalized_label] = entry
        self._save_json(self.cache_file, self.cache)

    def _translate_fallback(self, label: str, target_lang: str) -> str:
        if target_lang == "en":
            return label

        # Simdilik yer tutucu fallback.
        # Gercek ceviri backend'ini bir sonraki adimda baglayacagiz.
        return label