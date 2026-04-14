import json
from pathlib import Path

from app.config import PROJECT_ROOT
from tools.label_registry import LabelRegistry
from tools.label_locale_store import LabelLocaleStore


def load_json(path: Path) -> dict:
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return {}


def merge_legacy_sources(cache_data: dict, override_data: dict) -> dict:
    merged = {}

    for key, value in cache_data.items():
        merged[key] = value

    for key, value in override_data.items():
        merged[key] = value

    return merged


def main():
    legacy_cache_file =    Path("/mnt//sda1//furkan//objectDetectionProject//labels//legacy_cache.json")
    legacy_override_file = Path("/mnt//sda1//furkan//objectDetectionProject//labels//legacy_override.json")

    legacy_cache = load_json(legacy_cache_file)
    legacy_override = load_json(legacy_override_file)

    print(f"[INFO] cache entries   : {len(legacy_cache)}")
    print(f"[INFO] override entries: {len(legacy_override)}")

    merged = merge_legacy_sources(legacy_cache, legacy_override)
    print(f"[INFO] merged entries  : {len(merged)}")

    registry = LabelRegistry(PROJECT_ROOT)
    tr_store = LabelLocaleStore(PROJECT_ROOT, "tr")

    imported_count = 0
    skipped_count = 0

    for raw_label, entry in merged.items():
        if not isinstance(entry, dict):
            skipped_count += 1
            continue

        source_label = entry.get("source", raw_label)
        translations = entry.get("translations", {})

        if not isinstance(source_label, str) or not source_label.strip():
            skipped_count += 1
            continue

        if not isinstance(translations, dict):
            skipped_count += 1
            continue

        tr_value = translations.get("tr")
        if not isinstance(tr_value, str) or not tr_value.strip():
            skipped_count += 1
            continue

        registry_entry = registry.get_or_create_label(source_label)
        label_id = registry_entry["id"]

        tr_store.upsert_translation(
            label_id=label_id,
            source_label=registry_entry["label"],
            translation=tr_value,
            status="ready",
        )

        imported_count += 1

    print(f"[MIGRATION] Imported: {imported_count}")
    print(f"[MIGRATION] Skipped : {skipped_count}")
    print("[MIGRATION] Done.")


if __name__ == "__main__":
    main()