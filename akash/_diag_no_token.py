"""Sample no_token failures for manual inspection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code" / "mirage"))

from Dataset.pentad_generator import (
    AUDIT_SOURCES,
    _build_full_prompt,
    _canonical_category,
    _load_equiv_sets,
    _resolve_swap_target,
)
from Dataset.sample_seeds import sample_seeds


def main() -> None:
    main_seeds, _ = sample_seeds()
    audit = main_seeds[
        main_seeds["seed_source"].astype(str).str.lower().isin(AUDIT_SOURCES)
    ]
    equiv = _load_equiv_sets()
    by_cat: dict[str, list] = {}

    for _, row in audit.iterrows():
        d = row.to_dict()
        pt, gold = _build_full_prompt(d)
        if not gold or str(gold).strip().lower() in ("unknown", "none", "nan"):
            continue
        st, _ = _resolve_swap_target(pt, d, equiv)
        if st:
            continue
        cat = _canonical_category(d)
        by_cat.setdefault(cat, []).append((d["seed_id"], d.get("seed_source"), pt[:200], d.get("ans0"), d.get("ans1"), d.get("ans2")))

    for cat in sorted(by_cat, key=lambda c: -len(by_cat[c])):
        print(f"\n=== {cat} ({len(by_cat[cat])}) ===")
        for item in by_cat[cat][:3]:
            print(item)


if __name__ == "__main__":
    main()
