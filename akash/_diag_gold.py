"""Inspect gold failures."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code" / "mirage"))

from Dataset.pentad_generator import AUDIT_SOURCES, _build_full_prompt, _is_scorable_gold
from Dataset.sample_seeds import sample_seeds


def main() -> None:
    main_seeds, _ = sample_seeds()
    audit = main_seeds[main_seeds["seed_source"].astype(str).str.lower().isin(AUDIT_SOURCES)]
    for _, row in audit.iterrows():
        d = row.to_dict()
        pt, gold = _build_full_prompt(d)
        src = str(d.get("seed_source", "")).lower()
        if not _is_scorable_gold(gold, src):
            print(d["seed_id"], d.get("seed_category"), repr(gold))
            print("  label", d.get("label"), "ans0-2", d.get("ans0"), d.get("ans1"), d.get("ans2"))
            print("  pt", pt[:150])
            print()


if __name__ == "__main__":
    main()
