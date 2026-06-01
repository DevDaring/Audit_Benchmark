"""Diagnose pentad generation failures on VM."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code" / "mirage"))

import numpy as np
from Dataset.pentad_generator import (
    AUDIT_SOURCES,
    _build_full_prompt,
    _build_slot_b,
    _build_slot_c,
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
    rng = np.random.default_rng(42)
    failures: list[tuple] = []

    for _, row in audit.iterrows():
        d = row.to_dict()
        sid = d.get("seed_id")
        cat = d.get("seed_category")
        try:
            pt, gold = _build_full_prompt(d)
            if not gold or str(gold).strip().lower() in ("unknown", "none", "nan"):
                failures.append((sid, "gold", cat, pt[:100]))
                continue
            st, mode = _resolve_swap_target(pt, d, equiv)
            if not st:
                failures.append((sid, "no_token", cat, pt[:120]))
                continue
            sb = _build_slot_b(d, equiv)
            if sb["prompt_text"].strip() == pt.strip():
                failures.append((sid, "slot_b_identical", cat, pt[:120]))
                continue
            _build_slot_c(d, equiv, rng)
        except Exception as exc:
            failures.append((sid, "slot_c", cat, str(exc)[:150]))

    print(f"TOTAL {len(audit)} FAILURES {len(failures)}")
    print("BY_REASON", dict(Counter(f[1] for f in failures)))
    by_cat = Counter((f[1], f[2]) for f in failures)
    for k, v in sorted(by_cat.items(), key=lambda x: -x[1])[:30]:
        print(v, k)
    print("---SAMPLES---")
    for f in failures[:20]:
        print(f)


if __name__ == "__main__":
    main()
