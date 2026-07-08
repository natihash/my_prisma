"""
Build a human-readable "top groups" view from all_group_scores_lp.json.

For each head we keep only its most important groups:
  - Sort the groups by score (descending), dropping zero-score groups.
  - Find the natural break point: the largest *relative* gap between two
    consecutive scores. Everything above that gap is kept.
  - Hard cap of MAX_GROUPS (6) regardless of where the gap falls.

The result is written next to the source file with a different name; no
existing script or json is overwritten.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "all_group_scores_lp.json")
OUT = os.path.join(HERE, "top_groups_readable_lp.json")

MAX_GROUPS = 6


def select_top_groups(scores):
    """Return an ordered list of (group, score) for the most important groups."""
    # Sort by score desc, drop zero/negative (a head with no signal -> empty list).
    ranked = sorted(
        ((g, s) for g, s in scores.items() if s > 0),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if not ranked:
        return []

    # Never consider more than the hard cap, then look for a natural break point
    # inside that window via the largest relative drop between neighbours.
    ranked = ranked[:MAX_GROUPS]
    if len(ranked) == 1:
        return ranked

    cut = len(ranked)  # default: keep them all (up to the cap)

    # Pick the position of the single largest relative gap; keep everything above it.
    gaps = [
        ((ranked[i][1] - ranked[i + 1][1]) / ranked[i][1] if ranked[i][1] > 0 else 0.0, i)
        for i in range(len(ranked) - 1)
    ]
    max_gap_ratio, max_gap_idx = max(gaps, key=lambda x: x[0])

    # Only treat it as a real break if the drop is meaningful (>=50% of the
    # higher value). Otherwise the scores are comparable -> keep the full window.
    if max_gap_ratio >= 0.5:
        cut = max_gap_idx + 1

    return ranked[:cut]


def main():
    with open(SRC) as f:
        data = json.load(f)

    out = {}
    for head_key, head_val in data.items():
        top = select_top_groups(head_val["Scores"])
        out[head_key] = {
            "Head_Name": head_val["Head_Name"],
            "Top_Groups": [
                {"Group": g, "Score": round(s, 4)} for g, s in top
            ],
        }

    with open(OUT, "w") as f:
        json.dump(out, f, indent=4)

    print(f"Wrote {len(out)} heads -> {OUT}")
    # Quick sanity preview
    for hk in list(out)[:3]:
        groups = [g["Group"] for g in out[hk]["Top_Groups"]]
        print(f"  {out[hk]['Head_Name']}: {groups}")


if __name__ == "__main__":
    main()
