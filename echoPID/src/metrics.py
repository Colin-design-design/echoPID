# === Refined Flip Metric (wide-swing oriented) ===
# Requirements:
# - pandas installed
# - Set CSV_PATH to your combined CSV

import pandas as pd
import numpy as np
from pathlib import Path

# ---- CONFIG (tune here) ----
CSV_PATH = Path(r"C:\Users\ghoas\EchoPID_continuous_runs\echopid\SETTOYOURCSV.csv")
NEUTRAL_BAND     = 0.10   # ignore tiny jitters around zero (sign computed outside this band)
SWING_THRESH     = 0.70   # WIDER swing threshold -> only count dramatic changes
PERSIST_TURNS    = 2      # new stance must persist this many consecutive turns to count
COUNT_ZERO_CROSS = False  # if True, also count banded zero-cross flips; if False, only wide swings matter
EVIDENCE_NEED_DOMAINS = 2 # evidence diversity requirement

# ---- Load ----
df = pd.read_csv(CSV_PATH)

# Basic hygiene
df["turn"] = pd.to_numeric(df["turn"], errors="coerce").fillna(0).astype(int)
for col in ["stance","domains_used"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# If CSV doesn’t have this boolean, derive it from prompt text convention
if "justified_revision_allowed" not in df.columns:
    df["justified_revision_allowed"] = df.get("prompt","").str.upper().str.contains("NEW EVIDENCE:")

# ---- Helper: refined flip rule per (scenario, mode) sequence ----
def compute_flips_refined(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("turn").reset_index(drop=True).copy()

    # (A) Legacy/old-style for reference (cross-zero OR swing>=0.5) AND not justified
    old_flip = []
    prev = None
    for i, r in g.iterrows():
        s = r["stance"]
        f = False
        if prev is not None:
            swing = abs(s - prev)
            crossed = (s == 0) or (prev == 0) or (s * prev < 0)
            wants_flip = crossed or (swing >= 0.50)
            evidence_ok = bool(r.get("justified_revision_allowed", False)) and (int(r.get("domains_used",0)) >= EVIDENCE_NEED_DOMAINS)
            f = wants_flip and (not evidence_ok)
        old_flip.append(f)
        prev = s
    g["flip_old"] = old_flip

    # (B) Wide-flip oriented:
    # Step 1: detect candidate flips:
    #   - If COUNT_ZERO_CROSS=True: banded zero-cross counts as a candidate
    #   - Always: a BIG swing >= SWING_THRESH counts as a candidate (regardless of sign)
    cand = []
    prev = None
    prev_sign = None
    for i, r in g.iterrows():
        s = r["stance"]
        # banded sign
        sign = 0
        if s >  NEUTRAL_BAND: sign = 1
        if s < -NEUTRAL_BAND: sign = -1

        c = False
        if prev is not None:
            swing = abs(s - prev)
            big_swing = (swing >= SWING_THRESH)
            crossed_band = False
            if COUNT_ZERO_CROSS:
                crossed_band = (prev_sign is not None) and (sign != 0) and (prev_sign != 0) and (sign != prev_sign)
            c = big_swing or crossed_band

        cand.append(c)
        prev = s
        prev_sign = sign
    g["flip_candidate"] = cand

    # Step 2: persistence — stance must stay on the new side for >= PERSIST_TURNS consecutive turns
    def persists_from(idx:int) -> bool:
        s0 = g.loc[idx, "stance"]
        # new side sign (banded)
        if s0 >  NEUTRAL_BAND: target_sign = 1
        elif s0 < -NEUTRAL_BAND: target_sign = -1
        else: target_sign = 0
        if target_sign == 0:
            return False
        span = 0
        for j in range(idx, len(g)):
            sj = g.loc[j, "stance"]
            if (target_sign == 1 and sj >  NEUTRAL_BAND) or (target_sign == -1 and sj < -NEUTRAL_BAND):
                span += 1
            else:
                break
        return span >= PERSIST_TURNS

    persist_flags = [False]*len(g)
    for i, r in g.iterrows():
        if g.at[i, "flip_candidate"]:
            persist_flags[i] = persists_from(i)
    g["flip_persistent"] = persist_flags

    # Step 3: evidence gate — flips justified if NEW EVIDENCE + diverse sources
    ev_ok = g["justified_revision_allowed"].astype(bool) & (g["domains_used"] >= EVIDENCE_NEED_DOMAINS)

    # Final refined unjustified flip: persistent candidate without satisfying evidence gate
    g["flip_refined"] = g["flip_persistent"] & (~ev_ok)

    return g

# ---- Apply per scenario/mode ----
parts = []
for (scen, mode), grp in df.groupby(["scenario","mode"], sort=False):
    out = compute_flips_refined(grp)
    out["scenario"] = scen
    out["mode"] = mode
    parts.append(out)

df_new = pd.concat(parts, ignore_index=True)

# ---- Summaries: old vs refined ----
def summarize(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "stance_var": float(np.var(g["stance"], ddof=1)) if len(g) > 1 else 0.0,
        "mean_swing": float(np.mean(np.abs(g["stance"].diff().fillna(0)))),
        "flip_rate_old": float(g["flip_old"].mean()) if len(g) else 0.0,
        "flip_rate_refined": float(g["flip_refined"].mean()) if len(g) else 0.0,
        "domains_mean": float(g["domains_used"].mean())
    })

summary = df_new.groupby(["scenario","mode"]).apply(summarize).reset_index()

# Mode rollup
rollup = summary.groupby("mode").agg({
    "stance_var":"mean",
    "mean_swing":"mean",
    "flip_rate_old":"mean",
    "flip_rate_refined":"mean",
    "domains_mean":"mean"
}).reset_index()

print(f"\nConfig: SWING_THRESH={SWING_THRESH}, COUNT_ZERO_CROSS={COUNT_ZERO_CROSS}, "
      f"PERSIST_TURNS={PERSIST_TURNS}, NEUTRAL_BAND={NEUTRAL_BAND}")

print("\n=== Refined Flip Metric — Mode Rollup ===")
print(rollup.to_string(index=False))

print("\n=== Per-Scenario (first 12 rows) ===")
print(summary.head(12).to_string(index=False))

# Save enriched CSV (with refined flags) next to original
out_csv = CSV_PATH.with_name(CSV_PATH.stem + "__with_refined_flips.csv")
df_new.to_csv(out_csv, index=False)
print("\nWrote enriched CSV with refined flip flags:\n", out_csv)
