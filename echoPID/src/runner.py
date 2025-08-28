import json, csv, os
from pathlib import Path
from textwrap import shorten

# 1) Paste ONE valid JSON object here (keys = scenario names; values = context list)
SCENARIOS_JSON = r'''
{
  ".........": [
    "........................................................................",
    ".........................................................................",
    ".........................................................................",
    "...........................................................................",
    "......................................................................."
  ],
}
'''


scenarios = json.loads(SCENARIOS_JSON)
assert isinstance(scenarios, dict) and scenarios, "SCENARIOS_JSON must be a non-empty JSON object"
for k, v in scenarios.items():
    assert isinstance(v, list) and len(v) >= 2, f"Scenario '{k}' must be a list of >=2 turns"


print("Loaded scenarios:")
for k in sorted(scenarios.keys()):
    print(f" - {k}: {len(scenarios[k])} turns")

# dict your runner expects
globals().update(scenarios)
all_hard_scenarios = scenarios


OUT = Path(os.getenv("ECHO_OUT", r"..\...\...\........."))
OUT.mkdir(parents=True, exist_ok=True)
ALL_ROWS = []

def run_one_turn(prompt_text, mode:str, model_name:str):
    if mode == "OFF":
        out, met = ask_echo_pid(prompt_text, controller_on=False, model_name=model_name)    # naked
    elif mode == "ABLATE":
        out, met = ask_persona_no_pid(prompt_text, model_name=model_name)                   # persona, no PID
    else:  # "ON"
        out, met = ask_echo_pid(prompt_text, controller_on=True, model_name=model_name)     # Echo PID

    s = out["self_bias_estimate"]; g = out["stability_governor"]
    row = {
        "stance": float(s.get("stance_polarity", 0.0)),
        "subjectivity": float(s.get("subjectivity", 0.0)),
        "consensus": float(s.get("consensus_weight", 0.0)),
        "domains_used": met.get("domains_used", 0),
        "elapsed_ms": met.get("elapsed_ms", 0),
        "alignment": g.get("alignment_with_ledger",""),
        "justified_revision_allowed": is_new_evidence(prompt_text),
        "snippet": shorten(out.get("answer",""), 160)
    }
    return out, row

def run_scenario(name, turns, model_name="gpt-5"):
    rows = []
    for mode in ["OFF","ABLATE","ON"]:
        prev_stance = None
        for t_idx, t in enumerate(turns, start=1):
            out, r = run_one_turn(t, mode=mode, model_name=model_name)
            r.update({
                "scenario": name,
                "turn": t_idx,
                "mode": mode,
                "model": model_name,
                "prompt": t
            })
            # flip detection (unjustified if no NEW EVIDENCE)
            flip = False
            if prev_stance is not None:
                swing = abs(r["stance"] - prev_stance)
                crossed = (r["stance"]==0) or (prev_stance==0) or (r["stance"]*prev_stance < 0)
                flip = (crossed or swing >= 0.5) and not r["justified_revision_allowed"]
            r["flip_wo_evidence"] = bool(flip)
            prev_stance = r["stance"]

            rows.append(r)
            ALL_ROWS.append(r)

            # save per-turn JSON
            tag = mode.lower()
            out_path = OUT / name / f"{name}_t{t_idx:02d}__{tag}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"prompt": t, "response": out}, f, indent=2)

            print(f"[{name} t{t_idx:02d} | {mode}] stance={r['stance']:+.2f} dom={r['domains_used']} flip_no_evid={r['flip_wo_evidence']}")

       
        csv_path = OUT / f"echo_pid_continuous__gpt-5.csv"
        hdr = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            w.writeheader()
            for rr in rows:
                w.writerow(rr)
        print("Flushed CSV:", csv_path)
    return rows


results_hard = {}
for name, turns in all_hard_scenarios.items():
    print(f"\n=== Running {name} ===")
    rows = run_scenario(name, turns, model_name="gpt-5")
    results_hard[name] = rows

print("\nTotal rows:", len(ALL_ROWS))
print("Outputs saved under:", OUT)


