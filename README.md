# EchoPID: A Feedback Controller for Stabilizing LLM Stances

EchoPID is a proportional–integral–derivative (PID) inspired **external** controller for large language models that:
- Reduces **unjustified flips** (0 with refined, persistence-based metrics)
- Suppresses **stance variance**
- Preserves **domain richness** (evidence breadth)

## Results (Baseline Configuration)

Using tuned thresholds (ε=0.10, swing=0.6, persist=2, Dmin=2) and fixed gains (Kp=0.6, Ki=0.2, Kd=0.1):

- **Unjustified flips**: >50% → 0  
- **Stance variance**: 0.058 → 0.029 (halved)  
- **Evidence domains**: sustained at ~2.8 (vs 1.6 OFF)

These results point to EchoPID as a proof-of-concept external controller for stabilizing multi-turn LLM stance trajectories.

## Repo Layout
- `src/` — controller + metrics + runner
- `scenarios/` — adversarial multi-turn scripts
- `results/` — CSV logs (selected)
- `paper/` — LaTeX (`.tex`) and compiled PDF


## Whitepaper
Full technical details are available in the [EchoPID Whitepaper](paper/echopid_whitepaper.pdf).
