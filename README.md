# EchoPID: A Feedback Controller for Stabilizing LLM Stances

EchoPID is a proportional–integral–derivative (PID) inspired **external** controller for large language models that:
- Reduces **unjustified flips** (0 with refined, persistence-based metrics)
- Suppresses **stance variance**
- Preserves **domain richness** (evidence breadth)

## Repo Layout
- `src/` — controller + metrics + runner
- `scenarios/` — adversarial multi-turn scripts
- `results/` — CSV logs (selected)
- `paper/` — LaTeX (`.tex`) and compiled PDF

## Install
```bash
pip install -r requirements.txt
