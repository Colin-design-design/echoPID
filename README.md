# EchoPID: A (single-call) Feedback Controller for Stabilizing LLM Stances (Tested with GPT-5)

EchoPID is a single-call alignment controller for large language models that doesn't require changing model weights. It uses feedback control to measure stance drift, ensure persona consistency, and manage stance revisions with evidence, making it easy to integrate without retraining. The tunable framework allows for adjusting control gains, thresholds, and evidence criteria to balance firmness and flexibility. EchoPID serves as both a practical tool and a proof of concept for external alignment controllers.

## Results (Baseline Configuration)

Using tuned thresholds (ε=0.10, swing=0.6, persist=2, Dmin=2) and fixed gains (Kp=0.6, Ki=0.2, Kd=0.1):

- **Unjustified flips**: >50% → 0  
- **Stance variance**: 0.058 → 0.029 (halved)  
- **Evidence domains**: sustained at ~2.8 (vs 1.6 OFF)

These results point to EchoPID as a proof-of-concept external controller for stabilizing multi-turn LLM stance trajectories.

## Repo Layout
- `src/` — controller + runner + metrics
- `results/` — CSV logs 




