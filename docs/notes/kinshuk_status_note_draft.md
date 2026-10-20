# Draft note for Kinshuk — status + a lab-wide disk flag
*(Draft for Sarthak to adapt/send in his own voice — Aug 25, 2026)*

Hi Kinshuk — quick update from acting on your feedback, plus one thing the lab should know about.

## ⚠️ Lab-wide disk issue (needs attention beyond my folder)
The `/hpc/group/youlab` group space is **99% full (46T/46T, only ~0.7T free)**. This is what killed my first fine-tune last week — the 8.6 GB checkpoint truncated mid-write. It will break anyone's large writes (checkpoints, datasets), not just mine. I've freed ~21 GB on my side (redundant probe checkpoints + training viz) and switched my sweeps to a **score-then-prune** discipline (never hold more than one run's checkpoints), but the group quota itself likely needs a lab-wide cleanup or a request to research computing.

## Progress on your feedback
- **Fine-tuning closes the OOD gap (the key result).** Scored the salvaged 2-species fine-tune on a clean held-out split: FID **359 → 92**, realism-vs-sibling **2.5× → 1.19×** the real-vs-real baseline, diversity healthy (no mode collapse). Best at ~2 epochs; epoch 3 slightly regresses. So the "breadth of data / adapt to the new domain" approach works.
- **Inference sweeps (your scale 9→15.1 point):** running a systematic `scale × strength` sweep on my own checkpoint, scored on both in-domain and OOD. One nuance I want to flag: with our empty prompts + guess_mode off, the control image is in both CFG branches, so `scale` (CFG) mostly changes sharpness — the knob that targets the "ignores the input" OOD failure is **control strength**, so I'm sweeping both.
- **Freezing (your point 5):** added a `FT_FREEZE` option to freeze early ControlNet layers during fine-tuning; running a none-vs-shallow ablation. (Had to freeze via optimizer-exclusion rather than `requires_grad=False` because the model's gradient checkpointing errors otherwise — noted.)
- **Flow matching:** wrote up a go/no-go — short version: it's a better backbone (SD3/FLUX have ControlNet analogs) but not a targeted OOD fix; worth a bounded SD3-medium pilot *after* we exhaust the current levers. Happy to walk through it.
- **Documentation:** every run now writes a manifest (hyperparams + seed + git SHA + job id) to a central registry, and results sync to a W&B dashboard — so all sweeps are captured as data.

Details/plots to follow once the sweeps finish.
