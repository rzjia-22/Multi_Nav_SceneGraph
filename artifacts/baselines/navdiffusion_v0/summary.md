# NavDiffusion V0 frozen baseline

NavDiffusion V0 is a reproducible historical baseline, not an active deployment
candidate. Its final readiness classification is **NOT_READY**.

## Identity

- Dataset: Dataset V0, 50 train / 10 validation / 10 test episodes
- Checkpoint: `models/trained/navdiffusion_v0/best.pt`
- SHA256: `7b3bca67af26c2d839555e9b27a7a420762b572fa88664a227986174d9a68ae0`
- Size: 142,598,595 bytes
- Full pre-cleanup evidence: annotated tag `pre-cleanup-navdiffusion-v0-final`

## Training and open-loop validation

- Best epoch: 100; early stopping at epoch 120
- Validation ADE: 0.090123 m
- Validation FDE: 0.169597 m
- Validation diffusion loss: 0.006512
- Test was not used for training, early stopping, or checkpoint selection

## Closed-loop result

The frozen model was evaluated in the shared Isaac Research Forest with the DIABLO
standing surrogate and D435i sensor contract. Hydra was disabled.

| Split | Success | Conservative collisions | Other terminal failure |
|---|---:|---:|---:|
| Validation | 4/10 | 6 | 0 |
| Final held-out test | 3/10 | 6 | 1 terrain exit |

All six test collisions were preceded by both an unsafe full 32-point prediction and
an unsafe first-eight-point control prediction. This repeats the validation failure
signature and is evidence that V0 lacks reliable local obstacle avoidance; it is not a
causal proof about a particular architecture component.

Dataset V0 test was first opened for this final V0 evaluation. Any future V1 design
informed by these results must use new held-out scenes for its final evaluation.

`results.json` is the current-tree machine-readable authority. Detailed Gate A/B,
per-episode, exhaustive-validation, and final-test evidence remains recoverable from
the pre-cleanup tag and Git history.
