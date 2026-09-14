# NavDiffusion V0 validation versus final test

| Split | Success | Collisions |
| --- | ---: | ---: |
| Validation | 4/10 | 6 |
| Test | 3/10 | 6 |

Held-out generalization classification: **similar** (test minus validation success count: -1).

## Route buckets

| Bucket | Validation success | Test success | Validation collisions | Test collisions |
| --- | ---: | ---: | ---: | ---: |
| short | 2/3 | 1/3 | 1 | 2 |
| medium | 1/4 | 1/4 | 3 | 2 |
| long | 1/3 | 1/3 | 2 | 2 |

## Prediction risk

- PASS: validation full/control unsafe 5.26%/0.00%; test 2.02%/0.00%.
- FAIL: validation full/control unsafe 50.00%/15.28%; test 33.70%/11.60%.

## Safety

- PASS: Safety intervened in validation 0/4 and test 2/3 episodes; mean override fraction 0.00%/2.84%; mean first intervention n/a/18.055 s.
- FAIL: Safety intervened in validation 4/6 and test 5/7 episodes; mean override fraction 4.77%/6.22%; mean first intervention 6.299999999999999/13.862 s.

## Environment factors

### Terrain

- Validation: flat=3/5 success, 2 collision(s), moderate=1/5 success, 4 collision(s).
- Test: gentle=3/5 success, 1 collision(s), moderate=0/5 success, 5 collision(s).

### Density

- Validation: high=1/5 success, 4 collision(s), medium=3/5 success, 2 collision(s).
- Test: high=0/5 success, 5 collision(s), low=3/5 success, 1 collision(s).

### Ground

- Validation: bare_soil=1/5 success, 4 collision(s), grass=3/5 success, 2 collision(s).
- Test: bare_soil=0/5 success, 5 collision(s), grass=3/5 success, 1 collision(s).

### Lighting

- Validation: bright=1/5 success, 4 collision(s), dim=3/5 success, 2 collision(s).
- Test: bright=3/5 success, 1 collision(s), normal=0/5 success, 5 collision(s).


## Failure signature

- Test collision failures: 6
- Full prediction unsafe before collision: 6/6
- Control prediction unsafe before collision: 6/6
- Validation unsafe-trajectory failure signature reproduced: **True**

## Answers to the final-evaluation questions

- **YES** — Validation reached 4/10 and final test reached 3/10; both recorded 6 and 6 conservative collisions respectively.
- **YES** — All 6 test collisions were preceded by both a full-horizon unsafe prediction and an unsafe first-eight control prediction.
- **YES** — Failures were concentrated in validation_scene_001 during validation and test_scene_001 during test. Their shared factors were ground, terrain, tree_density; this is descriptive and the co-varying scene factors do not establish causality.

## Descriptive patterns

- Test mean planned length was 6.404 m for PASS and 6.547 m for FAIL; each route bucket had exactly one success, so this sample does not show a monotonic length trend.
- Mean expert planning clearance was 0.046 m for PASS and 0.042 m for FAIL, a small difference in this ten-mission sample.
- Mean normalized camera-perturbation extremeness was 0.855 for PASS and 0.748 for FAIL; failures were not concentrated at the more extreme settings.
- Mean simulated duration was 17.110 s for PASS and 13.524 s for FAIL, so failures were not confined to longer rollouts.

## Held-out scene governance

Dataset V0 test was first opened for this NavDiffusion V0 final closed-loop evaluation. Validation influenced early stopping, checkpoint selection and development interpretation; test did not.

Any future NavDiffusion V1 design informed by these results must use new held-out scenes for its final evaluation. The scene and environment comparisons here are descriptive and do not establish causality.
