# NavDiffusion V0 exhaustive closed-loop validation analysis

All **10/10** validation missions were executed without experiment-level fail-fast. Navigation succeeded on **4/10**; test was not evaluated and Hydra was disabled.

Readiness: **NOT_READY**

## Episode overview

| Episode | Bucket | Expert len [m] | Result | Failure | Goal err [m] | Min clearance [m] | Safety | Full unsafe | Control unsafe |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `validation_scene_000_episode_000` | short | 4.167 | PASS | — | 0.348 | 0.142 | 0.00% | 13.64% | 0.00% |
| `validation_scene_000_episode_001` | medium | 5.781 | FAIL | MODEL | 4.259 | -0.004 | 0.00% | 75.00% | 25.00% |
| `validation_scene_000_episode_002` | medium | 7.364 | PASS | — | 0.348 | 0.201 | 0.00% | 0.00% | 0.00% |
| `validation_scene_000_episode_003` | long | 9.733 | PASS | — | 0.349 | 0.199 | 0.00% | 0.00% | 0.00% |
| `validation_scene_000_episode_004` | long | 9.913 | FAIL | MODEL | 6.676 | -0.001 | 5.75% | 58.33% | 8.33% |
| `validation_scene_001_episode_000` | short | 4.509 | FAIL | MODEL | 3.409 | -0.001 | 9.51% | 22.22% | 22.22% |
| `validation_scene_001_episode_001` | short | 3.837 | PASS | — | 0.349 | 0.238 | 0.00% | 14.29% | 0.00% |
| `validation_scene_001_episode_002` | medium | 6.599 | FAIL | MODEL | 3.600 | -0.002 | 0.00% | 50.00% | 14.29% |
| `validation_scene_001_episode_003` | medium | 5.366 | FAIL | MODEL | 2.488 | -0.001 | 12.94% | 72.73% | 27.27% |
| `validation_scene_001_episode_004` | long | 8.791 | FAIL | MODEL | 3.521 | -0.005 | 0.40% | 33.33% | 5.56% |

## Route buckets

- long: 1/3 success, 2 collision(s), mean goal error 3.515 m, mean clearance 0.064 m, mean Safety 2.05%.
- medium: 1/4 success, 3 collision(s), mean goal error 2.674 m, mean clearance 0.048 m, mean Safety 3.24%.
- short: 2/3 success, 1 collision(s), mean goal error 1.369 m, mean clearance 0.126 m, mean Safety 3.17%.

## Scenes

- `validation_scene_000`: 3/5 success, 2 collision(s), mean goal error 2.396 m; flat terrain, medium density, grass, dim lighting.
- `validation_scene_001`: 1/5 success, 4 collision(s), mean goal error 2.673 m; moderate terrain, high density, bare_soil, bright lighting.

## Prediction risk and Safety

- PASS: 6/114 full predictions unsafe (5.26%); 0/114 control predictions unsafe (0.00%); Safety intervened in 0/4 episodes for 0.00 s total.
- FAIL: 36/72 full predictions unsafe (50.00%); 11/72 control predictions unsafe (15.28%); Safety intervened in 4/6 episodes for 1.74 s total.

## Failure timelines

- `validation_scene_000_episode_001` — primary `MODEL`; secondary ['SAFETY_NO_INTERVENTION']; full unsafe 1.8399999999999999, control unsafe 3.869999999999999, Safety None, progress degradation None, expert-corridor deviation None, collision 4.49 s; collision lead from full/control/Safety = 2.6500000000000004/0.620000000000001/None s.
- `validation_scene_000_episode_004` — primary `MODEL`; secondary none; full unsafe 1.0200000000000031, control unsafe 6.520000000000003, Safety 6.399999999999999, progress degradation None, expert-corridor deviation None, collision 6.939999999999998 s; collision lead from full/control/Safety = 5.919999999999995/0.4199999999999946/0.5399999999999991 s.
- `validation_scene_001_episode_000` — primary `MODEL`; secondary none; full unsafe 4.3, control unsafe 4.3, Safety 3.8499999999999996, progress degradation None, expert-corridor deviation None, collision 5.25 s; collision lead from full/control/Safety = 0.9500000000000002/0.9500000000000002/1.4000000000000004 s.
- `validation_scene_001_episode_002` — primary `MODEL`; secondary ['SAFETY_NO_INTERVENTION', 'EXPERT_CORRIDOR_DEVIATION']; full unsafe 4.640000000000001, control unsafe 7.109999999999999, Safety None, progress degradation None, expert-corridor deviation 5.91, collision 7.829999999999998 s; collision lead from full/control/Safety = 3.1899999999999977/0.7199999999999989/None s.
- `validation_scene_001_episode_003` — primary `MODEL`; secondary none; full unsafe 2.530000000000001, control unsafe 5.030000000000001, Safety 5.079999999999998, progress degradation None, expert-corridor deviation None, collision 6.169999999999998 s; collision lead from full/control/Safety = 3.639999999999997/1.139999999999997/1.0899999999999999 s.
- `validation_scene_001_episode_004` — primary `MODEL`; secondary ['SAFETY_LATE']; full unsafe 7.100000000000001, control unsafe 9.600000000000001, Safety 9.869999999999997, progress degradation None, expert-corridor deviation None, collision 9.89 s; collision lead from full/control/Safety = 2.789999999999999/0.28999999999999915/0.020000000000003126 s.

## Descriptive findings

- Failed missions averaged 6.827 m planned length versus 6.275 m for successful missions.
- Failure counts by scene: validation_scene_000=2, validation_scene_001=4.
- Mean expert planning clearance: PASS=0.0328m, FAIL=0.0202m.
- Mean Safety override fraction: PASS=0.0000, FAIL=0.0477.
- Mean full-prediction unsafe fraction: PASS=0.0698, FAIL=0.5194.
- Mean control-prediction unsafe fraction: PASS=0.0000, FAIL=0.1711.
- Mean expert cross-track error: PASS=0.4183m, FAIL=0.1717m.
- Mean simulated duration: PASS=15.0200s, FAIL=6.7600s.
- Mean camera perturbation extremeness: PASS=0.6787, FAIL=0.8163.
- Route-bucket success: short=2/3, medium=1/4, long=1/3.
- These ten validation missions support descriptive associations only; they do not establish causality.

## Episode 004 repeatability

Repeatable failure under the declared same-mode/same-tree/±2 s/0.75 m trajectory criteria: **True**. The collision repeated on the same tree with similar timing and trajectory.
Prior/current collision: `tree_021` at 9.739999999999998 s / `tree_021` at 9.89 s; normalized executed-trajectory mean distance 0.106 m.

## Scope

- Split: validation only (2 scenes, 10 episodes)
- Test split evaluated: **NO**
- Hydra/mapping: disabled
- Model, preprocessing, controller, Safety, DIABLO profile, D435i profile and scenes: frozen
- Expert path: post-run analysis only; never used for control
