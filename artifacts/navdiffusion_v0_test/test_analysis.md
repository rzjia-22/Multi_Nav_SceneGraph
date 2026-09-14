# NavDiffusion V0 final held-out closed-loop test

All **10/10** canonical test missions were executed exactly once. Navigation succeeded on **3/10** and recorded **6** conservative collision(s).

## Episode overview

| Episode | Scene | Bucket | Expert len [m] | Result | Failure | Goal err [m] | Min clearance [m] | Safety | Full unsafe | Control unsafe |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `test_scene_000_episode_000` | `test_scene_000` | short | 4.724 | PASS | — | 0.349 | 0.198 | 0.00% | 4.76% | 0.00% |
| `test_scene_000_episode_001` | `test_scene_000` | medium | 6.609 | FAIL | MODEL | 4.410 | 0.228 | 0.00% | 27.27% | 0.00% |
| `test_scene_000_episode_002` | `test_scene_000` | medium | 6.115 | PASS | — | 0.345 | 0.157 | 4.97% | 0.00% | 0.00% |
| `test_scene_000_episode_003` | `test_scene_000` | long | 8.119 | FAIL | MODEL | 3.985 | -0.000 | 3.35% | 25.00% | 8.33% |
| `test_scene_000_episode_004` | `test_scene_000` | long | 8.373 | PASS | — | 0.350 | 0.038 | 3.56% | 1.82% | 0.00% |
| `test_scene_001_episode_000` | `test_scene_001` | short | 3.838 | FAIL | MODEL | 0.886 | -0.001 | 11.10% | 34.88% | 18.60% |
| `test_scene_001_episode_001` | `test_scene_001` | short | 3.794 | FAIL | MODEL | 1.663 | -0.003 | 0.00% | 63.64% | 18.18% |
| `test_scene_001_episode_002` | `test_scene_001` | medium | 6.774 | FAIL | MODEL | 0.525 | -0.001 | 14.50% | 16.67% | 8.33% |
| `test_scene_001_episode_003` | `test_scene_001` | medium | 7.219 | FAIL | MODEL | 2.167 | -0.001 | 10.22% | 44.44% | 16.67% |
| `test_scene_001_episode_004` | `test_scene_001` | long | 9.477 | FAIL | MODEL | 3.070 | -0.000 | 4.33% | 50.00% | 7.69% |

## Route buckets

- long: 1/3 success, 2 collision(s), mean goal error 2.468 m, mean clearance 0.013 m, mean Safety 3.75%, full/control unsafe 25.61%/5.34%.
- medium: 1/4 success, 2 collision(s), mean goal error 1.862 m, mean clearance 0.096 m, mean Safety 7.42%, full/control unsafe 22.10%/6.25%.
- short: 1/3 success, 2 collision(s), mean goal error 0.966 m, mean clearance 0.065 m, mean Safety 3.70%, full/control unsafe 34.43%/12.26%.

## Test scenes

- `test_scene_000`: 3/5 success, 1 collision(s), mean goal error 1.888 m; gentle terrain, low density, grass, bright lighting.
- `test_scene_001`: 0/5 success, 5 collision(s), mean goal error 1.662 m; moderate terrain, high density, bare_soil, normal lighting.

## Failure timelines

- `test_scene_000_episode_001`: primary `MODEL`, secondary none; full/control/Safety/collision at 0.8100000000000005/None/None/None s; collision lead None/None/None s.
- `test_scene_000_episode_003`: primary `MODEL`, secondary ['EXPERT_CORRIDOR_DEVIATION']; full/control/Safety/collision at 8.82/17.32/17.590000000000003/18.47 s; collision lead 9.649999999999999/1.1499999999999986/0.8799999999999955 s.
- `test_scene_001_episode_000`: primary `MODEL`, secondary ['EXPERT_CORRIDOR_DEVIATION']; full/control/Safety/collision at 4.29/6.29/16.83/21.8 s; collision lead 17.51/15.510000000000002/4.970000000000002 s.
- `test_scene_001_episode_001`: primary `MODEL`, secondary ['SAFETY_NO_INTERVENTION']; full/control/Safety/collision at 1.6999999999999993/5.699999999999999/None/6.370000000000001 s; collision lead 4.670000000000002/0.6700000000000017/None s.
- `test_scene_001_episode_002`: primary `MODEL`, secondary ['EXPERT_CORRIDOR_DEVIATION']; full/control/Safety/collision at 13.600000000000001/16.6/13.340000000000003/18.61 s; collision lead 5.009999999999998/2.009999999999998/5.269999999999996 s.
- `test_scene_001_episode_003`: primary `MODEL`, secondary none; full/control/Safety/collision at 6.229999999999997/8.699999999999996/8.589999999999996/9.769999999999996 s; collision lead 3.539999999999999/1.0700000000000003/1.1799999999999997 s.
- `test_scene_001_episode_004`: primary `MODEL`, secondary ['EXPERT_CORRIDOR_DEVIATION']; full/control/Safety/collision at 4.18/13.239999999999995/12.959999999999994/13.840000000000003 s; collision lead 9.660000000000004/0.6000000000000085/0.8800000000000097 s.

## Scope and governance

- Evaluation split: test only (2 scenes, 10 episodes)
- Each canonical mission was run once; no per-episode reruns
- Train and validation missions were not executed by this command
- Hydra/mapping: disabled
- Expert path: post-run analysis only; never used for control
- Model, checkpoint, preprocessing, controller, Safety, DIABLO, D435i and environments: frozen
- Dataset V0 test split was first opened for NavDiffusion V0 final closed-loop evaluation
- Any future V1 informed by this result requires new held-out final-test scenes
