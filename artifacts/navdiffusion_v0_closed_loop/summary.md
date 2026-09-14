# NavDiffusion V0 closed-loop validation

Readiness: **NOT_READY**

- Gate A: 1/1
- Gate B: 1/3

## Gate A episodes

- `validation_scene_000_episode_000`: PASS; goal error 0.348 m; clearance 0.187 m; Safety 1.601%.

## Gate B episodes

- `validation_scene_001_episode_001`: PASS; goal error 0.350 m; clearance 0.214 m; Safety 0.000%.
- `validation_scene_001_episode_004`: FAIL (MODEL: collision); goal error 3.638 m; clearance -0.004 m; Safety 1.025%.
- Not run after fail-fast: `validation_scene_000_episode_002`.

## Full validation

- Full validation: 4/10
- All experiments executed: True
- Conservative collisions: 6
- Timeout/stall: 0/0
- Mean goal error: 2.535 m
- Mean/p95 inference: 117.5/262.1 ms
- Mean safety override fraction: 0.029
- Simulated / complete session wall time: 100.64 / 437.25 s
- long: 1/3 success, 2 collision(s).
- medium: 1/4 success, 3 collision(s).
- short: 2/3 success, 1 collision(s).
- `validation_scene_000`: 3/5 success, 2 collision(s).
- `validation_scene_001`: 1/5 success, 4 collision(s).

- Dataset split used: validation only
- Test split: untouched
- Hydra/mapping: disabled
