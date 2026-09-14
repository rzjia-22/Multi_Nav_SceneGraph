# NavDiffusion V0 closed-loop validation

Readiness: **READY_FOR_REAL_SENSOR_ONLY**

- Gate A: 1/1
- Gate B: 1/3

## Gate A episodes

- `validation_scene_000_episode_000`: PASS; goal error 0.348 m; clearance 0.187 m; Safety 1.601%.

## Gate B episodes

- `validation_scene_001_episode_001`: PASS; goal error 0.350 m; clearance 0.214 m; Safety 0.000%.
- `validation_scene_001_episode_004`: FAIL (MODEL: collision); goal error 3.638 m; clearance -0.004 m; Safety 1.025%.
- Not run after fail-fast: `validation_scene_000_episode_002`.

Full ten-mission validation was not run because Gate B failed.

- Dataset split used: validation only
- Test split: untouched
- Hydra/mapping: disabled
