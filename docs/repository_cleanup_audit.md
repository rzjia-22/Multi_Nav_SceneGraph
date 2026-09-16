# Repository cleanup audit

Date: 2026-09-16  
Scope: audit and proposal only; no cleanup has been executed.

## Safety baseline

- Branch: `main`
- Local HEAD: `f29d6a00e0e99fadb9f6c66ce3ca838e849b13c1`
- Remote `origin/main`: `f29d6a00e0e99fadb9f6c66ce3ca838e849b13c1`
- Baseline commit: `Merge pull request #3 from rzjia-22/navdiffusion-v0-test-evaluation`
- Working tree was clean before this report was created.
- Dataset V0 final held-out test is present: validation was 4/10 successful and the
  first-opened test split was 3/10 successful; both had six conservative collisions.
- No reset, rebase, force push, history rewrite, tag, branch deletion, file deletion,
  move, rename, commit, push, LFS prune, or cleanup command was performed.
- Before any accepted cleanup is executed, create and push the annotated tag
  `pre-cleanup-navdiffusion-v0-final` at the HEAD above.

The audit used tracked-file inventory, ignored-file inventory, LFS inventory, disk
usage, Make/Compose/ROS entry points, Python imports, config/document references,
tests, launch files, and content hashes. File names alone were not treated as evidence
that code is obsolete.

## Classification vocabulary

Every major item is assigned one of four dispositions:

- `KEEP_CORE`: active, reusable project infrastructure or authoritative data.
- `KEEP_REFERENCE`: reproducibility/reference material that should remain available,
  but is not the primary future workflow.
- `REFACTOR_CANDIDATE`: useful behavior mixed with one-off V0 behavior; extract the
  reusable part before removing anything.
- `DELETE_CANDIDATE`: redundant, reproducible, ignored, or experiment-only material
  whose remaining references are known and can be removed in a controlled cleanup.

`DELETE_CANDIDATE` is a proposal, not authorization to delete.

## Quantitative inventory

| Item | Current value |
|---|---:|
| Tracked files | 536 |
| Tracked working-tree bytes | 4,884,387,810 B (4.55 GiB) |
| Current LFS files | 71 |
| Current LFS working-tree bytes | 4,842,357,832 B (4.51 GiB) |
| Current Dataset V0 HDF5 bytes | 4,699,759,237 B |
| NavDiffusion V0 `best.pt` | 142,598,595 B |
| `.git/lfs` local object store | 10,875,842,652 B, 187 objects |
| Non-LFS Git object store | about 37 MiB |
| Full checkout including ignored outputs | about 18 GiB |
| Ignored `runs/` | 2,337,824,033 B |
| Ignored pinned ForestNavigation checkout | 875,985,831 B |
| Free filesystem capacity during audit | about 404 GiB |

Tracked bytes by major area:

| Area | Files | Bytes | Disposition |
|---|---:|---:|---|
| `datasets/` | 274 | 4,706,756,113 | `KEEP_CORE` |
| `models/` | 11 | 142,625,536 | `KEEP_REFERENCE` plus active baseline code/config |
| `research_scenes/` | 28 | 19,978,815 | `KEEP_CORE` |
| `artifacts/` | 53 | 14,250,223 | mixed; detailed below |
| `ros_ws/` | 86 | 355,342 | `KEEP_CORE` plus a small V0 refactor boundary |
| `research_data/` | 16 | 181,525 | mixed; detailed below |
| `docs/` | 7 | 72,277 | `KEEP_CORE` / `KEEP_REFERENCE` |
| `tools/` | 14 | 56,488 | mostly `KEEP_CORE` / `KEEP_REFERENCE` |
| `config/` | 22 | 38,816 | mostly `KEEP_CORE`; V0 experiment configs are reference material |
| `tests/` | 9 | 33,291 | `KEEP_CORE` with V0 evidence assertions to refactor |
| `containers/` | 7 | 9,012 | `KEEP_CORE` |

The current LFS tree consists of 70 episode HDF5 files and one trained checkpoint.
Deleting them from the current tree would not delete their historical GitHub LFS
objects or immediately reduce remote LFS usage. LFS history cleanup is explicitly out
of scope and would be a separate high-risk operation.

## Major directory and top-level classification

| Path | Classification | Finding |
|---|---|---|
| `README.md`, `docs/` | `KEEP_CORE` | The seven documents have distinct authority: architecture, interfaces, environment, Dataset V0, NavDiffusion V0, migration, and development status. No temporary progress-document pile was found. |
| `config/` | `KEEP_CORE` | The robot, sensor, forest, Hydra, Nav2, mission, system, and model contracts are active. `navdiffusion_v0_closed_loop.yaml` and `navdiffusion_v0_test.yaml` are V0 reference contracts rather than future generic evaluator design. |
| `containers/`, `compose.yaml` | `KEEP_CORE` | Clean runtime boundaries exist for robotics, ML, simulation, and Research Navigation. No duplicate container stack was found. |
| `research_data/` | `KEEP_CORE` + `REFACTOR_CANDIDATE` | Dataset schema/depth/expert/forest/validation are core. Closed-loop orchestration mixes generic evaluation with V0-only gates and final-test governance. |
| `research_scenes/dataset_v0/` | `KEEP_CORE` | Deterministic scene definitions and generated USDA snapshots are Dataset provenance. |
| `datasets/dataset_v0/` | `KEEP_CORE` | The canonical 70-episode raw source of truth, index, plans, validation, and collection reports. No deletion is proposed. |
| `models/trained/navdiffusion_v0/` | `KEEP_REFERENCE` | The only completed train/validation/test baseline and checkpoint. The checkpoint remains required for reproduction and regression. |
| `models/ForestNavigation/` | `DELETE_CANDIDATE` (local only) | Ignored, pinned upstream checkout created by `make models`; reproducible from the pin and not part of Git. Keep scripts/pin, optionally remove the local checkout when disk is needed. |
| `ros_ws/src/` | `KEEP_CORE` | Eight coherent ROS packages retain Phase 1/2, navigation, motion, simulation, mapping, interfaces, and bringup capabilities. |
| `tools/` | `KEEP_CORE` / `KEEP_REFERENCE` | Acceptance, GPU, Hydra inspection, model-source verification, and sensor helpers remain useful. The unreferenced vegetation asset probe is a small diagnostics reference, not harmful dead weight. |
| `tests/` | `KEEP_CORE` + `REFACTOR_CANDIDATE` | Contract tests are valuable. Tests that assert exact historical V0 experiment results should eventually validate one compact baseline record instead of requiring full artifact trees. |
| `artifacts/dataset_v0_scene_review/` | `KEEP_REFERENCE` | Four RTX views, scene layout, review report, and strict-corner audit document the accepted visual domain; only 6.38 MB. |
| `artifacts/navdiffusion_v0_closed_loop/` | `REFACTOR_CANDIDATE` | 6.39 MB of Gate A/B, exhaustive validation, plots, per-failure images, and duplicated aggregate data. Preserve a concise baseline record; most individual images are history candidates. |
| `artifacts/navdiffusion_v0_test/` | `REFACTOR_CANDIDATE` | 1.48 MB of final one-pass test evidence. Preserve the final metrics/governance summary; most plots and per-failure images need not remain in the current tree. |
| `runs/` | `DELETE_CANDIDATE` (local only) | Correctly ignored high-volume runtime/cache/training data. It is not authoritative source. Delete selectively after confirming no active resume is needed. |
| `Makefile` | `KEEP_CORE` + `REFACTOR_CANDIDATE` | Core commands are clear. Dataset pilot/batch-gate and V0 Gate A/B/final-test targets should be simplified after generic evaluation extraction. |
| `.gitignore`, `.gitattributes` | `KEEP_CORE` | They correctly cover ROS products, caches, runs, model downloads, temporary episode files, and LFS types. No urgent missing pattern was found. |
| `LICENSE`, `THIRD_PARTY_NOTICES.md`, `VERSION`, `upstream.repos` | `KEEP_CORE` | Required licensing, provenance, and environment pinning. |
| `.agents/`, `.codex/` | `KEEP_REFERENCE` (local/app-owned) | Empty, untracked, negligible app-owned directories. Do not include them in repository cleanup commands. |

## Capability audit

### Research Forest, Dataset, sensor, and robot foundation

The following are `KEEP_CORE`:

- `research_data/common.py`, `dataset.py`, `depth.py`, `episode.py`, `expert.py`,
  `forest.py`, and `validation.py`;
- `ros_ws/src/mns_simulation/.../research_forest_scene.py`,
  `research_dataset_runtime.py`, `research_robot.py`, and
  `research_terrain_calibration.py`;
- `config/research_forests/`, `config/sensors/d435i_navigation_v0.yaml`, and
  `config/robots/diablo_standing.yaml`;
- the canonical Dataset V0 manifest, all 14 scene definitions, all 70 episode
  directories, dataset index, and collection reports;
- depth registration and Z16 calibration code, actual terrain surface queries,
  scene batching, atomic episode writes, and validation.

The Dataset V0 manifest remains the authoritative 14-scene/70-episode design. The
`pilot_plan` status describes V0's research role and should not be casually edited,
because the dataset is frozen and its provenance is already published.

### ROS2, simulation, navigation, motion, and Hydra

All eight ROS packages are `KEEP_CORE`. Phase 1/2, Go2, UAV, Coverage, Nav2, legacy
Diffusion, motion arbitration, Hydra integration, synthetic regression, and multi-robot
capability are still valid engineering capabilities even though they are not the
current Dataset V0 workflow. The audit found no second competing package tree.

`config/robots/uav_mapping.yaml` initially appears unreferenced by full path, but is
actively selected by the `uav-mapping` Make target through `MNS_ROBOT_CONFIG`; it is
not dead configuration. Similarly, `tools/sensor_validation.py` is imported by the
Isaac acceptance tool and its tests.

### NavDiffusion model and training split

| Path/function | Classification | Reason |
|---|---|---|
| `navdiffusion_v0/model.py` | `KEEP_REFERENCE` | Baseline model definition; required by checkpoint and predictor. |
| `navdiffusion_v0/preprocessing.py` | `KEEP_CORE` | Shared training/runtime contract and likely reusable by future compatible models. |
| `navdiffusion_v0/predictor.py` | `KEEP_REFERENCE` | Current checkpoint runtime adapter. |
| `navdiffusion_v0/data.py` | `KEEP_REFERENCE` | Reproducible window construction and cache invalidation; useful training foundation. |
| `navdiffusion_v0/training.py` | `KEEP_REFERENCE` | Plain-PyTorch, resumable baseline training; not a temporary script, though V0 defaults should remain clearly scoped. |
| `models/trained/navdiffusion_v0/best.pt` | `KEEP_REFERENCE` | Sole full closed-loop baseline, LFS-managed, SHA256 `7b3bca67af26c2d839555e9b27a7a420762b572fa88664a227986174d9a68ae0`. |
| config/preprocessing/history/metrics beside checkpoint | `KEEP_REFERENCE` | Needed to interpret and reproduce `best.pt`. |
| `sanity_single_batch.json`, `sanity_overfit.json`, `inference_smoke.json`, `ros_runtime_smoke.json` | `REFACTOR_CANDIDATE` | One-run evidence. The commands remain useful, but outputs should normally live under ignored runs and be summarized in one baseline record. Current training gates read two sanity files, so code must be changed before removal. |

The old pinned ForestNavigation source is used only as an upstream compatibility and
asset source. `tools/fetch_model_assets.sh`, `tools/verify_model_source.sh`, the pin,
and attribution are `KEEP_REFERENCE`. The downloaded 876 MB checkout and two ignored
symlinks are local cleanup candidates because they can be rebuilt with `make models`.

### Generic closed-loop infrastructure versus V0-only experiment logic

Generic and future-reusable (`KEEP_CORE` or extract as generic modules):

- shared Research Forest construction and actual terrain surface queries;
- DIABLO surrogate pose/surface following and D435i RGB/raw-depth registration;
- ROS sensor/command boundary and `research_navigation.launch.py`;
- per-episode reset, scene batching, command/safety logging, conservative collision
  checks, prediction traces, timeout/stall handling, and atomic result reports;
- post-run path, clearance, prediction-risk, progress, and cross-track metrics;
- lightweight trajectory and failure visualization.

V0 experiment-specific (`REFACTOR_CANDIDATE`, then archive/delete from active flow):

- fixed Gate A/Gate B episode selection and fail-fast readiness policy;
- V0-specific readiness labels and hard-coded threshold interpretation;
- comparison to the particular Gate B collision on `tree_021`;
- the one-time final-test seal, first-open governance, and validation-vs-test prose;
- artifact paths hard-coded to `navdiffusion_v0_closed_loop` and
  `navdiffusion_v0_test`;
- tests that assert exact 4/10 and 3/10 outcomes from large artifact trees.

The 793-line `research_navigation_runtime.py` is not disposable V0 experiment code;
it is the most valuable generic benchmark runtime and should be retained and renamed
or parameterized only when another navigator is added. In contrast,
`closed_loop_test.py` is mostly one-time V0 final-test governance. It can only be
removed after its generic split-safe evaluation/reporting pieces are extracted.

## Recommended tracked-file cleanup candidates

### Low risk

| Path | Reason | Current references | Future reuse | Restore | Risk |
|---|---|---|---|---|---|
| `artifacts/navdiffusion_v0_closed_loop/full_validation_report.json` | Byte-identical duplicate of `full_validation/aggregate_report.json` (same SHA256). | Written by one line in `closed_loop.py`; no reader requires this path. | None. | `git show pre-cleanup-navdiffusion-v0-final:<path>` | Low. Remove duplicate writer at the same time. |
| `artifacts/dataset_v0_preview/` | Empty, untracked residue from the rejected primitive preview. | Only obsolete removal code in `collect.regenerate_pilot`. | None. | Git history for old rejected preview; current empty dir has no content. | Low. |

### Medium risk: consolidate only after replacement exists

| Path | Reason | Current references | Future reuse | Restore | Risk |
|---|---|---|---|---|---|
| `artifacts/navdiffusion_v0_closed_loop/gate_a/` and `gate_b/` | One-time fail-fast experiment images and reports; the final exhaustive validation supersedes them as model-performance evidence. | `closed_loop_analysis.py`, historical-result tests, top aggregate report, and docs. | Low as raw evidence; the gate concept can remain in generic tooling. | Pre-cleanup tag or PR #2 history. | Medium: extract repeatability summary and remove artifact-coupled tests first. |
| `artifacts/navdiffusion_v0_closed_loop/full_validation/episodes/*/*.png` and `plots/` | Per-failure visualizations are useful for the completed diagnosis but are not needed for runtime or future training. | JSON/Markdown reports refer conceptually to the analysis; code can regenerate from ignored traces only while traces exist. | Low; keep at most one representative failure image if desired. | Pre-cleanup tag or PR #2 history. | Medium: once removed, regeneration may require the ignored raw traces. |
| `artifacts/navdiffusion_v0_test/episodes/` and `plots/` | Per-failure and descriptive test figures from the one-pass V0 final evaluation. | No runtime dependency; docs refer to the artifact root. | Low; the concise final result is scientifically more important. | Pre-cleanup tag or PR #3 history. | Medium for the same raw-trace reason. |
| Repeated closed-loop JSON/Markdown reports | `aggregate_report`, `full_validation_report`, `analysis`, `summary`, `test_report`, `test_analysis`, and comparison files repeat many values. | Tests consume several exact paths; docs consume summaries. | Medium for final metrics, low for duplicated representations. | Tag/history. | Medium: first create one schema-versioned compact baseline result. |
| `research_data/manifest.py` and CLI `init-manifest` | One-time generator for a now-frozen authoritative manifest; it can overwrite the published plan and still emits `pilot_plan`. | Imported only by `research_data/cli.py`; no Make target invokes it. | Low. Future datasets should use a versioned manifest authoring tool, not mutate V0. | Tag/history. | Medium: retain the generated manifest itself unconditionally. |
| `collect.regenerate_pilot`, `collect.batch_gate`, their Make targets, and obsolete preview removal | Completed V0 production gates, including explicit cleanup for the rejected preview tree. | Make targets and `collect.py` CLI only. | Low as named V0 operations; scene-batched collection remains valuable. | Tag/history. | Medium: keep/refactor `collect_scenes`, resume, atomic finalize, and validation. |
| V0-specific parts of `closed_loop.py`, `closed_loop_analysis.py`, and `closed_loop_test.py` | Hard-coded episode sets, artifact paths, readiness rules, repeatability comparison, and final-test governance. | Make targets, tests, configs, docs, and each other. | Low in current form; high after generic extraction. | Tag/history. | Medium/high: never delete the generic runner, metrics, reset, or diagnostics with them. |
| `config/navigation/navdiffusion_v0_test.yaml` | Frozen one-pass final-test contract has historical provenance value but should not remain an active reusable test command after the test split was opened. | `closed_loop_test.py` and tests. | Low operationally, high historically. | Tag/history or concise baseline record. | Medium. |
| Historical numeric assertions in `tests/test_navdiffusion_closed_loop*.py` | Tests currently couple normal repository validation to full past experiment artifact trees and exact outcomes. | Test suite. | Low as unit tests; preserve contract/schema tests and one compact signed result fixture. | Tag/history. | Medium. |
| Formal-model sanity/smoke JSON files | Generated evidence that is tiny but mutable and not model inputs. | Training gate reads the two sanity files; smoke tool writes others. | Low after checkpoint metadata and a compact training summary are authoritative. | Tag/history or rerun the corresponding smoke. | Medium until training gates write/read ignored run outputs. |

No deletion is proposed for `datasets/dataset_v0/`, `research_scenes/dataset_v0/`,
the accepted scene review, the checkpoint, shared preprocessing, or the generic
Research Navigation runtime.

## Ignored and local-only cleanup proposal

The current ignore rules are adequate. `git clean -ndX` reports only expected build,
cache, run, symlink, and bytecode outputs. A future cleanup should use explicit paths,
not a broad `git clean -fdx`.

| Path | Current size | Classification | Proposed handling |
|---|---:|---|---|
| `runs/cache/` | 1,562,203,245 B | `DELETE_CANDIDATE` | Low risk. Fully derived NavDiffusion arrays and downloaded Torch weight cache; rebuildable. |
| ROS `build/`, `install/`, `log/`, Python caches, `.pytest_cache/` | about 19.9 MB | `DELETE_CANDIDATE` | Low risk. Rebuilt by normal validation/build commands. |
| `runs/training/` | 193,720,514 B | `DELETE_CANDIDATE` | Medium risk. Contains `last.pt` and overfit state; remove only after deciding no V0 resume is needed. `best.pt` is separately tracked. |
| Other historical `runs/` | about 581.9 MB | `DELETE_CANDIDATE` | Medium risk. Phase 1/2, Hydra, Isaac, validation, and debug logs. Their authoritative summaries are in Git, but spot-check before removal. |
| `models/ForestNavigation/` | 875,985,831 B | `DELETE_CANDIDATE` | Medium risk local cleanup. Re-fetchable from the pinned revision, but removal loses offline operation until network/LFS fetch succeeds. |
| `models/go2_locomotion.pt`, `models/navdiffusion.ckpt` | ignored symlinks | `DELETE_CANDIDATE` | Remove with the local upstream checkout; recreated by `make models`. |
| `.agents/`, `.codex/` | negligible | `KEEP_REFERENCE` | App-owned and empty; leave alone. |

Estimated local reclaim:

- low-risk ignored cleanup: about 1.58 GB;
- additional medium-risk ignored cleanup: about 1.65 GB;
- total possible ignored/local reclaim: about 3.23 GB;
- `.git/lfs` remains about 10.88 GB and is not touched.

## Remote branch audit

All three published feature branches are ancestors of `main` and have zero commits
unique relative to `main`:

| Remote branch | Tip | Main-only / branch-only commits | Classification |
|---|---|---:|---|
| `origin/navdiffusion-v0-training` | `d2531784111a64a93bc76bd6ff1edec7b5269cef` | 15 / 0 | `REMOTE_BRANCH_DELETE_CANDIDATE` |
| `origin/navdiffusion-v0-closed-loop` | `cbbde5108b926d2c14c1e40519ae4568466a6cb5` | 7 / 0 | `REMOTE_BRANCH_DELETE_CANDIDATE` |
| `origin/navdiffusion-v0-test-evaluation` | `df5980e6a8aa050cdbf90b0456482845b4cc7754` | 1 / 0 | `REMOTE_BRANCH_DELETE_CANDIDATE` |

Delete neither remote nor local branches in this audit. After the annotated pre-cleanup
tag is pushed, deleting these feature refs would simplify branch listings without
losing commits. It will not materially reduce Git or LFS storage.

## Target repository shape

Avoid a broad rename of stable modules. The desired main tree is close to the current
layout:

```text
config/                 versioned runtime, robot, sensor, mapping, model contracts
docs/                   small authoritative document set
models/
  trained/navdiffusion_v0/  frozen checkpoint plus compact reproducibility metadata
research_data/          model-agnostic dataset and generic evaluation libraries
research_scenes/        deterministic scene specifications/snapshots
datasets/dataset_v0/    canonical raw source of truth
ros_ws/src/             modular ROS2 packages
tools/                  reusable diagnostics and acceptance entry points
tests/                  contract/unit tests, not large historical-result assertions
artifacts/
  dataset_v0_scene_review/
  baselines/navdiffusion_v0/
    summary.md
    results.json
    representative_failure.png   # optional, at most a few selected images
```

Recommended refactoring boundary:

```text
research_data/evaluation/
  runner.py              split-safe, scene-batched execution
  metrics.py             collision, progress, prediction risk, cross-track
  visualization.py       generic lightweight evidence

baseline-specific config/report
  NavDiffusion V0 episode selection, checkpoint identity, frozen result summary
```

This preserves the difficult, reusable infrastructure while allowing the Gate A/B and
final-test lifecycle to leave the active code path.

### Should V0 move to `models/baselines/navdiffusion_v0/`?

Not yet. The current `models/trained/navdiffusion_v0/` path is clear and is referenced
by configs, runtime code, tests, documentation, and LFS. Moving it now creates churn
without reducing ambiguity. Prefer these steps first:

1. keep `best.pt`, config, preprocessing, training history, data report, and final
   metrics in the current directory;
2. create one compact, schema-versioned baseline result under
   `artifacts/baselines/navdiffusion_v0/` (or beside the checkpoint);
3. remove duplicated/per-episode evidence only after tests and docs use that record;
4. introduce `models/baselines/` only when a second trained model makes the distinction
   materially useful.

Git history and the annotated tag are preferable to a permanent `old/`, `backup/`, or
`final_final/` tree.

## Estimated post-cleanup impact

Recommended cleanup does not target large authoritative assets, so tracked-size savings
are intentionally modest:

| Metric | Before | Estimated after recommended cleanup |
|---|---:|---:|
| Tracked files | 536 | approximately 490-505 |
| Tracked working-tree bytes | 4.884 GB | approximately 4.876 GB |
| Tracked artifact bytes | 14.25 MB | approximately 6.5-7.0 MB |
| Current LFS bytes | 4.842 GB | unchanged |
| Local `.git/lfs` bytes | 10.876 GB | unchanged |
| Full local checkout | about 18 GiB | about 14.8-15.7 GiB, depending on upstream clone retention |

The main benefit is reduced conceptual and test coupling, not repository byte size.
Deleting Dataset V0 or `best.pt` would produce much larger working-tree savings but is
specifically not recommended and would not reclaim historical remote LFS storage.

## Ambiguous decisions requiring user approval

1. **Baseline evidence depth.** Keep only summary JSON/Markdown, or retain one
   representative validation and one test failure image as visual evidence? The audit
   recommends the latter small compromise.
2. **Training resumability.** Delete ignored `runs/training/navdiffusion_v0/last.pt`
   now that V0 is frozen, or retain it until a future V1 plan is decided? It is not
   needed for inference or reproduction from the published `best.pt`.
3. **Offline legacy capability.** Remove the ignored 876 MB ForestNavigation checkout
   and rely on `make models`, or retain it for offline Go2/legacy Diffusion regression?
4. **Evaluation refactor timing.** Extract a navigator-agnostic benchmark layer now,
   or defer until a second navigator/model supplies a concrete second consumer? Do not
   simply delete the closed-loop modules before this choice.
5. **Historical test contracts.** Replace exact-result tests with a compact frozen
   baseline fixture, or keep the current full artifact-tree assertions through the next
   development milestone?

## Proposed execution order for the next approved round

1. Create and push annotated tag `pre-cleanup-navdiffusion-v0-final` at
   `f29d6a00e0e99fadb9f6c66ce3ca838e849b13c1`.
2. Create a cleanup feature branch; never rewrite published history.
3. Add the compact NavDiffusion V0 baseline summary and redirect docs/tests to it.
4. Extract generic closed-loop runner/metrics/visualization; retain Research Navigation
   runtime, D435i, DIABLO, collision, reset, and logging infrastructure.
5. Remove the exact duplicate JSON and approved one-off artifact trees/entry points.
6. Run repository validation, ROS build/tests, Compose validation, Dataset V0 validation,
   checkpoint hash verification, and model/ROS smoke tests.
7. Commit in small semantic units and submit a normal PR.
8. Only after merge, optionally delete the three fully merged feature branches.
9. Clean ignored local outputs with explicit reviewed paths; never use broad destructive
   commands and never prune or rewrite LFS history in this task.

## Audit conclusion

The repository is not structurally unhealthy. Its large size is overwhelmingly the
intentional Dataset V0 and baseline checkpoint, while the actual source tree is small
and modular. Cleanup should therefore focus on:

1. collapsing NavDiffusion V0 experiment evidence;
2. separating generic Research Navigation evaluation from V0-only gate/test logic;
3. removing reproducible local caches and historical run directories;
4. pruning fully merged branch refs after tagging;
5. preserving Dataset V0, scene provenance, the checkpoint, ROS2/Hydra infrastructure,
   and the shared Research Forest/D435i/DIABLO runtime.

