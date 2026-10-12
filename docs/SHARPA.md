# Sharpa Wave right hand: V3 / 34 spheres

This fork adds a floating, bare **right** Sharpa Wave hand for tabletop grasp synthesis. It does not add an arm, left hand, motion-planning configuration, or hardware controller.

## Run

Install BODex and prepare its DGN_2k tabletop object assets using the upstream README. Then, from the repository root:

```bash
python example_grasp/download_sharpa_assets.py
python example_grasp/sharpa_synthesis.py -c "$PWD/src/curobo/content/configs/manip/sharpa_right/tabletop.yml" -f output/sharpa_v3_s34 -m npy -w 20
```

The configuration generates 20 candidates per scene. For a small first run, copy the YAML and set `world.end: 2`. Use a fresh output directory. Squeeze uses the existing alpha=1 extrapolation with joint-limit clipping enabled for this configuration.

## Model and configuration

The official model is pinned to Sharpa commit `0d19cac602f46456b819e4b6a2c09a74982c9a3e`, `wave_01/right_sharpa_wave`. The adapted URDF retains the official geometry, kinematic axes, origins and inertial properties; meshes are downloaded unchanged with SHA-256 verification. Sharpa license and notice are retained beside the model. The URDF has adapted relative mesh paths and conservative joint bounds: it is not an unmodified official model or a hardware-certified workspace.

There are 22 independent joints. Output poses use xyz in metres, wxyz quaternions and joint angles in radians (29 dimensions), in the emitted `joint_names` order. The root is `right_hand_C_MC`. Do not apply Shadow's joint reorder or palm offset.

The V3 seed opens and opposes the thumb and slightly spreads the ring and little fingers. Thumb CMC FE is 1 rad; thumb MCP FE is -0.25 rad. See the complete ordered `seeder_cfg.q` in the YAML. The existing object-surface wrist sampling strategy is retained (128 sampled positions, original jitter and reference transform).

The collision proxy uses **34 spheres, including 4 palm spheres**. Sphere centres/radii are a subset of the original fit; all five fingertip contact anchors remain. Nonoverlap selection applies within rigid groups; moving adjacent links can still overlap. Contact optimisation uses the fingertip elastomer geometry. A coarser sphere proxy is not a guarantee against mesh penetration.

Two code changes are required: a separately registered `batch_reluqp_noalias` solver avoids an input/output alias in batched matrix multiplication; optional `clamp_squeeze_to_joint_limits` clips extrapolated squeeze joints. Existing hand configurations retain their solver and clipping defaults.

## Full pipeline results

Completed October 9, 2026. Object split: 1,918 training / 479 test objects (80/20); no additional validation holdout. Training validation uses the test split as in the established workflow.

| Metric | Previous V2 / original spheres | V3 / 34 spheres |
|---|---:|---:|
| Synthesis candidates passing physical screening | 65.55% | 69.68% |
| Learned 10k checkpoint, all candidates | 34.93% | 41.37% |
| Learned 20k checkpoint, all candidates | 47.54% | 50.19% |
| Learned 30k checkpoint, all candidates | 52.88% | 51.70% |
| Learned 40k checkpoint, all candidates | 54.24% | 56.79% |
| Learned 50k checkpoint, all candidates | 57.02% | 57.87% |

Screening covers 54,148 scenes × 20 = 1,082,960 candidates. Training uses nflow, 8 GPUs, global batch 256, 50,000 steps. Each learned checkpoint is physically evaluated on 10,892 test scenes × 100 candidates, all evaluated, without extra squeeze or pose postprocessing. These are candidate success rates, not scene success rates. The final model's probability-ranked Top-10 rate is 75.88%, a separate metric.

Both full runs use **uniform kp=12, damping ratio=0.9, alpha=1**, preserving per-joint torque caps. kp is a **DexGraspBench controller parameter**, not a BODex synthesis parameter. These full-run gains combine initialisation and sphere changes, and are not an isolated sphere ablation or multiple-training-seed estimate. The matched 1,000-scene sphere ablation at fixed V3/kp12/alpha1 yielded 66.555% for 147 spheres and 69.765% for 34 spheres.

The full experiments used deterministic mesh-surface sampling; `example_grasp/sharpa_synthesis.py` now publishes that sampler and seed audit. Stock upstream sampling may produce different individual candidates. Rates above are archived experimental results, not guarantees for a fresh run.

## Companion repositories

The companion integrations are now published: [DexGraspBench](https://github.com/MrSingleFXE/DexGraspBench) and [DexLearn](https://github.com/MrSingleFXE/DexLearn). See [the complete reproduction guide](SHARPA_PIPELINE.md) for asset setup, screened-label collection, eight-GPU training and all-candidate evaluation. Trained weights and generated datasets are not bundled.

## Development history

1. Established the upstream Shadow synthesis → physical screening → learning → evaluation pipeline.
2. Integrated the official bare-right Sharpa model; checked FK against MuJoCo, joint order, limits, contact geometry, torque caps and pose conventions. Fixed the QP alias issue and added cross-stage identity checks.
3. Built Viser inspections of raw candidates, optimisation frames, success/failure labels, isolated sphere models, and aligned/overlaid Shadow–Sharpa seed poses.
4. Explored alpha=1/2 squeeze, thumb CMC and MCP initialisation, finger spread (V1 → V2 → V3), uniform and joint-family kp sweeps, and seven sphere layouts (147 → 34).
5. Standardised the object split, migrated server storage with compatibility links, and added complete candidate/label/point-cloud audits and automatic multi-GPU training / CPU evaluation.
6. Selected the current V3 / 34-sphere / alpha1 synthesis configuration and kp12 downstream controller, and completed the full-scale pipeline above.

Original BODex authorship and citations remain in the main README. Official Sharpa assets: https://github.com/sharpa-robotics/sharpa-urdf-usd-xml
