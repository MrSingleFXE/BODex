# Sharpa 完整复现：V3 / 34 球 / α=1 / kp=12

输入 DGN_2k 场景 → BODex 合成三阶段候选 → Bench 物理筛选及成功标签收集 → DexLearn nflow 训练 → Bench 评测模型生成的全部候选。

## 三个仓库与环境

```bash
git clone https://github.com/MrSingleFXE/BODex.git
git clone https://github.com/MrSingleFXE/DexGraspBench.git
git clone https://github.com/MrSingleFXE/DexLearn.git
```

分别按三个仓库原 README 安装环境及 third_party/submodule；不能仅安装这份配置就运行。BODex 需要 CUDA/PyTorch 和编译好的 cuRobo，DexLearn 需要 MinkowskiEngine 等原依赖；Bench 使用 MuJoCo **3.3.2**（`MjSpec` 接口）、原仓库的力闭合求解依赖。合成、学习、仿真建议使用独立 Python 环境。辅助脚本要求 Python ≥3.10、NumPy、PyYAML；确定性表面采样要求 trimesh 的 `sample_surface` 和 `sample_surface_even` 支持 `seed` 参数。参考实验的具体环境另见 `VALIDATION.md`。

用原项目的数据下载说明准备 DGN_2k：网格/分解网格、`scene_cfg`、`valid_split/{train,test}.json`，以及 `vision_data/azure_kinect_dk` 的场景匹配点云。**不使用重新随机划分**：官方 1,918 train / 479 test 对象；54,148 个 tabletop 场景，test 10,892 个。训练期间 eval loader 使用 test split，没有额外 validation holdout；结果不能当作独立调参留出集。

令 `DATA` 为本机 DGN_2k 完整目录的绝对路径，三个仓库链接到同一数据，避免保存的相对场景路径失效（已有目录时先核对，不覆盖）。

```bash
export DATA=/absolute/path/to/DGN_2k
mkdir -p BODex/src/curobo/content/assets/object DexGraspBench/assets/object DexLearn/assets/object
ln -s "$DATA" BODex/src/curobo/content/assets/object/DGN_2k
ln -s "$DATA" DexGraspBench/assets/object/DGN_2k
ln -s "$DATA" DexLearn/assets/object/DGN_2k
(cd BODex && python example_grasp/download_sharpa_assets.py)
(cd DexGraspBench && python script/download_sharpa_assets.py)
```

两个下载器获取相同的 23 个官方网格，固定来源 commit 并逐个校验 SHA-256。源码仓库保留 URDF/MJCF、许可证、清单，网格不重复提交。所有下面命令均在对应仓库根目录执行，使用对应 Python 环境。`RUN`、`LABELS`、`EVAL` 使用绝对路径；输出目录必须是新的。

## 1. 数据合成（BODex 环境）

```bash
export RUN=/absolute/path/to/runs/sharpa_v3_s34
python example_grasp/sharpa_multi_gpu.py --output "$RUN/synthesis" --gpus 0,1,2,3,4,5,6,7
```

先检查环境可使用 `--scene-limit 2 --gpus 0 --parallel-worlds 2`，并使用另一个输出目录。每场景 20 候选；全量应为 54,148 个原始 NPY、1,082,960 个候选。原始数据在 `$RUN/synthesis/grasp_data`。GPU 分片沿用原数据加载器固定随机种子后的 scene 顺序；每 GPU 有独立配置、日志、wrist seed 哈希审计。

22 个关节，29D = xyz(m) + wxyz + 22 个 rad。保存 pregrasp、grasp、squeeze 三姿态。V3 拇指 CMC FE=1、MCP FE=-0.25，四指适度展开；配置保留原 128 个物体表面手腕采样位置和 jitter。α=1：`squeeze = grasp + (grasp - pregrasp)`，关节限位裁剪已开启。完整数组、34 球位置见配置。

## 2. 物理筛选与标签（Bench 环境）

```bash
python script/run_sharpa.py --raw "$RUN/synthesis/grasp_data" --output "$RUN/screening" \
  --expected-scenes 54148 --candidates-per-scene 20 --workers 192 --collect
```

新的 MJCF 默认所有位置执行器 **kp=12，dampratio=0.9**，保留各关节力矩上限、指腹接触参数。kp 是仿真位置伺服刚度，不改变 BODex 的合成目标；更大 kp 不表示取消力矩上限。MuJoCo arena=256 MiB/worker，192 个 worker 需要充足 RAM，可降低 `--workers`。

脚本逐项检查 raw→graspdata→evaluation 相对路径覆盖，success flag 与 succgrasp 一致，collect 行数一致；进程/仿真异常不算普通抓取失败。只有通过后才写 `coverage.json`。成功标签目录为 `$RUN/screening/succ_collect`。

## 3. 学习（DexLearn 环境）

```bash
export LABELS="$RUN/screening/succ_collect"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_sharpa.sh "$LABELS" sharpa_v3_s34_kp12_alpha1
```

入口先核对 Bench 覆盖审计、所有标签的手身份/关节顺序/数值和官方对象 split，并抽查实际 train/eval/test loader。nflow 50,000 steps，per-GPU batch=32、8 GPU global batch=256，lr=0.001→0.0001，10k 保存一次，1k eval 一次。不同 GPU 数会改变 global batch，不能直接视为同一训练设置。

输出 `output/bodex_tabletop_sharpa_right_nflow_sharpa_v3_s34_kp12_alpha1`；包含 `.hydra`、`ckpts`、离线 metrics、DDP 同步验证。断点恢复可直接用 `python -m torch.distributed.run ... --module dexlearn.train_ddp` 和原配置 `resume=true`，**保留原 max_iter**；新包装脚本故意要求新名称，避免误覆盖。

## 4. 模型评测（DexLearn 环境，CPU 可运行）

```bash
python scripts/evaluate_sharpa.py \
  --experiment bodex_tabletop_sharpa_right_nflow_sharpa_v3_s34_kp12_alpha1 \
  --bench /absolute/path/to/DexGraspBench \
  --bench-python /absolute/path/to/bench/env/bin/python \
  --output /absolute/path/to/new_evaluation --jobs 5 --workers 40
```

依次使用 10k/20k/30k/40k/50k checkpoint，支持 `--steps 50000` 单独评测。每个 checkpoint 10,892 test scenes ×100，`test_topk=100`，**全部物理评测**；直接执行模型预测的三姿态，不额外 squeeze、不裁剪越界预测（按失败计）。五任务并发时最多 200 physics workers，默认 jobs=1；按 CPU/RAM 调整。

主指标 `all_success_rate` 为候选成功率（0–1）；概率 Top-10 是另一个指标，不能代替全候选成功率。排名分析在完整覆盖后执行，输出小型 summary 与本地 CSV/NPZ。数据顺序、软件/硬件版本、GPU 分片数仍可能影响逐样本结果；这是可运行方法复现，不承诺数值逐位一致。

## 实现范围与不包含内容

发布最新官方模型适配、V3/34 球配置、QP 修复、squeeze 限位、确定性采样、Bench 转换/控制/接触修复、覆盖与标签审计、DDP 训练、checkpoint 身份保护、全候选与 Top-k 评测、运行说明及汇总指标。

不提交生成候选、成功标签、数据集/点云、模型权重、优化中间帧、USD/视频/截图包、日志和缓存；官方手网格由校验下载器获取。早期 V1/V2/α2、kp/球数探索的服务器绝对路径启动器和重复配置没有原样复制；其方法/结果摘要在 `SHARPA.md`，最新流水线替代了历史专用调度脚本。原始研究中的本地专用 Viser 界面与浏览数据包不属于此核心发布。

## 解释结果

筛选 69.68%；50k 模型全候选 57.87%，概率 Top-10 75.88%。V2 对照为 65.55%、57.02%、73.98%。全量对照同时改变初值与球模型，不能单独归因于球数。固定 V3/kp12/α1 的同场景球模型小实验：147 球 66.555%，34 球 69.765%。均是仿真结果，没有真实机器人验证。
