# BODex：Sharpa 灵巧手抓取流程

本 fork 在原作者 BODex 的基础上接入 **Sharpa Wave 裸右手（22 个关节）**，并打通数据合成、物理筛选、模型训练与评测流程。当前版本使用 **V3 关节初值、34 个碰撞球、α=1、仿真控制器统一 kp=12**，包含关节顺序校验、squeeze 限位、筛选覆盖率检查和多 GPU 训练等实现。

本节介绍本 fork 的新增内容；原作者的项目介绍、安装说明和引用信息完整保留在后半部分。

## 配套仓库

- **[BODex（本仓库）](https://github.com/MrSingleFXE/BODex)**：Sharpa 模型适配、初值与球模型、抓取候选合成。
- **[DexGraspBench fork](https://github.com/MrSingleFXE/DexGraspBench)**：Sharpa 控制器、物理筛选、成功标签收集和候选成功率统计。
- **[DexLearn fork](https://github.com/MrSingleFXE/DexLearn)**：Sharpa 数据加载、nflow 多 GPU 训练和模型候选采样评测。

## 简明复现流程

**DGN_2k 场景 → 合成候选 → 仿真筛选与标签收集 → 模型训练 → 全候选物理评测。**

先按三个仓库的安装说明配置各自环境，准备共享的 DGN_2k 场景、物体资产和匹配点云，使用官方 80/20 对象划分。资产路径、下载方式和全部参数见 **[完整中文复现指南](docs/SHARPA_PIPELINE.md)**。以下命令在各步骤对应的仓库根目录、相应 Python 环境中运行；`RUN` 设置为同一个绝对输出路径，示例中的其他绝对路径也需替换。

1. **合成（BODex）**：下载官方网格，然后用 V3／34 球配置生成每场景 20 条候选。

   ```bash
   export RUN=/absolute/path/to/runs/sharpa_v3_s34
   python example_grasp/download_sharpa_assets.py
   python example_grasp/sharpa_multi_gpu.py --output "$RUN/synthesis" --gpus 0,1,2,3,4,5,6,7
   ```

2. **筛选（DexGraspBench）**：用 kp=12 控制器评测全部候选，检查覆盖率并收集成功标签。

   ```bash
   python script/download_sharpa_assets.py
   python script/run_sharpa.py --raw "$RUN/synthesis/grasp_data" --output "$RUN/screening" \
     --expected-scenes 54148 --candidates-per-scene 20 --workers 192 --collect
   ```

3. **训练（DexLearn）**：使用筛选后的标签进行八卡 nflow 训练，共 50,000 steps。

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_sharpa.sh \
     "$RUN/screening/succ_collect" sharpa_v3_s34_kp12_alpha1
   ```

4. **评测（DexLearn 调用 Bench）**：对模型生成的每场景 100 条候选全部做物理评测，可在 CPU 上运行。

   ```bash
   python scripts/evaluate_sharpa.py \
     --experiment bodex_tabletop_sharpa_right_nflow_sharpa_v3_s34_kp12_alpha1 \
     --bench /absolute/path/to/DexGraspBench --bench-python /absolute/path/to/bench/env/bin/python \
     --output "$RUN/evaluation" --jobs 5 --workers 40
   ```

上述是全量配置；首次运行建议先按完整指南做两场景检查，并按 GPU、CPU 和内存资源调整并行数。已完成实验的合成候选筛选成功率为 **69.68%**，50k 模型的全候选成功率为 **57.87%**；具体协议、优化历史及结果解释见 [Sharpa 配置与实验说明](docs/SHARPA.md)，发布检查范围见 [验证记录](docs/VALIDATION.md)。数据集、生成数据、训练权重和大体积可视化输出不包含在仓库中。

---

# 原作者 README（以下保留原文）

## News (April 2026):

We have open-sourced the follow-up work Dexonomy [RSS 2025] to [here](https://github.com/JYChen18/Dexonomy). For grasp generation, I highly recommend using Dexonomy — it offers easier customization and better grasp synthesis results.




# BODex

A GPU-based efficient pipeline for dexterous grasp synthesis built on [cuRobo](https://github.com/NVlabs/curobo/tree/main), proposed in *[ICRA 2025] BODex: Scalable and Efficient Robotic Dexterous Grasp Synthesis Using Bilevel Optimization*.

[Project page](https://pku-epic.github.io/BODex/) ｜ [Paper](https://arxiv.org/abs/2412.16490) ｜ [Dataset](https://huggingface.co/datasets/JiayiChenPKU/BODex) ｜ [Benchmark code](https://github.com/JYChen18/DexGraspBench) ｜ [Learning code](https://github.com/JYChen18/DexLearn)

## Introduction
### Main Features
- **Grasp Synthesis**: Generate force-closure grasps for floating dexterous hands, such as the Shadow, Allegro, and Leap Hand.
- **Trajectory Planning**: Plan collision-free approaching trajectories with the table for hands mounted on robotic arms, e.g., UR10e + Shadow Hand systems.

### Highlights
- **Efficient**: Capable of synthesizing millions of grasps per day using a single NVIDIA 3090 GPU.
- **Generalizable**: Supports different hands and a wide range of objects.

### Follow-up Work
Some projects *make modifications* on our BODex pipeline to synthesize large-scale datasets of grasping poses, such as [DexGraspNet 2.0](https://pku-epic.github.io/DexGraspNet2.0/) and [GraspVLA](https://pku-epic.github.io/GraspVLA-web/).

Looking for more diverse and higher-quality dexterous grasps? Check out [Dexonomy](https://pku-epic.github.io/Dexonomy).



## Getting Started
1. **Install git lfs**: Before `git clone` this repository, please make sure that the git lfs has been installed by `sudo apt install git-lfs`.

2. **Install the Python environment**:
```
conda create -n bodex python=3.10
conda activate bodex

conda install pytorch==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia 

pip install -e . --no-build-isolation  

pip install usd-core 
conda install pytorch-scatter -c pyg
conda install coal==3.0.1 -c conda-forge   # https://github.com/coal-library/coal

pip uninstall numpy
pip install numpy==1.26.4

cd src/curobo/geom/cpp
python setup.py install    # install coal_openmp_wrapper
```

3. **Prepare object assets**: Download our pre-processed object assets `DGN_2k_processed.zip` from [Hugging Face](https://huggingface.co/datasets/JiayiChenPKU/BODex) and organize the unzipped folders as below. 
```
src/curobo/content/assets/object/DGN_2k
|- processed_data
|  |- core_bottle_1a7ba1f4c892e2da30711cdbdbc73924
|  |_ ...
|- scene_cfg
|  |- core_bottle_1a7ba1f4c892e2da30711cdbdbc73924
|  |_ ...
|- valid_split
|  |- all.json
|  |_ ...
```
Alternatively, new object assets can be pre-processed using [MeshProcess](https://github.com/JYChen18/MeshProcess).
   
4. **Synthesize grasp poses**: Each synthesized grasping data point includes a pre-grasp, a grasp, and a squeeze pose. 
```
# Single GPU version
CUDA_VISIBLE_DEVICES=7 python example_grasp/plan_batch_env.py -c sim_shadow/fc.yml -w 40 

# Debugging. The saved USD file has the whole optimization process, while the intermediate gradient on each contact point is denoted by the purple line.
CUDA_VISIBLE_DEVICES=7 python example_grasp/plan_batch_env.py -c sim_shadow/fc.yml -w 1 -m usd -debug -d all -i 0 1

# Multiple GPU version
python example_grasp/multi_gpu.py -c sim_shadow/fc.yml -t grasp -g 0 1 2 3 
```
We can also **synthesize approaching trajectories** that are collision-free with the table for hands mounted on arms, e.g., UR10e+Shadow Hand systems.
 ```
# Single GPU version
CUDA_VISIBLE_DEVICES=7 python example_grasp/plan_mogen_batch.py -c sim_shadow/tabletop.yml -t grasp_and_mogen

# Multiple GPU version
python example_grasp/multi_gpu.py -c sim_shadow/tabletop.yml -t grasp_and_mogen -g 0 1 2 3 
```
On a single GPU, the grasp synthesis supports parallizing different objects, but the motion planning only supports parallizing different trajectories for the same object.

5. **(Optional) Visualize synthesized poses**:
```
python example_grasp/visualize_npy.py -c sim_shadow/fc.yml -p debug -m grasp
```

6. **Evaluate grasp poses and filter out bad ones**: please see [DexGraspBench](https://github.com/JYChen18/DexGraspBench).


## License

The code from cuRobo is used under the NVIDIA license. All other parts of this work are licensed under [CC BY-NC 4.0][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png

## Citation

If you found this repository useful, please consider to cite the following works,

```
@article{chen2024bodex,
  title={BODex: Scalable and Efficient Robotic Dexterous Grasp Synthesis Using Bilevel Optimization},
  author={Chen, Jiayi and Ke, Yubin and Wang, He},
  journal={arXiv preprint arXiv:2412.16490},
  year={2024}
}

@misc{curobo_report23,
      title={cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation},
      author={Balakumar Sundaralingam and Siva Kumar Sastry Hari and Adam Fishman and Caelan Garrett
              and Karl Van Wyk and Valts Blukis and Alexander Millane and Helen Oleynikova and Ankur Handa
              and Fabio Ramos and Nathan Ratliff and Dieter Fox},
      year={2023},
      eprint={2310.17274},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
