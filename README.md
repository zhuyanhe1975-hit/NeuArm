# NeuArm

NeuArm：基于 mjlab 的工业机械臂强化学习/神经网络控制工程。

目标：在 MuJoCo(Warp) 仿真中对 ABB IRB2400 进行动态轨迹跟踪，控制结构为 **变增益 PID/PD + 前馈（重力补偿/bias/CTFF）+ 残差补偿(神经网络/RL)**。

仓库：`git@github.com:zhuyanhe1975-hit/NeuArm.git`

## 目录结构

- `assets/abb_irb2400/mjcf/irb2400_mjlab.xml`：IRB2400 MJCF（含 `site name="ee"`）
- `src/irb2400_rl/task/env_cfg.py`：环境配置（观测/奖励/终止/随机化）
- `src/irb2400_rl/task/commands.py`：关节轨迹命令生成（quintic 段轨迹）
- `src/irb2400_rl/task/actions.py`：核心控制器（PD/PID + gravity/CTFF + residual）
- `scripts/train_irb2400_ppo.py`：PPO 训练入口（rsl_rl）
- `scripts/eval_irb2400_tracking.py`：评估末端误差（mm）

## 环境准备

```bash
conda activate mjwarp_env
export MJLAB_SRC=/home/yhzhu/AI/mjlab/src
```

说明：
- Warp 的 kernel cache 会自动写到本仓库的 `.warp_cache/`（避免写 `~/.cache` 的权限问题）。
- 当前默认：仿真步长 `dt=1ms`，控制/策略更新周期 `5ms`（`decimation=5`）。

## 训练（PPO）

```bash
conda run -n mjwarp_env python scripts/train_irb2400_ppo.py --device cuda:0 --num-envs 1024 --max-iterations 2000
```

输出日志：`logs/rsl_rl/neuarm_irb2400_tracking/<timestamp>/`

## 评估（末端误差 mm）

不加载 checkpoint（仅“PID+CTFF”，残差=0）：

```bash
conda run -n mjwarp_env python scripts/eval_irb2400_tracking.py --device cuda:0 --steps 2000
```

加载训练好的 checkpoint：

```bash
conda run -n mjwarp_env python scripts/eval_irb2400_tracking.py --device cuda:0 --checkpoint logs/rsl_rl/neuarm_irb2400_tracking/<run>/model_*.pt --steps 2000
```

## 调参入口

- 控制器参数：`src/irb2400_rl/task/env_cfg.py` 里 `ResidualComputedTorqueActionCfg(...)`
- 前馈模式：`ff_mode="none" | "gravcomp" | "bias" | "ctff"`（建议先 `bias`，再开 residual/ctff）
- 轨迹幅度/频率：`src/irb2400_rl/task/env_cfg.py` 里 `JointTrajectoryCommandCfg(...)`
