# Milestone Vault

这个文件夹用于“长期封存”关键里程碑：包含当时的 checkpoint、评估图、关键指标摘要，以及可一键 `train / eval / replay` 的脚本。

## 目录

- `frictionless_damping0_model100/`：无摩擦（damping=0）最佳结果（TCP ~0.15mm）
- `friction_residual_model999/`：有摩擦（damping+frictionloss）最佳结果（TCP ~6.24mm，Residual p95 ~8.57 N·m）

