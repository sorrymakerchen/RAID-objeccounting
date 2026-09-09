# FSC147 不重训定位诊断

本工具只读取正式训练的最佳 checkpoint 和验证集，固定输入 448×448、K=150、
checkpoint 的 epoch 与路由阶段。不会更新参数，不使用测试集，不进行分块或变分辨率推理。

本次已知参考结果为验证集 1286 张、MAE 31.8462676262、RMSE 96.8439342990，
`best.pt` 对应 `epoch=33`。入口校验这些条件以及 checkpoint 的完整训练配置，
拒绝本机 8 图过拟合权重。诊断权重必须是服务器实际正式训练得到的文件。

## 准备输入

在服务器 RAID 项目根目录准备：

- `outputs/fsc147_text/best.pt`：正式最佳权重。
- `configs/server.json`：使用训练时的配置和相同三份预训练权重；可修改资源路径和 workers，
  保持 batch size 与 checkpoint 配置一致。
- `outputs/eval_val/val_predictions.csv`：之前导出的完整验证集参考预测。
- 数据根目录内的 `ImageClasses_FSC147.txt`：官方类别名称，不根据文本猜测类别。

三个权重、DINOv2 代码、CLIP 和 conda 环境沿用训练时的版本。无需新增 Python 依赖。
工具自动记录资源绝对路径、文件哈希、原 checkpoint 配置、生效配置和 torch 版本。
历史 checkpoint 没有保存预训练文件哈希，因此本次哈希用于留档，不能独自证明与历史文件一致。

## 第一步：三图冒烟

从项目根目录提交；脚本会在计算节点固定激活 `raid` conda 环境：

```bash
sbatch --partition=YOUR_GPU_PARTITION scripts/diagnose_fsc147.sbatch \
  --config configs/server.json \
  --checkpoint outputs/fsc147_text/best.pt \
  --reference-csv outputs/eval_val/val_predictions.csv \
  --smoke --output-dir outputs/diagnosis_smoke
```

如果集群需要账户，在脚本文件名前增加 `--account=YOUR_ACCOUNT`。
脚本默认 1 GPU、8 CPU、32GB 内存和 2 小时；固定加载
`/public/home/yzhang712/anaconda3` 下的 `raid` 环境，不读取提交端的 `RAID_PYTHON`
或 `CONDA_PREFIX`。
保留 Slurm 分配的 `CUDA_VISIBLE_DEVICES`。

冒烟只处理 `3425.jpg`、`3427.jpg`、`935.jpg`，仍要求提供完整参考 CSV。
检查作业输出以及 `outputs/diagnosis_smoke/summary.json`：

```text
status = smoke_passed_not_full_validation
reproduction.passed = true
```

查看三张 `samples/<image_id>/comparison.png`，确认图像、候选框和点标注对齐。
提交前在项目根目录执行 `mkdir -p log`；Slurm 在脚本执行前打开日志文件。
日志为 `log/slurm-raid-diagnose-JOB_ID.out/.err`，程序日志为诊断输出目录的 `run.log`。

## 第二步：完整验证诊断

三图检查通过后，使用新的输出目录提交完整作业：

```bash
sbatch --partition=YOUR_GPU_PARTITION scripts/diagnose_fsc147.sbatch \
  --config configs/server.json \
  --checkpoint outputs/fsc147_text/best.pt \
  --reference-csv outputs/eval_val/val_predictions.csv \
  --output-dir outputs/diagnosis_full
```

已有 GPU allocation 时，可以把上述 sbatch 前缀替换为 `python tools/diagnose_fsc147.py`。
不要在登录节点直接执行 GPU 推理。
工具不覆盖已有诊断结果；重跑时使用新的输出目录。

复现检查同时检查逐图 ID/文本/数量、MAE/RMSE 和逐图预测差值，绝对容差为 `1e-3`。
逐图检查避免聚合误差相抵掩盖问题。验证失败会保存差异 CSV 和报告，返回退出码 2，
不继续生成提示对照或归因结论。其他输入/运行错误返回 1，成功返回 0。

提示对照复用每张图在基线验证中的完整 batch、同批样本和位置，只替换目标图文本。
这样可以隔离文本变化，并避免 GPU 算子因 batch shape 改变而造成数值路径差异。

重点图由参考 CSV 确定：三个指定图、误差最大的十张图，以及六个数量区间中
相对误差最小的各两张，去重后按文件名排序。当前参考 CSV 对应 **23 张**。
相对误差定义为 `absolute_error / max(target, 1)`；并列时按图像 ID 排序。
数量区间为 0–24、25–49、50–99、100–199、200–499、≥500。

对每张重点图比较原始描述、`a photo of {官方类别名称}`、`the elephants` 三个提示。
三种提示均用单图推理，并核对原始提示与完整验证批次的结果一致；因此不会把
batch size 改变误当作文本效果。对照文本没有零数量真值假设，其结果不进入基准指标。

## 输出内容

```text
outputs/diagnosis_full/
  report.md                    # 指标、证据表和解释边界
  summary.json                 # 复现结论、配置、哈希、显存与路由汇总
  predictions.csv              # 全验证集预测、空间诊断指标及路由
  reference_differences.csv     # 每张图与历史预测的差值
  by_category.csv              # 分类误差、空间指标与路由汇总
  by_count.csv                 # 按数量区间汇总
  routes.csv                   # 每张图两级路由概率和实际混合权重
  samples.json                 # 固定样本清单及入选原因
  prompt_comparisons.csv        # 提示变化、候选重合率和数量变化
  samples/<image_id>/
    comparison.png             # 三行提示、五列证据的总览图
    original.png / points.png
    target_density.png
    original_* / template_* / control_*
    diagnostics.json           # 颜色范围、数量、空间指标和路由
```

每种提示导出相似度 PNG、候选覆盖 PNG、密度 PNG、压缩 NPZ。
NPZ 包括原始网格相似度/密度、原图尺寸密度/相似度、真实密度、原始点、候选索引、
原图坐标矩形、候选网格 mask 和点邻域 mask。路由数组保存在 JSON/CSV。

同一张图的所有提示共用相似度颜色范围，真实和预测密度共用密度范围。
相似度是原始余弦值；显示密度的单位为每个原图像素的计数质量。
密度恢复尺寸后保持积分。候选矩形为原图坐标 `x0,y0,x1,y1`，右下边界为排他边界。

## 如何理解诊断值

- `candidate_gt_mass_fraction`：候选 patch 内的真实密度质量占总质量的比例。
  与 `candidate_area_fraction` 对照观察，但 top-k 并不要求覆盖每个实例。
- `candidate_point_cell_hit_rate`：候选 patch 中包含真实点的比例。
- `candidate_neighborhood_hit_rate`：候选 patch 落在真实点所在格或一格邻域内的比例。
- `neighborhood_area_fraction`：该点邻域占全网格的比例；≥90% 时标记为区分能力有限。
- `pred_mass_inside/outside`：预测密度在点邻域内外的质量；它不是严格的前景/背景分割指标。
- `stage1_selected_*`：第一层实际 top-2 分派；第二层为稠密混合，
  `stage2_dominant_*` 仅表示最高权重专家，不表示其余专家未执行。

FSC147 的 `1989.jpg` 有一个 x=408.00000000000006 的点，图像宽度为 408。
工具允许不超过 `1e-6` 像素的边界舍入误差，只在诊断网格投影时截到边界，
不修改原始标注或真实数量，并记录 `point_boundary_roundoff_count`。
超过容差的异常坐标会明确报错。

根据证据调查文本定位、候选覆盖、密度回归或路由行为；热力图及相关性统计本身
不证明因果，不据此自动调整模型。后续方法改动和参数选择继续使用验证集。

## 测试与已验证范围

```bash
python -m unittest discover -s tests -v
```

本地测试覆盖诊断开关不改预测与 checkpoint、实际混合权重、候选坐标、密度积分、
标注与推理解耦、共享颜色范围、样本选择、报告输出和复现失败停止流程。
报告工作流使用明确标识的合成单元测试，不替代服务器正式 checkpoint 验证。
本机已核对全部 1286 张验证图的参考记录和点投影边界；正式权重推理由服务器作业完成。
