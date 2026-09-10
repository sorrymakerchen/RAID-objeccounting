# RAID 结构与局部数量监督实验

仅训练集和验证集；三张 V100 分别运行单卡作业。第一批 A0/A1/A2 各100轮，
seed42、448输入、batch2、累积4、AdamW lr/weight_decay=1e-4。保留原数据顺序和翻转策略。
新路由独立随机流，因此新 A0 是直接对照，不要求复现历史 T0 的最优指标。

| 组 | 计数头 | 局部数量损失权重 |
|---|---|---|
| A0 | 原两级 RAID MoE | 0 |
| A1 | 128通道三块残差膨胀卷积，GroupNorm | 0 |
| A2 | 原两级 RAID MoE | 0.1 |
| A3 | 同 A1 | 0.1 |

所有组保留 top150、原始特征、重建特征和文本相似度。A1是整个头替换，参数量也变动，
不能单独归因于路由。局部损失将32×32预测与真值分别求和到4×4和8×8，取
`mean(abs(pred_region-gt_region)/(gt_region+1))` 的两尺度平均，再乘0.1。
冻结编码器不获得梯度，此项不等价于QICA数量提示。

## 服务器提交

从 RAID 根目录运行；先同步本次全部修改，避免不同作业使用不同源码。
配置继续使用现有 configs/server.json，参考CSV路径替换成服务器实际的正式验证预测。
分区和账户放在 sbatch 脚本路径之前。不要设置 CUDA_VISIBLE_DEVICES。

```bash
mkdir -p log
export PARTITION=你的GPU分区
export REFERENCE=outputs/eval_val/val_predictions.csv
for group in A0 A1 A2; do
  sbatch --partition="$PARTITION" --job-name="${group}-smoke" \
    scripts/structure_fsc147.sbatch smoke --group "$group" \
    --config configs/server.json --reference-csv "$REFERENCE" \
    --output-dir "outputs/structure/${group}_smoke"
done
```

三个summary状态均为 `training_smoke_passed` 后提交计时。冒烟包括真实模型反向传播、
4张验证图、训练探针、保存恢复预测一致性；不代替100轮训练。

```bash
for group in A0 A1 A2; do
  sbatch --partition="$PARTITION" --job-name="${group}-profile" \
    scripts/structure_fsc147.sbatch profile --group "$group" \
    --config configs/server.json --reference-csv "$REFERENCE" \
    --output-dir "outputs/structure/${group}_profile"
done
```

激活raid环境后，读取各profile的 `recommended_time_minutes` 作为该作业时限。
`REMAINING_HOURS` 是本轮48小时窗口的剩余时间，扣除已用计时与排队时间；下面36仅为例子。
计时估算包含训练、验证和探针，后续路由或负载可能改变耗时。OOM不自动降配。

```bash
export REMAINING_HOURS=36
for group in A0 A1 A2; do
  minutes=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["recommended_time_minutes"])' \
    "outputs/structure/${group}_profile/summary.json")
  sbatch --partition="$PARTITION" --time="$minutes" --job-name="${group}-train" \
    scripts/structure_fsc147.sbatch train --group "$group" \
    --config configs/server.json --reference-csv "$REFERENCE" \
    --smoke-run "outputs/structure/${group}_smoke" \
    --profile-run "outputs/structure/${group}_profile" --budget-hours "$REMAINING_HOURS" \
    --output-dir "outputs/structure/${group}"
done
```

正式入口检查前置状态、配置、源码/资源哈希、参考CSV和预算；超过预算报错并保留待运行，
不会自动缩短训练。不会自动提交第二批，防止占用超出用户剩余预算的GPU。
所有 .out/.err 位于 log/；Slurm打开日志早于脚本执行，因此必须提前创建目录。

## 汇总与第二批

```bash
python tools/structure_fsc147.py summarize --config configs/server.json \
  --runs outputs/structure/A0 outputs/structure/A1 outputs/structure/A2 \
  --output-dir outputs/structure/comparison
```

汇总重算1286图CSV、检查图ID/文本/数量、配置、源码、权重和共享初始化，
生成 experiments.csv、report.md、summary.json及同图共享色域的对比图。
筛选同时要求：MAE≤A0×0.99、RMSE≤A0×1.01、≥200 MAE≤A0×0.95、0–24 MAE≤A0+0.5。

- A1/A2均通过：对A3 seed42重复smoke/profile/train；完成后汇总增加A3目录。
- A1虽然未满足seed42预设高数量门槛，但整体、类别和效率信号明显，因此登记为新的
  三种子确认假设。对A0/A1分别使用seed43和44重复三步。
- 都未通过：不追加训练。报告失败结果，保留A0。
- 剩余时间不够：维持summary中的pending状态，不将未训练方案判为失败。

恢复使用原目录、原组和原seed：给train命令添加 `--resume 输出目录/latest.pt`。
smoke/profile checkpoint禁止作为正式起点，恢复要求源码和资源一致。
普通评估/预测自动从checkpoint加载 `head_type`，旧checkpoint默认raid/局部损失0。

每组输出best/latest、完整配置与哈希、history、分量损失/路由/数量分组、梯度探针、
best_validation下逐图和分类CSV、重点图与NPZ。空间头路由统计是空值（不适用）。
最终summary含参数量、累计训练时间、推理时间及峰值显存。测试集封存；本地合成测试不能代替GPU验收。

## A0/A1三种子确认实验

为seed43和44分别运行A0/A1。四个正式作业可同时提交，调度器会先运行其中三个。
先分别提交smoke，成功后再提交对应profile：

```bash
for seed in 43 44; do
  for group in A0 A1; do
    sbatch --partition="$PARTITION" --job-name="${group}-s${seed}-smoke" \
      scripts/structure_fsc147.sbatch smoke --group "$group" --seed "$seed" \
      --config configs/server.json --reference-csv "$REFERENCE" \
      --output-dir "outputs/structure/${group}_s${seed}_smoke"
  done
done
```

```bash
for seed in 43 44; do
  for group in A0 A1; do
    sbatch --partition="$PARTITION" --job-name="${group}-s${seed}-profile" \
      scripts/structure_fsc147.sbatch profile --group "$group" --seed "$seed" \
      --config configs/server.json --reference-csv "$REFERENCE" \
      --output-dir "outputs/structure/${group}_s${seed}_profile"
  done
done
```

读取四个profile的 `recommended_time_minutes`，随后提交正式训练：

```bash
export REMAINING_HOURS=36
for seed in 43 44; do
  for group in A0 A1; do
    minutes=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["recommended_time_minutes"])' \
      "outputs/structure/${group}_s${seed}_profile/summary.json")
    sbatch --partition="$PARTITION" --time="$minutes" --job-name="${group}-s${seed}-train" \
      scripts/structure_fsc147.sbatch train --group "$group" --seed "$seed" \
      --config configs/server.json --reference-csv "$REFERENCE" \
      --smoke-run "outputs/structure/${group}_s${seed}_smoke" \
      --profile-run "outputs/structure/${group}_s${seed}_profile" \
      --budget-hours "$REMAINING_HOURS" --output-dir "outputs/structure/${group}_s${seed}"
  done
done
```

六个正式运行完成后按固定顺序汇总：

```bash
python tools/structure_fsc147.py summarize-replication --config configs/server.json \
  --runs outputs/structure/A0 outputs/structure/A1 \
         outputs/structure/A0_s43 outputs/structure/A1_s43 \
         outputs/structure/A0_s44 outputs/structure/A1_s44 \
  --output-dir outputs/structure/replication_comparison
```

汇总要求A1至少赢得2/3个种子的验证MAE，并对三种子平均结果应用原四项门槛。
它输出逐种子结果、均值±样本标准差、逐图三种子平均预测和最终选择。
增加seed44只改变实验包装器；模型、损失、数据、训练代码、预训练资源及共享投影初始化必须一致。

```bash
python -m unittest discover -s tests -v
```

## B0/B1检索输入消融

A1三种子确认通过后，使用空间头执行检索输入的单因素实验。六组均从头训练100轮，
不从A1 checkpoint继续训练，也不复用旧A1结果，确保B0/B1使用完全相同的当前源码。

| 组 | 进入空间计数头的输入 | 保持不变的内容 |
|---|---|---|
| B0 | 原始特征、重建特征、文本相似度、150通道匹配图 | 空间头、通道数、参数量、训练设置 |
| B1 | 原始特征、文本相似度；重建特征和匹配图置零 | 空间头、通道数、参数量、训练设置 |

B1仍计算候选索引以便诊断，但候选、重建和匹配值不进入预测。两组只允许
`retrieval_input_mode` 不同；每个种子还校验可训练计数模块初始状态及共享投影哈希一致。

先提交六个冒烟作业。调度器会按照三张GPU的可用情况分批运行：

```bash
mkdir -p log
export PARTITION=你的GPU分区
export REFERENCE=outputs/eval_val/val_predictions.csv
for seed in 42 43 44; do
  suffix=""
  if [ "$seed" != 42 ]; then suffix="_s${seed}"; fi
  for group in B0 B1; do
    sbatch --partition="$PARTITION" --job-name="${group}-s${seed}-smoke" \
      scripts/structure_fsc147.sbatch smoke --group "$group" --seed "$seed" \
      --config configs/server.json --reference-csv "$REFERENCE" \
      --output-dir "outputs/structure/${group}${suffix}_smoke"
  done
done
```

确认六个 `summary.json` 均为 `training_smoke_passed` 后提交计时：

```bash
for seed in 42 43 44; do
  suffix=""
  if [ "$seed" != 42 ]; then suffix="_s${seed}"; fi
  for group in B0 B1; do
    sbatch --partition="$PARTITION" --job-name="${group}-s${seed}-profile" \
      scripts/structure_fsc147.sbatch profile --group "$group" --seed "$seed" \
      --config configs/server.json --reference-csv "$REFERENCE" \
      --output-dir "outputs/structure/${group}${suffix}_profile"
  done
done
```

确认六个计时状态为 `profile_complete_not_formal_training`，并设置真实剩余预算后提交训练：

```bash
export REMAINING_HOURS=48
for seed in 42 43 44; do
  suffix=""
  if [ "$seed" != 42 ]; then suffix="_s${seed}"; fi
  for group in B0 B1; do
    run="outputs/structure/${group}${suffix}"
    minutes=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["recommended_time_minutes"])' \
      "${run}_profile/summary.json")
    sbatch --partition="$PARTITION" --time="$minutes" --job-name="${group}-s${seed}-train" \
      scripts/structure_fsc147.sbatch train --group "$group" --seed "$seed" \
      --config configs/server.json --reference-csv "$REFERENCE" \
      --smoke-run "${run}_smoke" --profile-run "${run}_profile" \
      --budget-hours "$REMAINING_HOURS" --output-dir "$run"
  done
done
```

所有 `.out` 和 `.err` 写入 `log/`。六个正式运行完成后按以下固定顺序汇总：

```bash
python tools/structure_fsc147.py summarize-retrieval --config configs/server.json \
  --runs outputs/structure/B0 outputs/structure/B1 \
         outputs/structure/B0_s43 outputs/structure/B1_s43 \
         outputs/structure/B0_s44 outputs/structure/B1_s44 \
  --output-dir outputs/structure/retrieval_comparison
```

结论在运行前固定：

- `retrieval_beneficial`：B0至少赢2/3个种子的MAE，且相对B1通过原四项筛选门槛；保留B0。
- `retrieval_harmful`：B1至少赢2/3个种子的MAE，且相对B0通过原四项筛选门槛；采用B1。
- `practically_equivalent`：三种子均值差异同时不超过MAE 1%、RMSE 1%、高数量MAE 5%及低数量MAE绝对值0.5；采用参数相同但后续可移除检索计算的B1。
- `inconclusive`：其余结果；暂时保留B0，不依据单张图或单个种子选择。

汇总器重新计算全部1286张验证图，核对提示、类别、真值、checkpoint、源码、资源和初始化，
生成逐种子、三种子均值及逐图配对结果。测试集继续封存。
