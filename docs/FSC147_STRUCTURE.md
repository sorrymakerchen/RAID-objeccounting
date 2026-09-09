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
- 仅一个通过：对A0和该组用 `--seed 43` 重复三步；各输出目录使用 `_s43` 后缀。
  两者完成后，`summarize --runs outputs/structure/A0_s43 outputs/structure/通过组_s43` 单独汇总复验。
- 都未通过：不追加训练。报告失败结果，保留A0。
- 剩余时间不够：维持summary中的pending状态，不将未训练方案判为失败。

恢复使用原目录、原组和原seed：给train命令添加 `--resume 输出目录/latest.pt`。
smoke/profile checkpoint禁止作为正式起点，恢复要求源码和资源一致。
普通评估/预测自动从checkpoint加载 `head_type`，旧checkpoint默认raid/局部损失0。

每组输出best/latest、完整配置与哈希、history、分量损失/路由/数量分组、梯度探针、
best_validation下逐图和分类CSV、重点图与NPZ。空间头路由统计是空值（不适用）。
最终summary含参数量、累计训练时间、推理时间及峰值显存。测试集封存；本地合成测试不能代替GPU验收。

```bash
python -m unittest discover -s tests -v
```
