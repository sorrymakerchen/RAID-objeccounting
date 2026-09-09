# FSC147 密度监督与尺度实验

只使用训练集和验证集。正式基线是 epoch=33 的 best.pt，验证 MAE 31.8462676262、
RMSE 96.8439342990。训练组从相同随机初始化开始，不加载该 best.pt。
所有命令在项目根目录执行，配置使用服务器现有本地资源路径。无新增依赖。

提交前执行 `mkdir -p log`。Slurm 在执行脚本前打开日志文件，因此目录必须提前存在。
三个 sbatch 脚本的标准输出/错误均保存到 `log/slurm-作业名-作业ID.out/.err`。

## 固定权重实验

```bash
sbatch --partition=YOUR_GPU_PARTITION scripts/ablate_fsc147.sbatch intervene \
  --config configs/server.json --checkpoint outputs/fsc147_text/best.pt \
  --reference-csv outputs/eval_val/val_predictions.csv \
  --smoke --output-dir outputs/ablations/D_smoke
```

冒烟保持指定图在完整验证集中的原 batch 同伴、位置及 batch size=2；只对三张指定图记分。
确认 summary.json 状态为 `smoke_passed_not_full_validation`，再提交同一命令，
删除 `--smoke`，输出改为 `outputs/ablations/D_full`。
默认依次执行 D0–D4；`--experiment D2` 等可只运行 D0 和指定变体。
D0 逐图或聚合误差不符合 1e-3 容差则退出 2，保留差异，不继续干预。

| 模式 | 行为 |
|---|---|
| D0 | 448、原始检索与路由 |
| D1 | 896、K=150，原生 64×64 密度积分，不除以4 |
| D2 | 32×32 网格分5×5区，每区相似度最高6个，合并按分数降序/索引升序 |
| D3 | 第一层实际权重固定为0、0.5、0.5 |
| D4 | 第二层实际权重固定为1/3 |

所有模式冻结参数和 BN 统计。路由 probability 表示原路由输出，weights 表示实际干预后的贡献。
最终 logits 和第二层3个专家 logits 另存 NPZ。Softplus 非线性使专家独立数量不可直接相加。
D1 同时改变候选面积比例；数量普遍增加不构成效果改善或因果证据。

## 三组训练：先冒烟和计时

对 T0、T1、T2 分别提交以下命令（替换 GROUP）：

```bash
sbatch --partition=YOUR_GPU_PARTITION scripts/ablate_fsc147.sbatch train \
  --group GROUP --config configs/server.json \
  --reference-csv outputs/eval_val/val_predictions.csv \
  --smoke --output-dir outputs/ablations/GROUP_smoke

sbatch --partition=YOUR_GPU_PARTITION scripts/ablate_fsc147.sbatch profile \
  --group GROUP --config configs/server.json \
  --reference-csv outputs/eval_val/val_predictions.csv \
  --output-dir outputs/ablations/GROUP_profile
```

训练冒烟包含8张训练图的一次梯度累积更新、4张验证图和固定训练探针。
profile 执行一次完整训练 epoch、完整验证和探针，输出 `recommended_time_minutes`：
实测时间×100×1.3，向上取整到分钟。它是初始估计，后续学习路由及硬件负载可能改变耗时。
冒烟/profile checkpoint 不允许作为正式实验的恢复起点。OOM 明确失败，不自动减 batch、改精度。

## 正式训练

把 MINUTES 换为该组 profile 的 recommended_time_minutes，分区和账户由集群实际值指定：

```bash
sbatch --partition=YOUR_GPU_PARTITION --time=MINUTES scripts/ablate_fsc147.sbatch train \
  --group GROUP --config configs/server.json \
  --reference-csv outputs/eval_val/val_predictions.csv \
  --output-dir outputs/ablations/GROUP
```

脚本默认单GPU、8CPU、32GB主存、2小时；正式训练必须按计时覆盖 `--time`。
使用 `/public/home/yzhang712/anaconda3` 中的 `raid` 环境；可通过 RAID_CONDA_ROOT、RAID_CONDA_ENV 指定。
需要账户时在脚本路径前加 `--account=YOUR_ACCOUNT`。不要覆盖 CUDA_VISIBLE_DEVICES。

| 组 | 图像 | 密度监督 | 数量项（系数0.1） |
|---|---|---|---|
| T0 | 448 | 32×32 MSE，密度乘100 | mean(abs误差/(GT+1)) |
| T1 | 448 | 同T0 | mean(abs误差) |
| T2 | 896 | 64×64预测经2×2求和池化到32×32；GT直接从官方密度生成32×32 | 同T0 |

统一 seed42、100 epochs、batch2、累积4、AdamW lr/weight_decay均1e-4、冻结编码器、水平翻转、
K150、前30 epochs第二层均匀路由。训练入口强制这些定义并写入 config.json。
数据顺序/翻转由独立 DataLoader generator（seed+epoch）控制，workers必须≥1；
与历史训练的随机消费路径可能不同，因此使用新的T0作为直接对照，不要求T0复现历史最佳精度。
三组计数模块初始state哈希必须一致；预训练资源及源码哈希也参与汇总一致性检查。

每轮保存损失分量、训练数量分组误差、专家平均权重和实际使用率、完整验证指标。
探针按训练图文件名排序，在每个非空数量区间取前2张，无增强、eval模式；输出对混合logits的
密度/数量梯度L2范数，以及乘0.1后的数量梯度。梯度对应batch平均损失，不更新参数和BN。
验证MAE严格更小时更新best.pt，因此并列保留较早epoch。正式训练结束自动评估最佳权重。

断点恢复使用相同输出目录，加 `--resume outputs/ablations/GROUP/latest.pt`。
只能恢复同组正式实验；数据、代码、权重和实验定义必须一致。普通入口新增
`--count-loss-mode relative|absolute`、`--density-supervision-size 32`，默认保持基线行为。

## 汇总与交付

三组完成后运行（在激活的原环境中，无GPU需求）：

```bash
python tools/ablate_fsc147.py summarize --config configs/server.json \
  --runs outputs/ablations/T0 outputs/ablations/T1 outputs/ablations/T2 \
  --output-dir outputs/ablations/training_comparison
```

各实验目录包含 predictions.csv、by_category.csv、by_count.csv、routes.csv、summary.json。
训练最佳结果位于 best_validation/；逐轮信息为 history.jsonl 和 probe_epoch_*.csv。
samples/ 保存23张重点图的密度、相似度、候选、实际路由和原始数组；干预及最终训练汇总图
按同一图所有变体共享颜色范围，候选坐标按各自32/64网格映射原图。

汇总要求完整T0/T1/T2，重算CSV指标并验证资源/初始化一致。筛选全部满足：
MAE≤0.99×T0、RMSE≤1.01×T0、≥200子集MAE≤0.95×T0、0–24子集MAE≤T0+0.5。
符合者按MAE排序，否则保留T0。summary.json保存每条条件，report.md解释结论边界。
本轮不组合T1/T2、不追加路由训练，不使用测试集；单种子结果仅筛选下一轮复验候选。

提交分析时提供 D_full/、三组 config/provenance/history/summary/probe/best_validation/samples、
training_comparison/ 及 Slurm 日志。无需上传预训练权重和数据集。

## 本地验收

```bash
python -m unittest discover -s tests -v
```

测试使用合成依赖验证行为，不能代替服务器正式模型的D0复现、GPU冒烟或100 epochs实验。
