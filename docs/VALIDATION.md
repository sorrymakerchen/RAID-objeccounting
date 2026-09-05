# 实施验证记录

日期：2026-09-05。验证设备为 Windows / RTX 4060 Ti 8GB；Python 3.8.20、
PyTorch 2.3.0、torchvision 0.18.0。Linux 服务器正式环境建议 Python 3.10。
没有连接 Linux 服务器或进行完整 100 轮 FSC147 训练。

## 自动化行为测试

```bash
python -m unittest discover -s tests -v
```

**17 项全部通过**。先建立测试并确认缺少实现时失败，再完成实现。
覆盖：

- 密度降采样、恢复原图尺寸时积分守恒，水平翻转同步。
- 划分重叠、缺失文本/密度、非法密度和验证增强检查。
- 文本变化影响相似度和候选检索；top-2 恰好有两个有效贡献者。
- 输出非负、计数与积分相等、计数模块有梯度且编码器冻结。
- 第二级 MoE 在 warmup 前后的路由梯度行为。
- checkpoint 恢复预测、优化器和 RNG；恢复后的下一轮与不中断训练参数逐项完全一致。
- 不完整梯度累积组和最后小 batch 与等价整批更新一致。
- MAE/RMSE 按每张图误差计算；非有限损失明确报错。
- Talk2DINO 旧权重键兼容及维度错误；CLIP 文本编码不依赖视觉分支。

另使用真实 CLIP 权重验证文本专用适配器：3 条英文文本的 512 维结果与官方
`encode_text` **逐项完全一致**。

## 真实预训练模型和 FSC147 数据

使用完整 448×448 输入、768 维 DINOv2 特征、150 通道匹配及默认 384 通道适配层。
未用模拟特征替代实际模型。DINOv2 固定版本为
`85a24602099d397264d5b30461ad7f3bfd726ca1`，配套 register 版本权重。

已检查 FSC147 train/val/test 的 3659/1286/1190 张图像、密度图、点标注和
FSC-147-D 文本索引。三份预训练权重成功加载；CLIP 权重的官方 SHA-256 校验通过。

固定前 8 张训练图、关闭增强，以 batch size 1 / 累积 1 训练 40 轮，再从 latest
checkpoint 恢复训练 1 轮。覆盖第二级 MoE 从均匀混合切换至学习路由的路径。

相同 8 张图、eval 模式的初始化模型与恢复后模型比较：

| 指标 | 初始化 | 训练后 |
|---|---:|---:|
| 总损失 | 36.3370 | 4.7307 |
| 计数 MAE | 21.2601 | 3.3898 |

训练模式日志的第一轮/第 40 轮平均 loss 为 33.5234 / 3.0178。
上表使用独立 eval 检查，因路由噪声与 BatchNorm 状态不同，不应与训练模式数值混用。

证据：`outputs/overfit/history.jsonl`、`outputs/acceptance/acceptance.json`。
复验脚本：`tools/validate_counting.py`。

## 命令与可视化验收

- 训练、验证、测试、单图推理三个入口均用真实数据跑通。
- 单独验证 batch size 2 / 累积 4 / 5 张训练图，最后小 batch 和不完整累积组正常更新。
- 使用最佳验证 checkpoint 在前 8 张 val/test 图上验证 JSON、CSV 和 PNG 导出。
- 重复模型推理的密度图逐项完全一致。
- 原图尺寸密度积分检查通过：一次单图预测为 23.414646，恢复密度积分为 23.414644。
- 已查看生成的密度叠加图，原图尺寸和显示正常。

小样本最佳验证 checkpoint 的导出指标，仅用于流程检查：

| 划分 | 图像数量 | MAE | RMSE |
|---|---:|---:|---:|
| val 子集 | 8 | 29.5367 | 37.0477 |
| test 子集 | 8 | 18.8568 | 31.8407 |

输出位于 `outputs/eval_val_smoke/`、`outputs/eval_test_smoke/`、
`outputs/predict_smoke/`；metrics JSON 显式标记 `limited: true`。
这些不是完整 FSC147 基准结果，不能与正式论文指标直接比较。

## 已知行为与边界

同一张训练图的提示由 `the peppers` 改为 `the elephants` 后，相似度图平均绝对变化
为 0.09008，候选集合变化，预测数量从 16.43 变为 19.53。提示路径生效，但不存在目标
的计数没有归零；这印证首版不具备可靠无目标拒识。对应图片和数据保存在
`outputs/acceptance/target_*` 与 `outputs/acceptance/alternate_*`。

原 RAID 的 `moe1.py`、`expert.py`、`backbones.py` 和两份异常检测训练入口，与
`RAID-official` 参考副本的 SHA-256 相同。未修改原异常检测行为；未重新运行其完整
异常检测数据集实验。

本机补齐的测试依赖/预训练权重保存在忽略目录 `runtime/`，没有安装到已有 Python
环境。Linux 使用正式依赖文件和运行文档部署即可。
