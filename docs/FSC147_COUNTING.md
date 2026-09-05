# FSC147 文本提示目标计数

## 环境和预训练权重

建议 Linux、Python 3.10、单张 NVIDIA GPU。下面的环境使用 PyTorch 2.3.0 /
torchvision 0.18.0；本机已用此组合完成真实数据验证。

```bash
conda create -n raid-count python=3.10 -y
conda activate raid-count
python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-counting.txt
```

从项目根目录执行命令。数据和权重目录均由参数传入，不写死本机路径。
不要沿用原项目包含大量异常检测组件的 Linux 环境导出文件。
无需安装 mmcv、mmsegmentation、FAISS 或 xFormers。

### 排查 `ModuleNotFoundError: No module named 'clip'`

该错误表示运行训练的 Python 环境缺少 OpenAI CLIP 代码包。
`--clip-weights ./models/ViT-B-16.pt` 只指定权重，不能替代安装代码包。
在启动训练的同一 conda 环境、同一计算节点或作业环境中执行：

```bash
python -c "import sys; print(sys.executable)"
python -m pip install git+https://github.com/openai/CLIP.git
python -c "import clip; print(clip.__file__); print(clip.tokenize(['the apples']).shape); assert callable(clip.load)"
```

验证成功后重跑原训练命令。使用 `python -m pip` 保证安装到该 Python 对应的环境。
应安装上面的 OpenAI 仓库，不要用 `pip install clip` 替代。

如果服务器无法访问 GitHub，可在可联网机器下载 OpenAI CLIP 源码并传到服务器，
然后安装包含 `setup.py` 的目录，例如：

```bash
python -m pip install ./external/CLIP
# 若已上传本项目本地验证时使用的源码，也可使用：
python -m pip install ./runtime/pretrained/CLIP-main
```

以上两个路径按实际上传位置选择一个。本地源码安装仍需满足 PyTorch、torchvision、
ftfy、regex、tqdm 等依赖；完全离线时需预先准备与服务器 Python/系统匹配的依赖 wheel，
再用 `python -m pip install --no-index --find-links /path/to/wheels /path/to/CLIP` 安装。

准备以下三个权重：

1. Talk2DINO：`vitb_mlp_infonce.pth`，从
   [官方 weights](https://github.com/lorebianchi98/Talk2DINO/tree/main/weights) 获取。
   维度必须是 `512 → 768 → 768`，不接受 ViT-L/DINOv3 权重。
2. [DINOv2 ViT-B/14-register 权重](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth)。
3. [CLIP ViT-B/16 权重](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)。

默认 DINOv2/CLIP 可以联网加载，Talk2DINO 投影必须提供本地路径。
离线运行时同时传入 DINOv2 仓库及权重、CLIP 权重路径：

```bash
git clone https://github.com/facebookresearch/dinov2.git external/dinov2
git -C external/dinov2 checkout 85a24602099d397264d5b30461ad7f3bfd726ca1
```

代码固定此 DINOv2 版本，避免上游 main 分支变动影响复现；本地仓库也应使用同一版本。
各预训练模块均冻结，训练期间保持 eval 模式；CLIP 只保留文本部分。

## 数据和任务协议

```text
/datasets/fsc147/
  images_384_VarV2/
  gt_density_map_adaptive_384_VarV2/
  annotation_FSC147_384.json
  Train_Test_Val_FSC_147.json
  ImageClasses_FSC147.txt
/datasets/FSC-147-D.json
```

FSC147 图像、密度和点标注来自
[LearningToCountEverything](https://github.com/cvlab-stonybrook/LearningToCountEverything)。
文本采用 [CounTX 的 FSC-147-D.json](https://github.com/niki-amini-naieni/CounTX/blob/main/FSC-147-D.json)
中的 `text_description` 原文。

只使用官方 `train`（3659）、`val`（1286）、`test`（1190）三个互斥划分。
`val_coco`、`test_coco` 是子集，不与主划分合并。每图示例框不进入模型。
验证和测试的点标注仅用于评价，不进入模型前向或检索过程。

“开放类别”指测试类别不属于 FSC147 计数训练类别；不声明预训练视觉/语言模型
从未见过这些类别。首版不做训练集外部记忆库或验证/测试数据的跨图检索。

## 训练

默认 448×448 整图缩放、32×32 密度图、150 个文本相关候选 patch、3+3 个专家。
图像和密度只做同步水平翻转；不裁剪、无滑窗、无多尺度。

```bash
python run_train_fsc147.py \
  --data-root /datasets/fsc147 \
  --text-annotations /datasets/FSC-147-D.json \
  --projection-weights /models/vitb_mlp_infonce.pth \
  --dino-repo external/dinov2 \
  --dino-weights /models/dinov2_vitb14_reg4_pretrain.pth \
  --clip-weights /models/ViT-B-16.pt \
  --device cuda:0 --output-dir outputs/fsc147_text
```

默认 100 epochs、batch size 2、梯度累积 4、AdamW、学习率/weight decay 均为
`1e-4`、seed 42。可通过 `--config my_config.json` 覆盖默认配置，命令行优先。
JSON 中的相对路径按当前工作目录解析。

需要降低显存时使用 `--batch-size 1 --accumulation 8`。
本机 8GB 显卡已跑通 batch size 1，以及 batch size 2 / 梯度累积 4 的冒烟检查。
首版采用 FP32，不自动调整 batch size 或训练精度。

训练输出：

- `config.json`：完整生效配置。
- `history.jsonl`：每轮训练 loss/MAE、验证 MAE/RMSE。
- `best.pt`：验证 MAE 最优的 checkpoint。
- `latest.pt`：最新 checkpoint。
- `run.log`：带时间和上下文的日志；错误包含堆栈。

checkpoint 包含计数模型参数/缓冲区、优化器、epoch、配置和 Python/NumPy/PyTorch/CUDA
随机状态。冻结的预训练参数不重复存储，因此搬迁 checkpoint 时必须保留相同的三份
预训练权重和 DINOv2 代码版本。只加载可信来源的训练 checkpoint。

恢复到下一个 epoch：

```bash
python run_train_fsc147.py --resume outputs/fsc147_text/latest.pt --epochs 100
```

`--epochs` 是总轮数。精确恢复要求使用相同的数据、batch size、累积步数、workers、
软件和设备；修改这些训练设置将作为新的实验条件。新运行不会覆盖已有 checkpoint，
必须显式恢复或选择新的输出目录。恢复到新目录时，该目录重新建立 best checkpoint。

### 先做冒烟和过拟合检查

将上述路径保存到自己的配置，例如 `configs/server.json`，然后执行：

```bash
python run_train_fsc147.py --config configs/server.json \
  --limit-train 8 --limit-val 8 --no-augment \
  --batch-size 1 --accumulation 1 --workers 0 --epochs 1 \
  --output-dir outputs/smoke

python run_train_fsc147.py --config configs/server.json \
  --limit-train 8 --limit-val 8 --no-augment \
  --batch-size 1 --accumulation 1 --workers 0 --epochs 40 \
  --output-dir outputs/overfit

python tools/validate_counting.py --checkpoint outputs/overfit/latest.pt
```

验收脚本使用真实预训练模型，比较固定 8 张训练图上的初始化与训练后 loss/MAE，
检查提示变化、重复推理一致性、密度恢复积分，并导出目标/替换提示的热力图。
这些结果不用于宣称完整 FSC147 精度；完整训练必须去掉子集限制。

## Slurm / sbatch 提交正式训练

仓库提供 `scripts/train_fsc147.sbatch`，默认申请单节点、1 张 GPU、8 个 CPU、32GB
内存、24 小时。分区、账户及 QoS 按集群要求通过 sbatch 参数指定，不预设集群名称。
配置示例为 `configs/server.example.json`；将其复制为 `configs/server.json` 并替换真实路径。

先在服务器进入 RAID 项目根目录并激活已准备好的 conda 环境：

```bash
cd /absolute/path/to/RAID
conda activate YOUR_ENV
sbatch --partition=YOUR_GPU_PARTITION scripts/train_fsc147.sbatch configs/server.json
```

如集群有 GPU 默认分区，可以省略 `--partition`。如需账户，在脚本文件名前增加
`--account=YOUR_ACCOUNT`；其他资源也可用 sbatch 参数覆盖，例如 `--time=12:00:00`。
脚本使用提交时 conda 环境中的 Python，不依赖计算节点交互式 shell 初始化。
也可以在提交前显式指定 `export RAID_PYTHON=/path/to/conda/envs/YOUR_ENV/bin/python`。

脚本保留 Slurm 分配的 `CUDA_VISIBLE_DEVICES`，以 `cuda:0` 使用分配到的 GPU。
执行前检查 CLIP 导入、CUDA、准备好的本地权重、DINOv2 仓库及完整数据训练配置。
正式训练脚本拒绝 `limit_train/limit_val` 子集配置；冒烟训练仍使用前面的直接命令。

提交成功返回 `Submitted batch job JOB_ID`。查看队列和日志：

```bash
squeue -j JOB_ID
tail -f slurm-raid-fsc147-JOB_ID.out
```

错误输出为 `slurm-raid-fsc147-JOB_ID.err`，训练日志仍位于配置的输出目录中。
需要恢复时，确认先前作业已经结束，再提交：

```bash
sbatch --partition=YOUR_GPU_PARTITION scripts/train_fsc147.sbatch \
  configs/server.json --resume outputs/fsc147_text/latest.pt --epochs 100
```

项目、数据、conda 环境和权重必须已在计算节点可访问的文件系统上；sbatch 只提交脚本，
不会自动上传这些文件。选项及作业行为参见 [Slurm 官方文档](https://slurm.schedmd.com/sbatch.html)。

## 评估与单图推理

```bash
python eval_fsc147.py --checkpoint outputs/fsc147_text/best.pt \
  --split val --output-dir outputs/eval_val

python eval_fsc147.py --checkpoint outputs/fsc147_text/best.pt \
  --split test --output-dir outputs/eval_test

python predict_fsc147.py --checkpoint outputs/fsc147_text/best.pt \
  --image /path/to/image.jpg --text "the apples" \
  --output-dir outputs/apple_prediction
```

评估不继承训练中的 `limit_train/limit_val`，默认覆盖指定完整划分。
仅冒烟时使用 `--limit 8`；对应 metrics JSON 标记 `limited: true`。
评估输出 MAE/RMSE JSON、逐图误差 CSV，以及默认 3 张密度/文本响应叠加图。

单图推理输出 `prediction.json`、原始尺寸的 `*_density.npy` 和 `*_similarity.npy`、
两张 PNG 叠加图。密度图的积分等于预测数量；PNG 为显示对颜色归一化，不能从颜色
直接读出计数。数量保留浮点数，计算 MAE/RMSE 时不四舍五入。

Python 接口：

```python
from src.counting.cli import build_model
from src.counting.inference import CountingPredictor
from src.counting.training import read_checkpoint, restore_checkpoint

path = "outputs/fsc147_text/best.pt"
state = read_checkpoint(path)
config = state["config"]
model = build_model(config)
restore_checkpoint(path, model, restore_rng=False)
predictor = CountingPredictor(model, config["device"], config["image_size"], state["epoch"])
result = predictor.predict("image.jpg", "the apples")
print(result["count"], result["density"].shape)
```

## 模块与计算定义

- `dataset/fsc147.py`：数据验证、ImageNet 图像归一化、密度积分守恒。
- `src/counting/encoders.py`：预训练权重适配与冻结特征提取。
- `src/counting/model.py`：文本 top-k 检索、共享投影、计数专用路由与 MoE。
- `src/counting/training.py`：损失、累积梯度、指标及 checkpoint。
- `src/counting/inference.py`：`predict(image, text)` 和可视化。

输入特征与文本归一化后，计算余弦相似度并取 top-150 patch。
所有 patch 对候选特征的余弦匹配组成 150 通道匹配图；以温度 `0.1` 的 softmax
对候选特征加权重建。原特征/重建特征分别通过同一 768→384 投影，再拼接文本
响应图，得到 769 通道引导特征。

第一层只将样本分派给 top-2 专家，组合权重在这两个专家之间归一化，未选专家不贡献
残差或 bias。第二层第 0～29 轮均匀混合，第 30 轮起学习混合权重。
专家块复用原 RAID 实现；计数路由独立，原异常检测默认行为不改变。
最终混合 logits 经过 Softplus，求和得到数量。

损失为：

```text
MSE(100 × predicted_density, 100 × target_density)
+ 0.1 × mean(abs(predicted_count - target_count) / (target_count + 1))
+ 0.005 × routing_balance_loss
```

密度图重采样采用 area 插值并保持积分，监督目标再校正至点标注数量；预测只使用
自己的积分。第二层专家末端 bias 初始化为 -3，避免 Softplus 初始产生过高总计数。

## 限制

首版是对 RAID 的文本计数适配，不是原论文异常检测系统的严格复现，也没有完成
论文级计数对比或消融。固定 top-k 在目标缺失时仍会选出候选，不能将其视为可靠的
无目标拒识。FSC147 单目标类别监督不足以保证复杂场景中的多类别区分能力。
支持英文单目标描述，不自动翻译、不提供中文或组合关系提示的准确性保证。
