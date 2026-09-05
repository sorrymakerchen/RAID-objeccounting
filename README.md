# RAID text-conditioned counting

本目录在 RAID 异常检测代码之外新增 FSC147 文本提示计数分支。

输入一张图像和英文目标描述，输出非负密度图及数量。模型使用冻结的 DINOv2
ViT-B/14-register、CLIP 文本编码器和 Talk2DINO 投影，以文本选择图内候选特征，
再由 RAID 两级 MoE 回归密度。

- [安装、数据准备、训练、评估、推理](docs/FSC147_COUNTING.md)
- [默认配置](configs/fsc147_text.json)
- [本次验证记录](docs/VALIDATION.md)
- [第三方来源与许可](THIRD_PARTY_NOTICES.md)

```bash
python -m unittest discover -s tests -v
python run_train_fsc147.py --help
python eval_fsc147.py --help
python predict_fsc147.py --help
```

原有 `run_train_mvtec.py`、`run_train_visa.py` 和异常检测模块保留。
计数分支不依赖 FAISS、异常合成数据或类别正常样本库。
