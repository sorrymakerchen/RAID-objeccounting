# Third-party notices

This counting adaptation reuses the existing RAID expert blocks. The original RAID project is
[Mingxiu-Cai/RAID](https://github.com/Mingxiu-Cai/RAID); its original authors retain their rights.

## Talk2DINO

The compatible projection layout in `src/counting/encoders.py` is adapted from the
`ProjectionLayer` implementation in
[lorebianchi98/Talk2DINO](https://github.com/lorebianchi98/Talk2DINO), licensed under Apache-2.0.
The local adaptation keeps only the released ViT-B inference projection and legacy checkpoint
key conversion; it does not include the segmentation framework or projection training code.

Paper: *Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for
Open-Vocabulary Segmentation*, ICCV 2025.

## CLIP

The text-only computation in `src/counting/encoders.py` follows `CLIP.encode_text` from
[openai/CLIP](https://github.com/openai/CLIP), Copyright (c) 2021 OpenAI, MIT License.
The adaptation removes the unused visual encoder and obtains dtype from the text embedding.
The license is reproduced in `third_party/licenses/CLIP.txt`.

## DINOv2 and datasets

DINOv2 is loaded as an external dependency from
[facebookresearch/dinov2](https://github.com/facebookresearch/dinov2), Apache-2.0.
A copy of the Apache-2.0 license is included in `third_party/licenses/Apache-2.0.txt`.

FSC147 and FSC-147-D remain external data sources. Follow their original distribution terms:
[FSC147](https://github.com/cvlab-stonybrook/LearningToCountEverything),
[FSC-147-D / CounTX](https://github.com/niki-amini-naieni/CounTX).
No dataset files or pretrained model weights are included in the source implementation.
