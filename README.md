# NeuViT: Spiking the Transformer with Single-Step Attention for Efficient High-Resolution UAV Vision

![1-s2 0-S2590123025049977-gr4_lrg](https://github.com/user-attachments/assets/de3c8f93-56d4-47a4-9c68-cf0828196118)

This repository contains the official PyTorch implementation of:

### Spiking the Transformer: NeuViT with Single-Step Attention for Efficient High-Resolution UAV Vision

**Jay Shen Teoh**, Chee Keong Tan, Vishnu Monn Baskaran, Wai Peng Wong  
*Results in Engineering*, Volume 29, 2026  
**[[Paper](https://www.sciencedirect.com/science/article/pii/S2590123025049977)] [[DOI](https://doi.org/10.1016/j.rineng.2025.108955)]**

## Highlights

NeuViT is a spiking vision transformer designed for real-time, high-resolution object detection on power-constrained UAV platforms. Key results on VisDrone2019 at **1500×2000** resolution:

| Model | AP | Energy (J) | Power (W) | FPS | UAV Feasible |
|---|---:|---:|---:|---:|:---:|
| Swin-T | 27.0 | 1.61 | 24.5 | 15.2 | ✗ |
| SparseViT | 26.7 | 1.32 | 23.3 | 17.6 | ✗ |
| EfficientNet-B0 | 26.0 | 0.18 | 9.25 | 51.4 | ✓ |
| SEW-ResNet18 | 22.7 | 0.17 | 4.21 | 24.8 | ✓ |
| **NeuViT (Ours)** | **21.4** | **0.18** | **3.98** | **22.1** | **✓** |

NeuViT achieves **88.8% lower energy** and **31% lower latency** than Swin Transformer while operating within the **10 W / 15 FPS** UAV deployment envelope.

## Code

🚧 We’re currently experiencing server issues. The code will be released once the server is back online.
