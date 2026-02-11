# NeuViT: Spiking the Transformer with Single-Step Attention for Efficient High-Resolution UAV Vision

![1-s2 0-S2590123025049977-gr4_lrg](https://github.com/user-attachments/assets/de3c8f93-56d4-47a4-9c68-cf0828196118)

This repository contains the official PyTorch implementation of:

### Spiking the Transformer: NeuViT with Single-Step Attention for Efficient High-Resolution UAV Vision

**Jay Shen Teoh**, Chee Keong Tan, Vishnu Monn Baskaran, Wai Peng Wong  
*Results in Engineering*, Volume 29, 2026  
**[[Paper](https://pdf.sciencedirectassets.com/320278/1-s2.0-S2590123025X00050/1-s2.0-S2590123025049977/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIHkschH9tSX92T8Sf4GqGk0Gyqkc4Q%2FLK201upieoUSvAiB5vHKZaVwT7k9jqT3FxmPmmuVEmtc15iPnNtVnXBfYeCq8BQiz%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMtyuWWHEcxn%2BBHqTjKpAFO0U26bJG8cFqTh5a9xYpwPtK%2F3fe8Tqzu9Ufkg1PwUfqu4xx%2Bq4FBshQ1FtxuexGoopb6y%2Bz%2F7FTtR3QSyS%2F96AVJdMqJh4AejnncXumyjGM68GmEb3s7tTdJwwqv%2FnQTwEDc7srrIO%2BIM3M6npFRKDzIPKvVoIsgfrJ%2FMfHjlGYdLQfjV8%2FImgzdJtvydwl5EhisZ%2BVxHoMcJticdNbqHtV8YV47scqWTWG22cOvvs6%2FfBtYCpOKYNK8Sh2pNvSLrjESQSGRitFmN9NsUrQQb9twlk34O54LsvZL%2BwgWN702lKlw9uVAu2i6ozKkpy7S47Q4jX3bVXrRp2%2F97TxQFTtXd0%2BNSAwJYkc9RFaIaYs%2BOsR9NZphRTwkbyPTOQNh0563o58zOzO1lwaaivA567SD0hVYfk65a7vwpEN6sDVU1N%2FdZMYdrix5m1f1ayRM73llmfJjRvHfy%2BHBVV%2Fl4RuQ1%2ByvpJ%2FNL9Zl%2B9M2fUM15MMWQ%2BbyOHtEos98%2B7CkWY%2BKJ89hycwfi%2FC5DwL%2FfW5FJ5qANKrrgfDZLoPY9JtKiWXky8m34IuRoN4RbNmyHubDnnE8yT5QcKJqwvKxMFQNFaPi4VqWhR5PH58hfNiSs7M6RcBGu14iMtheoinqh6G9exffE64nf0ag2s0yBlRERPvgQOXIFohxv5eFoL0gqerqTNFEYsPDEocf8SIVBFbiBUWUYbEc21fxlKXFsLN9aonLddRbUUsTdMoD9iEzJrUeIuJhSgCLsIYKiJhnTT9XyxHGcFDiEM0puuGaJHnfVjIAOzSZFjsYXd6Dn4%2Bu%2FgtSyDsfEyi1wEVVkkhAU9kFCgnSLg7PrBsPaKqaPMFc2%2FO5eQlJ1uBH9sVUO8wsbSvzAY6sgF9ujEgfUr27B%2FLyg5AkWEX%2FJP9V2E2gDQBfeukJXjV9J8AxE1MM8frgzuob7mxvKurYmTKAGUI%2FDMdSoWDyvy96TJCqknbSLwdhpxczBSGIGSFd41YH0kVNWydq7u6scZ%2F586znKzPpr1fA0nv8j9FW%2BJom49X50xZuRNX3eB3q81pHvkjzunkR6%2F2Z1Qp3ihhJjcaB8bkjT8qt01qzo9metR2dicB1VrrbJp7L3vvgx%2BM&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260211T013944Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYVD3XN4OW%2F20260211%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=719101b68a5bfb3c8b4c1eef53e0c9cb8684c197650b88c22e3df7b52b5cce86&hash=1e98c2c6df70d647a48990e6d309b776e304538c6d1d0ceca1e6154149b8095b&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S2590123025049977&tid=spdf-b25b69bc-82bc-4d73-9f6e-a450eda39f8a&sid=c978d092384411477d1bf934d81075b369b5gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=171f570607535c5c550306&rr=9cc0225cdcfead7a&cc=my)] [[DOI](https://doi.org/10.1016/j.rineng.2025.108955)]**

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
