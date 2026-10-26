# CUDA Data Residency Lab

同時提供瀏覽器互動模型、可測試的 Python 成本模型，以及真實 CUDA kernel。用途是量化 H2D／D2H、批次大小、PCIe 頻寬與資料常駐策略。

## 快速驗證

```bash
python -m unittest discover -s tests -v
python src/residency_model.py --payload-mb 1024 --iterations 100 --transfers 2
```

## NVIDIA 實機

需要 NVIDIA Driver、CUDA Toolkit、CMake 3.24+：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/vector_add 16777216 20
```

或使用固定 CUDA 13.4.3 的容器：

```bash
docker build -t nvidia-lab/cuda .
docker run --rm --gpus all nvidia-lab/cuda 16777216 20
```

輸出是可保存的 JSON。自架 GitHub Actions Runner 可手動執行 `gpu.yml`；一般 push／PR 由 `ci.yml` 驗證，`pages.yml` 部署互動網站。

## 驗收重點

- 比較逐輪傳輸與 device-resident pipeline。
- 用 Nsight Systems 驗證 memcpy 與 kernel 是否重疊。
- 固定資料量、warm-up、GPU clocks 和重複次數後才比較結果。

