# CUDA Data Residency Lab

同時提供瀏覽器互動模型、可測試的 Python 成本模型，以及真實 CUDA kernel。benchmark 以相同 deterministic input（seed `12345`）、資料大小、迭代數與重複次數比較兩條路徑：`cuda-transfer-per-iteration` 每輪 H2D → kernel → D2H；`cuda-gpu-resident` 只在開始 H2D 一次，warm-up 後在量測迴圈重複 kernel，最後才 D2H 驗證。

## CPU-only 驗證

```bash
python3 -m unittest discover -s tests -v
python3 scripts/validate_benchmark_result.py tests/fixtures/skipped_result.json
python3 src/residency_model.py --payload-mb 1024 --iterations 100 --transfers 2
```

fixture 的 skipped 結果只驗證輸出契約，不是假造 GPU timing。沒有 NVIDIA GPU/CUDA 時，執行 CUDA binary 應視為 unavailable/failed，不會輸出零值或估算結果。

## NVIDIA 實機

需要 NVIDIA Driver、CUDA Toolkit 13.4.3（或相容版本）、CMake 3.31.10；固定參數與工具鏈記錄於 [`benchmark-manifest.json`](benchmark-manifest.json)。本次查證 NVIDIA 官方 registry 可見固定 tag `nvidia/cuda:13.4.3-devel-ubuntu22.04`（2026-11-24）；本 sandbox 無法 pull 或 GPU build，因此未驗證實際下載/編譯。共用輸出契約為 [`benchmark-results.schema.json`](benchmark-results.schema.json)，規則見 [`BENCHMARK_STANDARD.md`](BENCHMARK_STANDARD.md)。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/vector_add 16777216 20 5 1 > result.json
python3 scripts/validate_benchmark_result.py result.json
```

CLI 參數依序為 `elements iterations repetitions warmup_iterations`。每筆 JSON 會記錄 Git rev/dirty、OS/arch/Python、CUDA runtime、CUDA Driver API 的最高支援版本（`cudaDriverGetVersion`）、`nvidia-smi` 讀到的實際 NVIDIA driver（`cuda_driver`，無法取得時為 `null`）、GPU model/compute capability/memory、nvcc、所有參數、raw samples 與 median/mean/stdev/min/max。每筆 measurement 另記 `iterations_per_sample`。measurement 以 seconds 為單位：`baseline.transfer-inclusive` 每個 sample 是所有 iterations 的同步 wall-clock，且每 iteration 含 2 次 H2D、kernel、1 次 D2H；`resident.total-inclusive` 每個 sample 含恰好一次 2-input H2D、所有 kernel iterations、一次 D2H，與 baseline transfer-inclusive 可比；`resident.kernel-loop-wall` 只量 resident kernel loop，loop 本身不做 H2D/D2H。warmup 會先同步，且不計入 samples；分母是每個 sample 的 `iterations_per_sample`。baseline 的最後 D2H 會在 resident 覆蓋 device buffer 前驗證，resident 最後結果也會驗證。

或使用固定 CUDA 13.4.3 / CMake 3.31.10 容器：

```bash
docker build -t nvidia-lab/cuda .
docker run --rm --gpus all nvidia-lab/cuda 16777216 20 5 1 > result.json
```

本 sandbox 檢查時沒有 `nvcc`、CMake 或 NVIDIA GPU，因此未宣稱或產生 GPU 結果；有 GPU 的環境才可編譯與量測。未修改 `.github/workflows/*`。
