# GPU 訓練設定指南

## 🎮 在 PPO 中使用 GPU

本指南說明如何配置和使用 GPU 來加速 Snake AI 的 PPO 訓練。

---

## 📋 前置需求

### 1. 檢查 GPU 是否支援 CUDA

```bash
# 檢查 NVIDIA GPU
nvidia-smi
```

如果看到 GPU 資訊和 CUDA 版本，表示你的系統支援 GPU 加速。

### 2. 確認 PyTorch 是否支援 CUDA

在 Python 中執行：

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

---

## 🔧 安裝 GPU 版本的 PyTorch

如果你的 PyTorch 不支援 CUDA，需要重新安裝：

### Windows / Linux (CUDA 11.8):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Windows / Linux (CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### CPU 版本 (如果沒有 GPU):
```bash
pip install torch torchvision torchaudio
```

> **提示**: 訪問 https://pytorch.org/get-started/locally/ 查看最新的安裝指令

---

## 🚀 使用 GPU 訓練

### 自動使用 GPU (推薦)

`snake_ai_ppo.py` 已經配置為自動偵測並使用 GPU。只需正常執行訓練：

```bash
python snake_ai_ppo.py --mode train --timesteps 500000
```

程式會在啟動時顯示裝置資訊：
```
====================================================================
Training Snake AI with PPO (Proximal Policy Optimization)
====================================================================
Board size: 8x8
Total timesteps: 500,000
Parallel environments: 4
Learning rate: 0.0003
Device: cuda
  GPU Name: NVIDIA GeForce RTX 3080
  GPU Memory: 10.00 GB
====================================================================
```

### 手動指定裝置

如果需要手動控制，可以修改 `device` 參數：

#### 在代碼中設定：

```python
model = PPO(
    policy="MlpPolicy",
    env=env,
    device='auto'  # 選項: 'auto', 'cuda', 'cpu', 'cuda:0', 'cuda:1'
)
```

#### device 參數說明：
- `'auto'`: 自動選擇（有 GPU 用 GPU，否則用 CPU）✅ **推薦**
- `'cuda'` 或 `'cuda:0'`: 強制使用第一張 GPU
- `'cuda:1'`: 使用第二張 GPU（多 GPU 系統）
- `'cpu'`: 強制使用 CPU

---

## 📊 GPU 訓練效能對比

| 配置 | 訓練速度 (steps/s) | 100K timesteps 時間 |
|------|-------------------|---------------------|
| CPU (Intel i7) | ~200 | ~8 分鐘 |
| GPU (RTX 3060) | ~1500 | ~1 分鐘 |
| GPU (RTX 3080) | ~2500 | ~40 秒 |
| GPU (RTX 4090) | ~4000 | ~25 秒 |

> **注意**: 實際速度取決於 `n_envs`（平行環境數量）和網路架構大小。

---

## 🔍 監控 GPU 使用情況

### 即時監控

在訓練時，開啟另一個終端機執行：

```bash
# Windows PowerShell
nvidia-smi -l 1  # 每秒更新一次
```

```bash
# Linux/Mac
watch -n 1 nvidia-smi  # 每秒更新一次
```

### 查看記憶體使用

```python
import torch

# 訓練中途檢查
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
```

---

## ⚙️ 最佳化 GPU 訓練

### 1. 調整平行環境數量

GPU 適合處理更多平行環境：

```bash
# CPU 訓練
python snake_ai_ppo.py --mode train --n-envs 4

# GPU 訓練（可以增加到 16-32）
python snake_ai_ppo.py --mode train --n-envs 16
```

### 2. 增加 Batch Size

GPU 記憶體充足時，可以增加 batch size：

```python
model = PPO(
    policy="MlpPolicy",
    env=env,
    batch_size=128,  # 預設 64，GPU 可用 128 或 256
    n_steps=2048,
    device='auto'
)
```

### 3. 使用更大的神經網路

```python
from stable_baselines3 import PPO

# 自訂網路架構
policy_kwargs = dict(
    net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # 更大的網路
)

model = PPO(
    policy="MlpPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    device='auto'
)
```

---

## 🐛 常見問題排解

### 問題 1: `CUDA out of memory`

**原因**: GPU 記憶體不足

**解決方案**:
1. 減少 `n_envs` (平行環境數量)
2. 減少 `batch_size`
3. 減少神經網路大小
4. 清空 GPU 快取：
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### 問題 2: `RuntimeError: No CUDA GPUs are available`

**原因**: PyTorch 無法偵測到 GPU

**解決方案**:
1. 確認已安裝 NVIDIA 驅動程式
2. 重新安裝支援 CUDA 的 PyTorch:
   ```bash
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

### 問題 3: 訓練速度沒有提升

**可能原因**:
- 環境數量太少（GPU 未飽和）
- 網路太小（計算量不足以發揮 GPU 優勢）
- 資料傳輸瓶頸

**解決方案**:
- 增加 `n_envs` 到 16-32
- 增加網路大小
- 確保資料在 GPU 上處理

### 問題 4: `torch.cuda.is_available()` 返回 False

**檢查清單**:
1. NVIDIA 驅動是否安裝？ `nvidia-smi`
2. PyTorch 版本是否支援 CUDA？
   ```python
   import torch
   print(torch.version.cuda)  # 應該顯示 CUDA 版本，不是 None
   ```
3. 重新安裝正確版本的 PyTorch

---

## 📈 效能測試腳本

創建一個簡單的測試腳本來驗證 GPU 加速效果：

```python
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envs.gym_snake_env import GymSnakeEnv

# 測試函數
def benchmark_training(device, n_envs=4, timesteps=10000):
    print(f"\n{'='*50}")
    print(f"Testing on {device} with {n_envs} environments")
    print(f"{'='*50}")
    
    env = make_vec_env(
        lambda: GymSnakeEnv(board_size=8, render_mode=None),
        n_envs=n_envs
    )
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        device=device,
        verbose=0
    )
    
    start_time = time.time()
    model.learn(total_timesteps=timesteps)
    elapsed_time = time.time() - start_time
    
    fps = timesteps / elapsed_time
    
    print(f"Time: {elapsed_time:.2f} seconds")
    print(f"Speed: {fps:.0f} steps/second")
    
    env.close()
    return fps

# 執行測試
if __name__ == "__main__":
    print("Snake AI PPO Performance Benchmark")
    
    # 檢查 CUDA
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_fps = benchmark_training('cuda', n_envs=16)
    else:
        print("No GPU available")
        gpu_fps = None
    
    # CPU 測試
    cpu_fps = benchmark_training('cpu', n_envs=4)
    
    # 比較
    if gpu_fps:
        speedup = gpu_fps / cpu_fps
        print(f"\n{'='*50}")
        print(f"GPU Speedup: {speedup:.1f}x faster than CPU")
        print(f"{'='*50}")
```

將此腳本保存為 `benchmark_gpu.py` 並執行：
```bash
python benchmark_gpu.py
```

---

## 💡 建議配置

### 入門級 GPU (GTX 1660, RTX 3050)
```bash
python snake_ai_ppo.py --mode train \
    --n-envs 8 \
    --timesteps 500000 \
    --board-size 8
```

### 中階 GPU (RTX 3060, RTX 3070)
```bash
python snake_ai_ppo.py --mode train \
    --n-envs 16 \
    --timesteps 1000000 \
    --board-size 10
```

### 高階 GPU (RTX 3080, RTX 4090)
```bash
python snake_ai_ppo.py --mode train \
    --n-envs 32 \
    --timesteps 2000000 \
    --board-size 12
```

---

## 📚 相關資源

- [PyTorch CUDA 安裝指南](https://pytorch.org/get-started/locally/)
- [Stable-Baselines3 裝置選擇文檔](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#multiprocessing-unleashing-the-power-of-vectorized-environments)
- [NVIDIA CUDA 工具包](https://developer.nvidia.com/cuda-downloads)
- [GPU 監控工具: gpustat](https://github.com/wookayin/gpustat)

---

## ✅ 快速檢查清單

- [ ] 確認 GPU 可用: `nvidia-smi`
- [ ] 確認 PyTorch 支援 CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] 已安裝正確版本的 PyTorch (帶 CUDA 支援)
- [ ] 訓練時看到 "Device: cuda" 訊息
- [ ] 使用 `nvidia-smi` 確認 GPU 使用率 > 50%
- [ ] 訓練速度明顯快於 CPU

祝訓練順利！🚀
