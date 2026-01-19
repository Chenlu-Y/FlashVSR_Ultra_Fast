# ComfyUI-FlashVSR_Ultra_Fast
在低显存环境下运行 FlashVSR，同时保持无伪影高质量输出。  
**[[📃English](./README.md)]**

## 更新日志
#### 2025-10-31
- **新增:** 独立的 `infer_video.py` 脚本，无需 ComfyUI 即可处理视频
- **新增:** 多GPU并行处理 (`--multi_gpu`) - 自动将视频按帧分割到多个GPU
- **新增:** 自适应tile批处理 (`--adaptive_batch_size`) - 根据GPU显存动态调整tile并发数
- **新增:** GPU显存监控和优化，充分利用24-32GB显卡
- **新增:** 总耗时统计功能，便于性能监控
- **改进:** 视频读取增加OpenCV兜底，提升编解码器兼容性
- **修复:** 空帧处理和负维度错误

#### 2025-10-24
- 新增长视频管道, 可显著降低长视频放大的显存用量  

#### 2025-10-22
- 使用`Sparse_SageAttention`替换了`Block-Sparse-Attention`, 无需编译安装任何自定义内核, 开箱即用.  
- 支持在 RTX50 系列显卡上运行.

#### 2025-10-21
- 项目首次发布, 引入了`tile_dit`等功能, 大幅度降低显存需求  

## 预览
![](./img/preview.jpg)

## 使用说明

### 独立推理脚本（新增！）

使用 `infer_video.py` 可在不使用 ComfyUI 的情况下直接处理视频：

#### Docker容器内运行

```bash
# 方法一：进入容器后运行
docker exec -it flashvsr_ultra_fast bash
cd /app/FlashVSR_Ultra_Fast
python infer_video.py \
  --input /app/input/video.mp4 \
  --output /app/output/output.mp4 \
  --mode tiny \
  --scale 4 \
  --tiled_dit True \
  --tile_size 256 \
  --tile_overlap 24 \
  --model_dir /app/models

# 方法二：从宿主机直接运行（推荐使用 -w 参数指定工作目录）
docker exec -w /app/FlashVSR_Ultra_Fast flashvsr_ultra_fast python /app/FlashVSR_Ultra_Fast/infer_video.py \
  --input /app/input/video.mp4 \
  --output /app/output/output.mp4 \
  --mode tiny \
  --scale 4 \
  --tiled_dit True \
  --model_dir /app/models
```

#### 完整参数说明

**必需参数：**
- `--input`: 输入视频路径
- `--output`: 输出视频路径

**超分相关参数：**
- `--scale`: 超分倍数，可选值：`2`, `3`, `4`（推荐使用 `4`）
  - `2`: 2倍放大，显存占用最低
  - `3`: 3倍放大，平衡质量和显存占用（**完全支持，不限于2的倍数**）
  - `4`: 4倍放大，效果最好（推荐）
- `--mode`: 运行模式
  - `tiny`: 快速模式（默认，显存占用较低，推荐）
  - `tiny-long`: 长视频模式（显存占用最低，适合超长视频）
  - `full`: 高质量模式（显存占用较高，质量最好）

**显存优化参数：**
- `--tiled_dit`: 启用DiT分块计算（`True`/`False`，默认`False`），显存不足时强烈推荐启用
- `--tile_size`: 分块大小（默认`256`），显存不足时可减小到`128`
- `--tile_overlap`: 分块重叠大小（默认`24`），建议为`tile_size`的10-15%
- `--tiled_vae`: 启用VAE分块解码（`True`/`False`，默认`True`）
- `--unload_dit`: 解码前卸载DiT模型（`True`/`False`，默认`False`），显存非常紧张时使用

**多GPU并行处理参数 ⚡：**
- `--multi_gpu`: 启用多GPU并行处理（无需参数值，直接添加即可）
  - 自动将视频按帧分割到多个GPU并行处理
  - 需要2个或以上GPU
  - 接近线性的加速比（2个GPU约2倍速度）
  - 适用于长视频（>500帧）
- `--adaptive_batch_size`: 启用自适应批处理大小（无需参数值，直接添加即可）
  - 根据GPU显存动态调整同时处理的tile数量
  - 需要启用 `--tiled_dit`
  - 大显存GPU（24GB+）效果最明显

**设备与精度参数：**
- `--device`: 指定使用的GPU设备（默认`cuda:0`），启用`--multi_gpu`时会被忽略
- `--precision`: 计算精度（默认`bf16`），可选`fp16`、`bf16`、`fp32`

**高级参数（质量调优）：**
- `--color_fix`: 颜色修正（`True`/`False`，默认`True`）
- `--sparse_ratio`: 稀疏比率（默认`2.0`，范围`1.5-2.0`），`2.0`更稳定
- `--kv_ratio`: KV缓存比率（默认`3.0`，范围`1.0-3.0`），`3.0`质量更高
- `--local_range`: 局部范围（默认`11`，可选`9`或`11`），`11`更稳定
- `--attention_mode`: 注意力模式（默认`sparse_sage_attention`）

**其他参数：**
- `--model_dir`: 模型目录路径（默认：`/app/models`）
- `--seed`: 随机种子（默认`0`）

#### 常用命令示例

**基础4倍超分（显存充足）：**
```bash
python infer_video.py \
  --input /app/input/video.mp4 \
  --output /app/output/output_4x.mp4 \
  --mode tiny \
  --scale 4 \
  --tiled_dit True
```

**3倍放大（平衡质量和显存）：**
```bash
python infer_video.py \
  --input /app/input/video.mp4 \
  --output /app/output/output_3x.mp4 \
  --mode tiny \
  --scale 3 \
  --tiled_dit True \
  --tile_size 256
```

**低显存模式：**
```bash
python infer_video.py \
  --input /app/input/video.mp4 \
  --output /app/output/output_4x.mp4 \
  --mode tiny-long \
  --scale 4 \
  --tiled_dit True \
  --tile_size 128 \
  --tile_overlap 16 \
  --unload_dit True
```

**多GPU加速（2个以上GPU）：**
```bash
python infer_video.py \
  --input /app/input/video.mp4 \
  --output /app/output/output_4x.mp4 \
  --mode tiny \
  --scale 4 \
  --tiled_dit True \
  --multi_gpu \
  --adaptive_batch_size
```

**根据显存大小选择配置：**
- **显存 < 12GB**: `--mode tiny-long --scale 2 --tiled_dit True --tile_size 128 --unload_dit True`
- **显存 12-16GB**: `--mode tiny --scale 4 --tiled_dit True --tile_size 256`
- **显存 16-24GB**: `--mode tiny --scale 4 --tiled_dit True --tile_size 256 --adaptive_batch_size`
- **显存 > 24GB**: `--mode full --scale 4 --tiled_dit True --tile_size 512 --adaptive_batch_size`

### ComfyUI 节点

- **mode（模式）：**  
  `tiny` → 更快（默认）；`tiny-long` → 长视频低显存；`full` → 更高质量  
- **scale（放大倍数）：**  
  支持 `2`, `3`, `4` 倍放大（不限于2的倍数），通常使用 `4` 效果更好；如果显存不足，可使用 `2` 或 `3`  
- **color_fix（颜色修正）：**  
  使用小波变换方法修正输出视频的颜色偏差。  
- **tiled_vae（VAE分块解码）：**  
  启用后可显著降低显存占用，但会降低解码速度。  
- **tiled_dit（DiT分块计算）：**  
  大幅减少显存占用，但会降低推理速度。  
- **tile_size / tile_overlap（分块大小与重叠）：**  
  控制输入视频在推理时的分块方式。  
- **unload_dit（卸载DiT模型）：**  
  解码前卸载 DiT 模型以降低显存峰值，但会略微降低速度。  

## 安装步骤

#### Docker 配置说明

**GPU 配置：**
- 默认配置：容器仅使用 GPU1 和 GPU2（物理 GPU），GPU0 不会被使用
- 容器内重新编号：GPU1 和 GPU2 在容器内会被重新编号为 `cuda:0` 和 `cuda:1`
- 多 GPU 模式：使用 `--multi_gpu` 时，会自动使用容器内可见的所有 GPU（即 GPU1 和 GPU2）
- 如需修改 GPU 配置，请编辑 `docker-compose.yml` 中的 `NVIDIA_VISIBLE_DEVICES` 和 `device_ids` 参数

#### 安装节点:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast.git
python -m pip install -r ComfyUI-FlashVSR_Ultra_Fast/requirements.txt
```
📢: 要在RTX20系或更早的GPU上运行, 请安装`triton<3.3.0`:  

```bash
# Windows
python -m pip install -U triton-windows<3.3.0
# Linux
python -m pip install -U triton<3.3.0
```

#### 模型下载:
- 从[这里](https://huggingface.co/JunhaoZhuang/FlashVSR)下载整个`FlashVSR`文件夹和它里面的所有文件, 并将其放到`ComfyUI/models`目录中。  

```
├── ComfyUI/models/FlashVSR
|     ├── LQ_proj_in.ckpt
|     ├── TCDecoder.ckpt
|     ├── diffusion_pytorch_model_streaming_dmd.safetensors
|     ├── Wan2.1_VAE.pth
```

## 性能优化

### 多GPU并行处理
对于拥有2+个GPU的系统，启用 `--multi_gpu` 可获得接近线性的加速：
- 自动将视频帧分割到可用GPU
- 每个GPU独立处理视频片段
- 结果无缝合并，处理重叠区域

### 自适应批处理大小
启用 `--adaptive_batch_size` 最大化GPU利用率：
- 根据可用显存动态调整tile批处理大小
- 对于32GB GPU，可同时处理6-16个tiles
- 如果显存变化，处理过程中自动重新平衡

**预期性能：**
- **双GPU + 自适应批处理**: 相比单GPU提升3-5倍
- **显存使用**: 32GB GPU峰值使用20-25GB（未优化时约13GB）

## 故障排除

### GPU访问问题

**问题：`RuntimeError: No CUDA GPUs are available`**

**解决方案：**
1. 检查宿主机GPU：`nvidia-smi`
2. 检查容器GPU访问：`docker exec flashvsr_ultra_fast nvidia-smi`
3. 如果容器内无法访问GPU，重启容器：
   ```bash
   docker-compose down
   docker-compose up -d
   docker exec flashvsr_ultra_fast nvidia-smi
   ```
4. 检查Docker GPU支持：`docker info | grep -i runtime`（应该看到`nvidia`）
5. 如果缺少nvidia runtime，安装nvidia-docker2或nvidia-container-toolkit

### 显存不足（OOM）

**解决方案：**
- 使用 `--mode tiny-long`
- 减小 `--tile_size`（如128或64）
- 启用 `--unload_dit True`
- 使用 `--scale 2` 或 `3` 或`4`
- 减小 `--kv_ratio`（如1.0）

### 视频读取问题

**问题：`torchvision read_video failed: PyAV is not installed`**

**解决方案：**
1. 安装 PyAV（推荐，以获得更好的性能和兼容性）：
   ```bash
   pip install av
   ```
   或者重新安装所有依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 如果未安装 PyAV，代码会自动回退到 OpenCV 或 FFmpeg，功能不受影响，但性能可能略低
3. 确保已安装 FFmpeg（用于视频编解码）：
   ```bash
   # Ubuntu/Debian
   sudo apt-get install -y ffmpeg libavcodec-dev
   
   # 或在 Docker 容器中
   apt-get update && apt-get install -y ffmpeg libavcodec-dev
   ```

### 多GPU使用注意事项

- 需要至少2个GPU才能启用 `--multi_gpu`
- 每个GPU都会加载完整模型，显存需求不变
- 适用于长视频，短视频可能不会显著加速
- 确保所有GPU都有足够显存

## Multi-GPU + Segmented 模式详解

### 概述
当同时使用 `--multi_gpu` 和 `--segmented` 时，视频会经过两层分割：
1. **第一层（multi_gpu）**：按GPU数量分割成多个worker segments
2. **第二层（segmented）**：每个worker内部再分割成多个sub-segments

### 详细流程

#### 1. 第一层分割：multi_gpu模式

**分割逻辑：**
- 函数：`split_video_by_frames(frames, num_gpus, overlap=segment_overlap)`
- 计算方式：
  ```python
  segment_size = N // num_gpus  # N是总帧数
  for i in range(num_gpus):
      start_idx = max(0, i * segment_size - overlap if i > 0 else 0)
      end_idx = min(N, (i + 1) * segment_size + overlap if i < num_gpus - 1 else N)
  ```

**示例（612帧，2个GPU，overlap=2）：**
- **Segment 0 (Worker 0)**: frames 0-308 (共308帧)
- **Segment 1 (Worker 1)**: frames 304-612 (共308帧)
  - 注意：有4帧overlap (308-304=4)

#### 2. 目录结构和文件命名

**主目录名：**
- 函数：`get_video_based_dir_name(input_path, scale)`
- 格式：`{视频名}_{scale}x`
- 示例：`3D_cat_1080_30fps_4x`

**multi_gpu checkpoint：**
- **路径**：`/tmp/flashvsr_checkpoints/{video_dir_name}/`
- **文件**：`checkpoint.json`
- **内容**：记录每个worker的绝对帧范围和输出文件路径

**multi_gpu worker输出：**
- **路径**：`/tmp/flashvsr_multigpu/{video_dir_name}/`
- **文件命名**：`worker_{worker_id}_{uuid}.pt`
  - `worker_id`: 0, 1, 2, ... (对应segment索引)
  - `uuid`: 随机UUID，避免文件名冲突

#### 3. 第二层分割：segmented模式（在worker内部）

**如果worker启用了segmented模式：**
每个worker进程会：
1. 接收分配给它的frames（例如Worker 0接收frames 0-308）
2. 在worker内部，再次分割成多个sub-segments
3. 每个sub-segment独立处理并保存

**segmented目录结构：**
- **路径**：`/tmp/flashvsr_segments/{video_dir_name}/`
- **video_dir_name的确定**：
  - 如果在worker模式下：`worker_{worker_start_idx}_{worker_end_idx}_{scale}x`
    - 示例：`worker_0_308_4x` (Worker 0处理frames 0-308)
  - 如果不在worker模式：使用`get_video_based_dir_name(input_path, scale)`

**segmented文件命名：**
- **.pt文件**：`segment_{seg_idx:04d}.pt`
  - `seg_idx`: 0, 1, 2, ... (sub-segment索引，从0开始)
- **.json元数据文件**：`segment_{seg_idx:04d}.json`
  - 记录绝对帧范围（相对于原始视频）

#### 4. 完整示例流程

假设：视频612帧，2个GPU，启用segmented，每个sub-segment最大100帧

**步骤1：multi_gpu分割**
```
原始视频: 612帧
├── Worker 0: frames 0-308 (308帧)
└── Worker 1: frames 304-612 (308帧)
```

**步骤2：Worker 0内部segmented分割**
```
Worker 0接收: 308帧
├── Sub-segment 0: frames 0-100 (相对于worker: 0-100, 绝对: 0-100)
├── Sub-segment 1: frames 98-200 (相对于worker: 98-200, 绝对: 98-200)
├── Sub-segment 2: frames 198-300 (相对于worker: 198-300, 绝对: 198-300)
└── Sub-segment 3: frames 298-308 (相对于worker: 298-308, 绝对: 298-308)

保存位置: /tmp/flashvsr_segments/worker_0_308_4x/
├── segment_0000.pt + segment_0000.json
├── segment_0001.pt + segment_0001.json
├── segment_0002.pt + segment_0002.json
└── segment_0003.pt + segment_0003.json
```

**步骤3：Worker 0合并sub-segments**
```
Worker 0处理完所有sub-segments后：
1. 按seg_idx顺序加载所有sub-segments
2. 处理overlap（跳过重复帧）
3. 合并成最终输出
4. 保存到: /tmp/flashvsr_multigpu/3D_cat_1080_30fps_4x/worker_0_{uuid}.pt
```

**步骤4：主进程合并所有workers**
```
主进程：
1. 从checkpoint.json读取所有worker信息
2. 按start_idx排序
3. 加载每个worker的输出文件
4. 处理overlap（Worker 1跳过前4帧）
5. 合并成最终视频
```

#### 5. 关键点总结

1. **目录命名规则**：
   - multi_gpu: `/tmp/flashvsr_multigpu/{video_dir_name}/`
   - segmented (worker模式): `/tmp/flashvsr_segments/worker_{start}_{end}_{scale}x/`
   - segmented (非worker模式): `/tmp/flashvsr_segments/{video_dir_name}/`
   - checkpoint: `/tmp/flashvsr_checkpoints/{video_dir_name}/`

2. **文件命名规则**：
   - worker输出: `worker_{worker_id}_{uuid}.pt`
   - sub-segment: `segment_{seg_idx:04d}.pt` + `segment_{seg_idx:04d}.json`

3. **帧范围记录**：
   - checkpoint.json: 记录worker的绝对帧范围
   - segment_*.json: 记录sub-segment的绝对帧范围（相对于原始视频）

4. **Overlap处理**：
   - multi_gpu层：worker之间有overlap（例如4帧）
   - segmented层：sub-segment之间有overlap（例如2帧）
   - 合并时都会跳过overlap部分

5. **断点续传**：
   - multi_gpu: 检查`/tmp/flashvsr_checkpoints/{video_dir_name}/checkpoint.json`
   - segmented: 检查`/tmp/flashvsr_segments/{video_dir_name}/segment_*.pt`文件
   - 使用`--resume`参数启用断点续传，否则默认覆盖重新开始

### 恢复工具

如果处理过程中断，可以使用恢复工具手动合并已完成的文件：

**从worker文件恢复：**
```bash
python recover_from_workers.py /tmp/flashvsr_multigpu/{video_dir_name} /app/output/recovered.mp4 --fps 30
```

**查找未合并的文件：**
```bash
python find_unmerged.py
```

## 致谢
- [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) @OpenImagingLab  
- [Sparse_SageAttention](https://github.com/jt-zhang/Sparse_SageAttention_API) @jt-zhang
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
