# ComfyUI-FlashVSR_Ultra_Fast
Running FlashVSR on lower VRAM without any artifacts.   
**[[📃中文版本](./README_zh.md)]**

## Changelog
#### 2025-10-31
- **New:** Standalone `infer_video.py` script for video processing without ComfyUI
- **New:** Multi-GPU parallel processing (`--multi_gpu`) - automatically splits video by frames across GPUs
- **New:** Adaptive tile batch size (`--adaptive_batch_size`) - dynamically adjusts tile concurrency based on GPU memory
- **New:** GPU memory monitoring and optimization for 24-32GB GPUs
- **New:** Total elapsed time tracking for performance monitoring
- **Improved:** Video reading with OpenCV fallback for better codec compatibility
- **Fixed:** Empty frame handling and negative dimension errors

#### 2025-10-24
- Added long video pipeline that significantly reduces VRAM usage when upscaling long videos.

#### 2025-10-22
- Replaced `Block-Sparse-Attention` with `Sparse_Sage`, removing the need to compile any custom kernels.  
- Added support for running on RTX 50 series GPUs.

#### 2025-10-21
- Initial this project, introducing features such as `tile_dit` to significantly reducing VRAM usage.

## Preview
![](./img/preview.jpg)

## Usage

### Standalone Inference (New!)

Use `scripts/infer_video.py` or `scripts/infer_video_distributed.py` for standalone video processing without ComfyUI:

```bash
# 单 GPU 或简单场景
python scripts/infer_video.py \
  --input ./inputs/video.mp4 \
  --output ./results/output.mp4 \
  --mode tiny \
  --scale 4 \
  --tiled_dit true \
  --tile_size 256 \
  --tile_overlap 64 \
  --multi_gpu \
  --adaptive_batch_size \
  --model_dir /path/to/FlashVSR

# 多 GPU 分布式推理（推荐）
python scripts/infer_video_distributed.py \
  --input ./inputs/video.mp4 \
  --output ./results/output.mp4 \
  --mode tiny \
  --scale 4 \
  --devices all
```

**Key Parameters:**
- **`--multi_gpu`**: Enable multi-GPU parallel processing (splits video by frames across GPUs)
- **`--adaptive_batch_size`**: Enable adaptive tile batch size based on available GPU memory
- **`--model_dir`**: Path to FlashVSR model directory (default: `/data01/volumes/flashvsr/`)

### ComfyUI Nodes

- **mode:**  
`tiny` -> faster (default); `full` -> higher quality  
- **scale:**  
`4` is always better, unless you are low on VRAM then use `2`    
- **color_fix:**  
Use wavelet transform to correct the color of output video.  
- **tiled_vae:**  
Set to True for lower VRAM consumption during decoding at the cost of speed.  
- **tiled_dit:**  
Significantly reduces VRAM usage at the cost of speed.
- **tile\_size, tile\_overlap**:  
How to split the input video.  
- **unload_dit:**  
Unload DiT before decoding to reduce VRAM peak at the cost of speed.  

## Installation

#### nodes: 

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast.git
python -m pip install -r ComfyUI-FlashVSR_Ultra_Fast/requirements.txt
```
📢: For Turing or older GPU, please install `triton<3.3.0`:  

```bash
# Windows
python -m pip install -U triton-windows<3.3.0
# Linux
python -m pip install -U triton<3.3.0
```

#### models:

- Download the entire `FlashVSR` folder with all the files inside it from [here](https://huggingface.co/JunhaoZhuang/FlashVSR) and put it in the `ComfyUI/models`

```
├── ComfyUI/models/FlashVSR
|     ├── LQ_proj_in.ckpt
|     ├── TCDecoder.ckpt
|     ├── diffusion_pytorch_model_streaming_dmd.safetensors
|     ├── Wan2.1_VAE.pth
```

## Performance Optimization

### Multi-GPU Processing
For systems with 2+ GPUs, enable `--multi_gpu` to achieve near-linear speedup:
- Automatically splits video frames across available GPUs
- Each GPU processes a video segment independently
- Results are merged seamlessly with overlap handling

### Adaptive Batch Size
Enable `--adaptive_batch_size` to maximize GPU utilization:
- Dynamically adjusts tile batch size based on available VRAM
- For 32GB GPUs, can process 6-16 tiles concurrently
- Automatically rebalances during processing if memory changes

**Expected Performance:**
- **2 GPUs + Adaptive Batch**: 3-5x speedup compared to single GPU
- **VRAM Usage**: 20-25GB peak on 32GB GPUs (vs 13GB without optimizations)

## Multi-GPU + Segmented Mode Details

### Overview
When using both `--multi_gpu` and `--segmented`, the video goes through two layers of segmentation:
1. **First layer (multi_gpu)**: Split into multiple worker segments by GPU count
2. **Second layer (segmented)**: Each worker internally splits into multiple sub-segments

### Detailed Process

#### 1. First Layer: multi_gpu Mode

**Segmentation Logic:**
- Function: `split_video_by_frames(frames, num_gpus, overlap=segment_overlap)`
- Calculation:
  ```python
  segment_size = N // num_gpus  # N is total frame count
  for i in range(num_gpus):
      start_idx = max(0, i * segment_size - overlap if i > 0 else 0)
      end_idx = min(N, (i + 1) * segment_size + overlap if i < num_gpus - 1 else N)
  ```

**Example (612 frames, 2 GPUs, overlap=2):**
- **Segment 0 (Worker 0)**: frames 0-308 (308 frames)
- **Segment 1 (Worker 1)**: frames 304-612 (308 frames)
  - Note: 4 frames overlap (308-304=4)

#### 2. Directory Structure and File Naming

**Main Directory Name:**
- Function: `get_video_based_dir_name(input_path, scale)`
- Format: `{video_name}_{scale}x`
- Example: `3D_cat_1080_30fps_4x`

**multi_gpu checkpoint:**
- **Path**: `/tmp/flashvsr_checkpoints/{video_dir_name}/`
- **File**: `checkpoint.json`
- **Content**: Records absolute frame ranges and output file paths for each worker

**multi_gpu worker output:**
- **Path**: `/tmp/flashvsr_multigpu/{video_dir_name}/`
- **File naming**: `worker_{worker_id}_{uuid}.pt`
  - `worker_id`: 0, 1, 2, ... (corresponds to segment index)
  - `uuid`: Random UUID to avoid filename conflicts

#### 3. Second Layer: segmented Mode (within worker)

**If worker enables segmented mode:**
Each worker process will:
1. Receive frames assigned to it (e.g., Worker 0 receives frames 0-308)
2. Split internally into multiple sub-segments
3. Process and save each sub-segment independently

**segmented directory structure:**
- **Path**: `/tmp/flashvsr_segments/{video_dir_name}/`
- **video_dir_name determination**:
  - In worker mode: `worker_{worker_start_idx}_{worker_end_idx}_{scale}x`
    - Example: `worker_0_308_4x` (Worker 0 processes frames 0-308)
  - Non-worker mode: Use `get_video_based_dir_name(input_path, scale)`

**segmented file naming:**
- **.pt file**: `segment_{seg_idx:04d}.pt`
  - `seg_idx`: 0, 1, 2, ... (sub-segment index, starting from 0)
- **.json metadata file**: `segment_{seg_idx:04d}.json`
  - Records absolute frame ranges (relative to original video)

#### 4. Complete Example Flow

Assume: 612 frames video, 2 GPUs, segmented enabled, max 100 frames per sub-segment

**Step 1: multi_gpu segmentation**
```
Original video: 612 frames
├── Worker 0: frames 0-308 (308 frames)
└── Worker 1: frames 304-612 (308 frames)
```

**Step 2: Worker 0 internal segmented segmentation**
```
Worker 0 receives: 308 frames
├── Sub-segment 0: frames 0-100 (relative to worker: 0-100, absolute: 0-100)
├── Sub-segment 1: frames 98-200 (relative to worker: 98-200, absolute: 98-200)
├── Sub-segment 2: frames 198-300 (relative to worker: 198-300, absolute: 198-300)
└── Sub-segment 3: frames 298-308 (relative to worker: 298-308, absolute: 298-308)

Save location: /tmp/flashvsr_segments/worker_0_308_4x/
├── segment_0000.pt + segment_0000.json
├── segment_0001.pt + segment_0001.json
├── segment_0002.pt + segment_0002.json
└── segment_0003.pt + segment_0003.json
```

**Step 3: Worker 0 merge sub-segments**
```
After Worker 0 processes all sub-segments:
1. Load all sub-segments in seg_idx order
2. Handle overlap (skip duplicate frames)
3. Merge into final output
4. Save to: /tmp/flashvsr_multigpu/3D_cat_1080_30fps_4x/worker_0_{uuid}.pt
```

**Step 4: Main process merge all workers**
```
Main process:
1. Read all worker info from checkpoint.json
2. Sort by start_idx
3. Load each worker's output file
4. Handle overlap (Worker 1 skips first 4 frames)
5. Merge into final video
```

#### 5. Key Points Summary

1. **Directory naming rules**:
   - multi_gpu: `/tmp/flashvsr_multigpu/{video_dir_name}/`
   - segmented (worker mode): `/tmp/flashvsr_segments/worker_{start}_{end}_{scale}x/`
   - segmented (non-worker mode): `/tmp/flashvsr_segments/{video_dir_name}/`
   - checkpoint: `/tmp/flashvsr_checkpoints/{video_dir_name}/`

2. **File naming rules**:
   - worker output: `worker_{worker_id}_{uuid}.pt`
   - sub-segment: `segment_{seg_idx:04d}.pt` + `segment_{seg_idx:04d}.json`

3. **Frame range recording**:
   - checkpoint.json: Records absolute frame ranges for workers
   - segment_*.json: Records absolute frame ranges for sub-segments (relative to original video)

4. **Overlap handling**:
   - multi_gpu layer: Overlap between workers (e.g., 4 frames)
   - segmented layer: Overlap between sub-segments (e.g., 2 frames)
   - Overlap is skipped during merging

5. **Resume from checkpoint**:
   - multi_gpu: Check `/tmp/flashvsr_checkpoints/{video_dir_name}/checkpoint.json`
   - segmented: Check `/tmp/flashvsr_segments/{video_dir_name}/segment_*.pt` files
   - Use `--resume` parameter to enable resume, otherwise defaults to overwrite and restart

### Recovery Tools

If processing is interrupted, you can use recovery tools to manually merge completed files:

**Recover from worker files:**
```bash
python tools/recover_distributed_inference.py /tmp/flashvsr_multigpu/{video_dir_name} /app/output/recovered.mp4 --fps 30
```

**Find unmerged files:**
```bash
python tools/find_unmerged.py
```

## Acknowledgments
- [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) @OpenImagingLab  
- [Sparse_SageAttention](https://github.com/jt-zhang/Sparse_SageAttention_API) @jt-zhang
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
