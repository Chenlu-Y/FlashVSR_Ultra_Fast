# HDR Tone Mapping 集成方案

## 方案概述

按照领导思路：**HDR 输入 → Tone Mapping（压缩高光）→ 超分（SDR 范围）→ Inverse Tone Mapping（还原高光）**

## 实现步骤

### 1. 添加参数

在 `infer_video_distributed.py` 的 argument parser 中添加：

```python
parser.add_argument("--hdr_mode", action="store_true",
                   help="启用 HDR 模式：自动检测 HDR 输入，应用 Tone Mapping，超分后还原")
parser.add_argument("--tone_mapping_method", type=str, default="logarithmic",
                   choices=["reinhard", "logarithmic", "aces"],
                   help="Tone Mapping 方法（默认: logarithmic，完全可逆）")
parser.add_argument("--tone_mapping_exposure", type=float, default=1.0,
                   help="Tone Mapping 曝光调整（默认: 1.0）")
```

### 2. 输入阶段：检测 HDR 并应用 Tone Mapping

**位置**：`read_input_frames_range()` 或 `read_image_sequence()` 之后

```python
# 在 read_input_frames_range 返回后
frames, fps = read_input_frames_range(...)

# 检测 HDR
if args.hdr_mode and detect_hdr_range(frames):
    log(f"[HDR] 检测到 HDR 输入，应用 Tone Mapping ({args.tone_mapping_method})...", "info")
    frames, tone_mapping_params = apply_tone_mapping_to_frames(
        frames,
        method=args.tone_mapping_method,
        exposure=args.tone_mapping_exposure,
        per_frame=True  # 每帧独立参数，保证一致性
    )
    # 保存参数到 checkpoint_dir（如果存在）
    if checkpoint_dir:
        params_file = os.path.join(checkpoint_dir, f"rank_{rank}_tone_mapping_params.json")
        # 将 params_list 序列化保存
        with open(params_file, 'w') as f:
            json.dump(tone_mapping_params, f, default=str)  # default=str 处理 numpy/torch 类型
    log(f"[HDR] Tone Mapping 完成，SDR 范围: [{frames.min():.4f}, {frames.max():.4f}]", "info")
else:
    tone_mapping_params = None
```

### 3. 超分阶段：正常进行（SDR 范围内）

模型在 SDR [0, 1] 范围内正常工作，无需修改。

### 4. 输出阶段：应用 Inverse Tone Mapping

**位置**：`run_inference_distributed_segment()` 返回后，保存输出前

```python
# 在 run_inference_distributed_segment 返回 output 后
output = run_inference_distributed_segment(...)  # output 在 [0, 1] 或 [-1, 1]

# 如果启用了 HDR 模式，还原 HDR
if args.hdr_mode:
    # 加载 tone mapping 参数
    if checkpoint_dir:
        params_file = os.path.join(checkpoint_dir, f"rank_{rank}_tone_mapping_params.json")
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                tone_mapping_params = json.load(f)
            # 将 output 从 [-1, 1] 转换到 [0, 1]（如果需要）
            if output.min() < 0:
                output = (output + 1.0) / 2.0
            # 应用 Inverse Tone Mapping
            log(f"[HDR] 应用 Inverse Tone Mapping 还原 HDR...", "info")
            output = apply_inverse_tone_mapping_to_frames(output, tone_mapping_params)
            log(f"[HDR] HDR 还原完成，范围: [{output.min():.4f}, {output.max():.4f}]", "info")
```

### 5. 分布式场景处理

**问题**：每个 rank 处理不同的帧段，tone mapping 参数需要：
- **选项 A（推荐）**：每帧独立参数（`per_frame=True`）
  - 优点：每帧使用自己的参数，更准确
  - 缺点：参数文件稍大
- **选项 B**：全局统一参数（使用所有帧的最大值）
  - 优点：参数文件小，一致性更好
  - 缺点：需要预先扫描或使用预估值

**实现**：使用选项 A，每帧独立参数。

### 6. 参数保存格式

```json
[
  {
    "method": "logarithmic",
    "exposure": 1.0,
    "l_max": 5.0,
    "max_hdr": 5.0
  },
  {
    "method": "logarithmic",
    "exposure": 1.0,
    "l_max": 4.8,
    "max_hdr": 4.8
  },
  ...
]
```

### 7. 与现有功能集成

- **与 `--output_format dpx10` 配合**：HDR 还原后，输出 10-bit DPX
- **与 checkpoint 恢复配合**：参数保存在 checkpoint_dir，恢复时自动加载
- **与单 GPU/多 GPU 兼容**：两种模式都支持

## 优势

1. ✅ **不修改模型**：模型仍在熟悉的 SDR 范围内工作
2. ✅ **保留高光信息**：通过 tone mapping 压缩，inverse 还原
3. ✅ **可逆性好**：logarithmic 方法完全可逆（MSE=0）
4. ✅ **向后兼容**：默认关闭，不影响现有 SDR 流程

## 注意事项

1. **参数文件大小**：每帧一个参数，1000 帧约 50KB（可接受）
2. **性能影响**：tone mapping 计算开销很小（每帧 < 1ms）
3. **精度**：logarithmic 完全可逆，reinhard 近似可逆（误差 < 0.1%）
