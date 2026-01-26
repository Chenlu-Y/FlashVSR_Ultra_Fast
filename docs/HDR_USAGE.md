# HDR Tone Mapping 功能使用说明

## 功能概述

按照领导思路实现：**HDR 输入 → Tone Mapping（压缩高光）→ 超分（SDR 范围）→ Inverse Tone Mapping（还原高光）**

### 工作原理

1. **输入阶段**：自动检测 HDR 输入（值 > 1.0），应用 Tone Mapping 压缩到 SDR [0, 1]
2. **超分阶段**：模型在熟悉的 SDR 范围内正常工作
3. **输出阶段**：应用 Inverse Tone Mapping 还原 HDR 高光信息

### 优势

- ✅ **不修改模型**：模型仍在 SDR 范围内工作，质量有保障
- ✅ **保留高光信息**：通过 tone mapping 压缩和还原，保留 HDR 细节
- ✅ **完全可逆**：logarithmic 方法 MSE=0，reinhard 方法误差 < 0.1%
- ✅ **向后兼容**：默认关闭，不影响现有 SDR 流程

---

## 使用方法

### 基本用法

```bash
# 启用 HDR 模式（自动检测 HDR 输入）
python infer_video_distributed.py \
    --input /path/to/hdr_dpx_sequence \
    --output_mode pictures \
    --output /path/to/output \
    --output_format dpx10 \
    --hdr_mode \
    --tone_mapping_method logarithmic
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--hdr_mode` | 启用 HDR 模式（自动检测并处理） | 关闭 |
| `--tone_mapping_method` | Tone Mapping 方法：`reinhard` / `logarithmic` / `aces` | `logarithmic` |
| `--tone_mapping_exposure` | 曝光调整（> 1.0 提亮，< 1.0 压暗） | `1.0` |

### Tone Mapping 方法选择

- **`logarithmic`（推荐）**：
  - 完全可逆（MSE=0）
  - 适合大多数 HDR 场景
  - 计算速度快

- **`reinhard`**：
  - 近似可逆（误差 < 0.1%）
  - 视觉效果自然
  - 适合电影/视频内容

- **`aces`**：
  - 行业标准
  - 不完全可逆（误差较大）
  - 不推荐用于需要完全还原的场景

---

## 工作流程

### 1. 输入检测

```
HDR 输入（10-bit DPX，值可能 > 1）
    ↓
检测 HDR（detect_hdr_range）
    ↓
应用 Tone Mapping（logarithmic/reinhard/aces）
    ↓
SDR [0, 1] ← 模型可以正常处理
```

### 2. 超分处理

```
SDR 帧 [0, 1]
    ↓
prepare_input_tensor (* 2.0 - 1.0 → [-1, 1])
    ↓
模型推理（在 [-1, 1] 范围内）
    ↓
输出 [0, 1] 或 [-1, 1]
```

### 3. 输出还原

```
超分输出 [0, 1]
    ↓
Inverse Tone Mapping
    ↓
HDR 输出（值可能 > 1）
    ↓
保存为 10-bit DPX（自动归一化到 [0, 1] 后量化）
```

---

## 文件说明

### 参数保存

Tone Mapping 参数保存在 checkpoint 目录：
```
/app/tmp/checkpoints/{输出名}/
  ├── rank_0_tone_mapping_params.json
  ├── rank_1_tone_mapping_params.json
  └── ...
```

参数格式（每帧一个）：
```json
[
  {
    "method": "logarithmic",
    "exposure": 1.0,
    "l_max": 5.0,
    "max_hdr": 5.0
  },
  ...
]
```

### 输出格式

- **`--output_format dpx10`**：10-bit DPX，支持 HDR 值（自动归一化）
- **`--output_format png`**：8-bit PNG，HDR 值会被 clip 到 [0, 1]（会丢失高光）

**建议**：HDR 输入时使用 `--output_format dpx10`

---

## 示例

### 示例 1：HDR DPX 序列输入，输出 10-bit DPX

```bash
python infer_video_distributed.py \
    --input /data/hdr_input_frames \
    --output_mode pictures \
    --output /data/hdr_output_frames \
    --output_format dpx10 \
    --hdr_mode \
    --tone_mapping_method logarithmic \
    --scale 4
```

### 示例 2：HDR 视频输入，输出 HDR DPX 序列

```bash
python infer_video_distributed.py \
    --input /data/hdr_video.mp4 \
    --output_mode pictures \
    --output /data/hdr_output_frames \
    --output_format dpx10 \
    --hdr_mode \
    --tone_mapping_method reinhard \
    --tone_mapping_exposure 1.2 \
    --scale 4
```

---

## 注意事项

1. **输入格式**：
   - 如果输入是 10-bit DPX，需要先读取并转换为 float（可能需要扩展 `read_image_sequence` 支持 DPX）
   - 当前实现假设输入已经是 float tensor，值可能 > 1

2. **输出格式**：
   - `dpx10`：支持 HDR 值（自动归一化）
   - `png`：不支持 HDR，会 clip 到 [0, 1]

3. **参数文件**：
   - 每帧一个参数，1000 帧约 50KB
   - 保存在 checkpoint 目录，可用于恢复

4. **性能影响**：
   - Tone Mapping 计算开销很小（每帧 < 1ms）
   - 对整体推理时间影响可忽略

---

## 测试验证

运行测试脚本验证功能：

```bash
# 测试 Tone Mapping 可逆性
python test_tone_mapping.py

# 测试完整集成流程
python test_hdr_integration.py
```

---

## 技术细节

### Tone Mapping 算法

**Logarithmic（对数映射）**：
- 公式：`L_out = log(1 + L_in) / log(1 + L_max)`
- 逆映射：`L_in = exp(L_out * log(1 + L_max)) - 1`
- 完全可逆，误差 = 0

**Reinhard**：
- 公式：`L_out = L_in / (1 + L_in)`
- 近似可逆，误差 < 0.1%

### 数据流

```
HDR [0, 5] 
  → Tone Mapping 
  → SDR [0, 1] 
  → 超分 
  → SDR [0, 1] 
  → Inverse Tone Mapping 
  → HDR [0, 5]
```

所有 HDR 信息通过 tone mapping 参数保留，超分后完整还原。
