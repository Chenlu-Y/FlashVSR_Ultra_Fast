# HDR 归一化调整说明

## 问题：为什么需要调整归一化？

### 图片建议的代码
```python
frames_np = (frames.clamp(0, 1) * 1023.0).round().clip(0, 1023).astype('uint16')
```

### 当前实现的问题

**方案 1：每帧独立归一化（之前的实现）**
```python
# save_frame_as_dpx10 内部
frame_max = frame.max()  # 每帧独立的最大值
frame = frame / frame_max  # 每帧独立归一化
f10 = (frame * 1023.0).round().clip(0, 1023)
```
**问题**：丢失绝对亮度关系。不同帧的最大值不同，导致同一亮度在不同帧中的量化值不同。

**方案 2：直接 clamp(0, 1)（图片建议，但不适合 Tone Mapping）**
```python
frames_np = (frames.clamp(0, 1) * 1023.0).round().clip(0, 1023)
```
**问题**：如果 Inverse Tone Mapping 后值 > 1，会被截断，丢失 HDR 信息。

## 改进方案：全局归一化

### 实现方式

使用 Tone Mapping 参数中的**全局最大值**进行归一化：

```python
# 1. 从 Tone Mapping 参数中获取全局最大值
global_hdr_max = max([p.get('max_hdr', 1.0) for p in params_list])

# 2. 使用全局最大值归一化（保留绝对亮度关系）
save_frame_as_dpx10(frame, path, hdr_max=global_hdr_max)
```

### 代码修改

**`save_frame_as_dpx10` 函数**：
```python
def save_frame_as_dpx10(frame: np.ndarray, path: str, hdr_max: float = None) -> bool:
    # ...
    if frame_max > 1.0:
        if hdr_max is not None and hdr_max > 1.0:
            # 使用全局最大值（保留绝对亮度关系）✅
            frame = frame / hdr_max
        else:
            # 使用帧内最大值（每帧独立归一化）⚠️
            frame = frame / frame_max
    
    # 10-bit: 0–1023（按照图片建议的方式）
    f10 = (frame * 1023.0).round().clip(0, 1023).astype(np.uint16)
```

**调用代码**：
```python
# 获取全局 HDR 最大值
global_hdr_max = max([p.get('max_hdr', 1.0) for p in tone_mapping_params])

# 使用全局归一化
save_frame_as_dpx10(frame, frame_filename, hdr_max=global_hdr_max)
```

## 三种方案对比

| 方案 | 归一化方式 | 优点 | 缺点 |
|------|-----------|------|------|
| **方案 1**：每帧独立 | `frame / frame.max()` | 简单 | ❌ 丢失绝对亮度关系 |
| **方案 2**：clamp(0,1) | `frames.clamp(0, 1)` | 简单 | ❌ 截断 HDR 信息 |
| **方案 3**：全局归一化 | `frame / global_max` | ✅ 保留绝对亮度关系<br>✅ 不截断 HDR | 需要传递全局最大值 |

## 为什么不在代码中直接改成图片建议的方式？

1. **图片建议的方式**：`frames.clamp(0, 1) * 1023`
   - 适用于：模型输出已经在 [0, 1] 范围内
   - 不适用于：Inverse Tone Mapping 后值可能 > 1

2. **我们的 Tone Mapping 方案**：
   - 输入：HDR [0, 5] → Tone Mapping → SDR [0, 1]
   - 超分：在 SDR 范围内
   - 输出：Inverse Tone Mapping → HDR [0, 5]（值 > 1）
   - **因此不能直接 clamp(0, 1)**

3. **改进后的方案**：
   - 使用全局最大值归一化：`frame / global_hdr_max`
   - 然后量化：`* 1023.0`
   - **既保留了 HDR 信息，又保留了绝对亮度关系**

## 总结

✅ **已修改**：使用全局 HDR 最大值进行归一化，而不是每帧独立归一化
✅ **保留**：图片建议的量化方式（`* 1023.0`）
✅ **改进**：不 clamp(0, 1)，允许 HDR 值，使用全局归一化

这样既符合图片建议的 10-bit 量化方式，又保留了 HDR 信息和绝对亮度关系。
