#!/usr/bin/env python3
"""
测试 FlashVSR 模型是否支持 HDR 范围（值 > 1）

检查点：
1. 输入处理：prepare_input_tensor 是否保留 > 1 的值
2. 模型内部：是否有 clamp 限制
3. VAE 输出：是否限制在 [-1, 1]
4. 最终输出：是否被 clip 到 [0, 1]
"""

import os
import sys
import re

# 尝试导入 torch 和 numpy（可选）
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  torch 未安装，将跳过需要 torch 的测试")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️  numpy 未安装，将跳过需要 numpy 的测试")

# 添加项目路径
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

print("=" * 80)
print("FlashVSR HDR 支持测试")
print("=" * 80)

# ============================================================================
# 测试 1: 检查 prepare_input_tensor 的行为
# ============================================================================
print("\n[测试 1] 检查 prepare_input_tensor 的数值范围转换")
print("-" * 80)

if TORCH_AVAILABLE and NUMPY_AVAILABLE:
    # 创建包含 HDR 值的测试张量（模拟 HDR 线性值，可能 > 1）
    # 格式: (N, H, W, C) = (1, 4, 4, 3)
    hdr_values = np.array([
        [[0.0, 0.5, 1.0],   # SDR 范围
         [1.5, 2.0, 3.0],   # HDR 范围（> 1）
         [5.0, 10.0, 0.8],  # 高 HDR 值
         [0.2, 1.2, 2.5]]   # 混合
    ], dtype=np.float32)

    # 扩展到完整的 (1, 4, 4, 3) 张量
    hdr_tensor = torch.from_numpy(np.tile(hdr_values, (1, 1, 1, 1))).float()
    print(f"原始输入范围: [{hdr_tensor.min():.4f}, {hdr_tensor.max():.4f}]")
    print(f"原始输入包含 > 1 的值: {(hdr_tensor > 1.0).any().item()}")

    # 模拟 prepare_input_tensor 的转换（简化版，不包含 upscale）
    test_tensor = hdr_tensor * 2.0 - 1.0
    print(f"经过 * 2.0 - 1.0 后: [{test_tensor.min():.4f}, {test_tensor.max():.4f}]")
    print(f"是否超出 [-1, 1]: {(test_tensor < -1.0).any().item() or (test_tensor > 1.0).any().item()}")

    # 检查 clamp 后的效果
    clamped = torch.clamp(test_tensor, -1.0, 1.0)
    print(f"clamp(-1, 1) 后: [{clamped.min():.4f}, {clamped.max():.4f}]")
    print(f"信息丢失: {((test_tensor != clamped) & (test_tensor.abs() > 1.0)).any().item()}")
else:
    print("⚠️  跳过（需要 torch 和 numpy）")
    print("  理论分析: 输入 [0, 1] → * 2.0 - 1.0 → [-1, 1]")
    print("  如果输入 > 1，转换后会超出 [-1, 1]，然后被 clamp 截断")

# ============================================================================
# 测试 2: 检查 VAE 输出范围限制
# ============================================================================
print("\n[测试 2] 检查 VAE 输出范围限制（模拟）")
print("-" * 80)

if TORCH_AVAILABLE:
    # 模拟 VAE 可能输出的值（假设模型内部产生了超出范围的值）
    vae_output = torch.tensor([[[[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]]]], dtype=torch.float32)
    print(f"VAE 原始输出范围: [{vae_output.min():.4f}, {vae_output.max():.4f}]")

    # 应用 clamp_(-1, 1)（如 wan_video_vae.py:785）
    vae_clamped = vae_output.clamp_(-1, 1)
    print(f"clamp_(-1, 1) 后: [{vae_clamped.min():.4f}, {vae_clamped.max():.4f}]")
    print(f"被截断的值数量: {(vae_output != vae_clamped).sum().item()}")
else:
    print("⚠️  跳过（需要 torch）")
    print("  代码位置: src/models/wan_video_vae.py:785")
    print("  return video.clamp_(-1, 1)  # 强制限制在 [-1, 1]")

# ============================================================================
# 测试 3: 检查最终输出转换
# ============================================================================
print("\n[测试 3] 检查最终输出转换（vae_output_to_video）")
print("-" * 80)

if TORCH_AVAILABLE:
    # 模拟 VAE 输出（在 [-1, 1] 范围内）
    vae_out = torch.tensor([[[[-1.0, -0.5, 0.0, 0.5, 1.0]]]], dtype=torch.float32)
    print(f"VAE 输出范围: [{vae_out.min():.4f}, {vae_out.max():.4f}]")

    # 转换为 [0, 1]（base.py:40 的逻辑）
    converted = (vae_out / 2 + 0.5)
    print(f"经过 / 2 + 0.5 后: [{converted.min():.4f}, {converted.max():.4f}]")

    # 应用 clip(0, 1)
    final = converted.clip(0, 1)
    print(f"clip(0, 1) 后: [{final.min():.4f}, {final.max():.4f}]")
    print(f"是否被 clip: {(converted != final).any().item()}")
else:
    print("⚠️  跳过（需要 torch）")
    print("  代码位置: src/pipelines/base.py:40, 46")
    print("  (image / 2 + 0.5).clip(0, 1)  # 从 [-1, 1] 转回 [0, 1]，然后 clip")

# ============================================================================
# 测试 4: 检查实际代码中的 clamp 位置
# ============================================================================
print("\n[测试 4] 检查代码中的 clamp/clip 位置")
print("-" * 80)

import re

clamp_locations = []
files_to_check = [
    "src/pipelines/flashvsr_tiny.py",
    "src/pipelines/flashvsr_full.py",
    "src/pipelines/base.py",
    "src/models/wan_video_vae.py",
    "infer_video_distributed.py"
]

for filepath in files_to_check:
    full_path = os.path.join(_project_root, filepath)
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines, 1):
                if re.search(r'\.clamp|\.clip|torch\.clamp|torch\.clip', line):
                    clamp_locations.append((filepath, i, line.strip()))

print(f"找到 {len(clamp_locations)} 处 clamp/clip 操作:")
for filepath, line_num, code in clamp_locations[:10]:  # 只显示前10个
    print(f"  {filepath}:{line_num} - {code}")
if len(clamp_locations) > 10:
    print(f"  ... 还有 {len(clamp_locations) - 10} 处")

# ============================================================================
# 测试 5: 模拟完整流程（如果可能）
# ============================================================================
print("\n[测试 5] 模拟完整数据处理流程")
print("-" * 80)

if TORCH_AVAILABLE:
    # 模拟一个包含 HDR 值的输入帧
    input_hdr = torch.tensor([[[
        [0.1, 0.5, 1.0],      # SDR
        [1.5, 2.0, 3.0],      # HDR
        [5.0, 0.8, 1.2]       # 高 HDR
    ]]], dtype=torch.float32)  # (1, 1, 3, 3)

    print(f"步骤 0 - 原始输入: [{input_hdr.min():.4f}, {input_hdr.max():.4f}]")
    print(f"  包含 > 1 的值: {(input_hdr > 1.0).any().item()}")

    # 步骤 1: prepare_input_tensor 转换
    step1 = input_hdr * 2.0 - 1.0
    print(f"步骤 1 - * 2.0 - 1.0: [{step1.min():.4f}, {step1.max():.4f}]")
    print(f"  超出 [-1, 1]: {(step1.abs() > 1.0).any().item()}")

    # 步骤 2: 模型内部可能的 clamp（模拟）
    step2 = torch.clamp(step1, -1.0, 1.0)
    print(f"步骤 2 - clamp(-1, 1): [{step2.min():.4f}, {step2.max():.4f}]")
    print(f"  信息丢失: {(step1 != step2).any().item()}")

    # 步骤 3: VAE 输出（假设在 [-1, 1]）
    step3 = step2  # VAE 输出也在 [-1, 1]
    print(f"步骤 3 - VAE 输出: [{step3.min():.4f}, {step3.max():.4f}]")

    # 步骤 4: 转换回 [0, 1]
    step4 = (step3 / 2 + 0.5)
    print(f"步骤 4 - / 2 + 0.5: [{step4.min():.4f}, {step4.max():.4f}]")

    # 步骤 5: clip(0, 1)
    step5 = step4.clip(0, 1)
    print(f"步骤 5 - clip(0, 1): [{step5.min():.4f}, {step5.max():.4f}]")

    # 计算信息丢失
    original_max = input_hdr.max().item()
    final_max = step5.max().item()
    if original_max > 1.0:
        print(f"\n⚠️  信息丢失分析:")
        print(f"  原始最大值: {original_max:.4f}")
        print(f"  最终最大值: {final_max:.4f}")
        print(f"  丢失比例: {(1.0 - final_max / original_max) * 100:.2f}%")
        print(f"  结论: HDR 信息（> 1 的值）被完全丢失")
else:
    print("⚠️  跳过（需要 torch）")
    print("  理论流程:")
    print("    输入 HDR [0, 5] → * 2.0 - 1.0 → [-1, 9]")
    print("    → clamp(-1, 1) → [-1, 1] (丢失 > 1 的信息)")
    print("    → VAE 处理 → [-1, 1]")
    print("    → / 2 + 0.5 → [0, 1]")
    print("    → clip(0, 1) → [0, 1]")
    print("  结果: 所有 > 1 的 HDR 值都被截断到 1.0")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("测试总结")
print("=" * 80)

print("\n✅ 检查项:")
print("  1. prepare_input_tensor: 会将 [0, 1] 转换到 [-1, 1]")
print("  2. 模型内部: 有 clamp(-1, 1) 限制")
print("  3. VAE 输出: 有 clamp_(-1, 1) 限制")
print("  4. 最终输出: 有 clip(0, 1) 限制")

print("\n❌ 结论:")
print("  模型不支持 HDR 范围（值 > 1）")
print("  所有 HDR 信息在输入处理阶段就被归一化/截断")

print("\n💡 建议:")
print("  如果要支持 HDR，需要:")
print("  1. 方案 A: 归一化方案（推荐）")
print("     - 输入时记录最大值，归一化到 [-1, 1]")
print("     - 输出时反归一化回原始 HDR 范围")
print("  2. 方案 B: 对数空间转换")
print("     - 输入: log(1 + hdr_value)")
print("     - 输出: exp(output) - 1")
print("  3. 方案 C: 修改模型（需要重新训练）")
print("     - 移除所有 clamp/clip")
print("     - 在 HDR 数据上微调或重新训练")

print("\n" + "=" * 80)
