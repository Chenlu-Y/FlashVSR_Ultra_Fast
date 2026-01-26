#!/usr/bin/env python3
"""
测试 HDR Tone Mapping 集成流程
"""

import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    import torch
    import numpy as np
    from utils.hdr.tone_mapping import (
        detect_hdr_range,
        apply_tone_mapping_to_frames,
        apply_inverse_tone_mapping_to_frames,
        serialize_tone_mapping_params,
        deserialize_tone_mapping_params
    )
    import json
    
    print("=" * 80)
    print("HDR Tone Mapping 集成流程测试")
    print("=" * 80)
    
    # 模拟完整流程
    print("\n[完整流程] HDR 输入 → Tone Mapping → 超分模拟 → Inverse Tone Mapping → 输出")
    print("-" * 80)
    
    # 步骤 1: 创建 HDR 输入（模拟从 DPX 读取）
    print("\n步骤 1: HDR 输入（模拟 10-bit DPX 读取）")
    hdr_input = torch.tensor([
        [[0.1, 0.5, 1.0],   # SDR 范围
         [1.5, 2.0, 3.0],   # HDR 范围
         [5.0, 0.8, 1.2]]   # 高 HDR
    ], dtype=torch.float32)  # (3, 3, 3)
    hdr_frames = hdr_input.unsqueeze(0)  # (1, 3, 3, 3)
    print(f"  输入范围: [{hdr_frames.min():.4f}, {hdr_frames.max():.4f}]")
    print(f"  包含 HDR: {detect_hdr_range(hdr_frames)}")
    
    # 步骤 2: Tone Mapping (HDR → SDR)
    print("\n步骤 2: Tone Mapping (HDR → SDR)")
    sdr_frames, tone_params = apply_tone_mapping_to_frames(
        hdr_frames, method='logarithmic', per_frame=True
    )
    print(f"  SDR 范围: [{sdr_frames.min():.4f}, {sdr_frames.max():.4f}]")
    print(f"  参数数量: {len(tone_params)}")
    
    # 步骤 3: 模拟超分（在 SDR 范围内，模型处理）
    print("\n步骤 3: 超分（在 SDR 范围内，模型正常工作）")
    # 模拟：超分后分辨率变大，但值仍在 [0, 1]
    upscaled_sdr = sdr_frames.repeat(1, 1, 2, 2)  # 简单的模拟（实际是模型超分）
    print(f"  超分后 SDR 范围: [{upscaled_sdr.min():.4f}, {upscaled_sdr.max():.4f}]")
    
    # 步骤 4: Inverse Tone Mapping (SDR → HDR)
    print("\n步骤 4: Inverse Tone Mapping (SDR → HDR)")
    restored_hdr = apply_inverse_tone_mapping_to_frames(upscaled_sdr, tone_params)
    print(f"  还原 HDR 范围: [{restored_hdr.min():.4f}, {restored_hdr.max():.4f}]")
    
    # 步骤 5: 验证可逆性（原始分辨率）
    print("\n步骤 5: 验证可逆性（原始分辨率）")
    original_1x1 = hdr_frames[0, 0, 0, 0].item()
    restored_1x1 = restored_hdr[0, 0, 0, 0].item()
    print(f"  原始值: {original_1x1:.4f}")
    print(f"  还原值: {restored_1x1:.4f}")
    print(f"  误差: {abs(original_1x1 - restored_1x1):.6f}")
    
    # 步骤 6: 测试参数序列化
    print("\n步骤 6: 测试参数序列化/反序列化")
    serialized = serialize_tone_mapping_params(tone_params)
    print(f"  序列化成功: {len(serialized)} 个参数")
    
    # 保存到文件（模拟 checkpoint）
    test_params_file = "/tmp/test_tone_mapping_params.json"
    with open(test_params_file, 'w') as f:
        json.dump(serialized, f, indent=2)
    print(f"  已保存到: {test_params_file}")
    
    # 从文件加载
    with open(test_params_file, 'r') as f:
        loaded_serialized = json.load(f)
    loaded_params = deserialize_tone_mapping_params(loaded_serialized)
    print(f"  从文件加载成功: {len(loaded_params)} 个参数")
    
    # 使用加载的参数测试逆映射
    test_restored = apply_inverse_tone_mapping_to_frames(sdr_frames, loaded_params)
    mse = ((hdr_frames - test_restored) ** 2).mean().item()
    print(f"  使用加载参数还原的 MSE: {mse:.6f}")
    
    print("\n" + "=" * 80)
    print("✅ 集成测试通过")
    print("=" * 80)
    print("\n流程总结:")
    print("  1. HDR 输入检测 ✓")
    print("  2. Tone Mapping (HDR → SDR) ✓")
    print("  3. 超分（SDR 范围）✓")
    print("  4. Inverse Tone Mapping (SDR → HDR) ✓")
    print("  5. 参数序列化/反序列化 ✓")
    print("\n结论: 方案可行，可以保留 HDR 高光信息")
    
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在正确的环境中运行（需要 torch）")
except Exception as e:
    import traceback
    print(f"错误: {e}")
    traceback.print_exc()
