#!/usr/bin/env python3
"""
测试 Tone Mapping 的可逆性和效果
"""

import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    import torch
    from utils.hdr.tone_mapping import (
        tone_map_hdr_to_sdr,
        inverse_tone_map_sdr_to_hdr,
        detect_hdr_range,
        apply_tone_mapping_to_frames,
        apply_inverse_tone_mapping_to_frames
    )
    
    print("=" * 80)
    print("Tone Mapping 可逆性测试")
    print("=" * 80)
    
    # 测试 1: 单帧 HDR 值
    print("\n[测试 1] 单帧 HDR 值测试")
    print("-" * 80)
    
    hdr_values = torch.tensor([
        [[0.0, 0.5, 1.0],   # SDR 范围
         [1.5, 2.0, 3.0],   # HDR 范围
         [5.0, 10.0, 0.8]]   # 高 HDR
    ], dtype=torch.float32)  # (3, 3, 3)
    
    print(f"原始 HDR 范围: [{hdr_values.min():.4f}, {hdr_values.max():.4f}]")
    print(f"包含 > 1 的值: {detect_hdr_range(hdr_values)}")
    
    for method in ['reinhard', 'logarithmic', 'aces']:
        print(f"\n方法: {method}")
        sdr, params = tone_map_hdr_to_sdr(hdr_values, method=method)
        print(f"  SDR 范围: [{sdr.min():.4f}, {sdr.max():.4f}]")
        
        hdr_restored = inverse_tone_map_sdr_to_hdr(sdr, params)
        print(f"  还原 HDR 范围: [{hdr_restored.min():.4f}, {hdr_restored.max():.4f}]")
        
        # 计算误差
        mse = ((hdr_values - hdr_restored) ** 2).mean().item()
        max_error = (hdr_values - hdr_restored).abs().max().item()
        print(f"  MSE: {mse:.6f}, 最大误差: {max_error:.4f}")
    
    # 测试 2: 帧序列
    print("\n[测试 2] 帧序列测试")
    print("-" * 80)
    
    frames = torch.stack([
        torch.ones(4, 4, 3) * 0.5,   # SDR 帧
        torch.ones(4, 4, 3) * 2.0,   # HDR 帧
        torch.ones(4, 4, 3) * 5.0,   # 高 HDR 帧
    ], dim=0)  # (3, 4, 4, 3)
    
    print(f"原始帧范围: [{frames.min():.4f}, {frames.max():.4f}]")
    
    sdr_frames, params_list = apply_tone_mapping_to_frames(frames, method='reinhard', per_frame=True)
    print(f"SDR 帧范围: [{sdr_frames.min():.4f}, {sdr_frames.max():.4f}]")
    print(f"参数数量: {len(params_list)}")
    
    hdr_frames_restored = apply_inverse_tone_mapping_to_frames(sdr_frames, params_list)
    print(f"还原 HDR 帧范围: [{hdr_frames_restored.min():.4f}, {hdr_frames_restored.max():.4f}]")
    
    mse = ((frames - hdr_frames_restored) ** 2).mean().item()
    print(f"整体 MSE: {mse:.6f}")
    
    # 测试 3: 模拟完整流程
    print("\n[测试 3] 完整流程模拟（HDR → Tone Map → 超分模拟 → Inverse Tone Map）")
    print("-" * 80)
    
    original_hdr = torch.tensor([[[
        [0.1, 0.5, 1.0],
        [1.5, 2.0, 3.0],
        [5.0, 0.8, 1.2]
    ]]], dtype=torch.float32)  # (1, 3, 3, 3)
    
    print(f"步骤 0 - 原始 HDR: [{original_hdr.min():.4f}, {original_hdr.max():.4f}]")
    
    # 步骤 1: Tone Mapping
    sdr, params = tone_map_hdr_to_sdr(original_hdr, method='reinhard')
    print(f"步骤 1 - Tone Mapping → SDR: [{sdr.min():.4f}, {sdr.max():.4f}]")
    
    # 步骤 2: 模拟超分（在 SDR 范围内，模型处理）
    # 这里只是模拟，实际超分会改变分辨率
    upscaled_sdr = sdr  # 假设超分后仍在 [0, 1]
    print(f"步骤 2 - 超分后 SDR: [{upscaled_sdr.min():.4f}, {upscaled_sdr.max():.4f}]")
    
    # 步骤 3: Inverse Tone Mapping
    restored_hdr = inverse_tone_map_sdr_to_hdr(upscaled_sdr, params)
    print(f"步骤 3 - Inverse Tone Mapping → HDR: [{restored_hdr.min():.4f}, {restored_hdr.max():.4f}]")
    
    mse = ((original_hdr - restored_hdr) ** 2).mean().item()
    max_error = (original_hdr - restored_hdr).abs().max().item()
    print(f"\n还原误差: MSE={mse:.6f}, 最大误差={max_error:.4f}")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
    
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在正确的环境中运行（需要 torch）")
