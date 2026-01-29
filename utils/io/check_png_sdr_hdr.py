#!/usr/bin/env python3
"""
自测 PNG 是 SDR 还是 HDR。

用法:
  python utils/io/check_png_sdr_hdr.py <一张 PNG 或 包含 PNG 的目录>

示例:
  python utils/io/check_png_sdr_hdr.py ./inputs/frames/frame_000000.png
  python utils/io/check_png_sdr_hdr.py ./inputs/frames/
"""
import os
import sys

def main():
    if len(sys.argv) < 2:
        print("用法: python utils/io/check_png_sdr_hdr.py <PNG文件或目录>")
        sys.exit(1)
    path = sys.argv[1].rstrip(os.sep)
    if not os.path.exists(path):
        print(f"错误: 路径不存在: {path}")
        sys.exit(1)

    import numpy as np
    try:
        import cv2
    except ImportError:
        print("需要 OpenCV: pip install opencv-python")
        sys.exit(1)

    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted([
            os.path.join(path, f) for f in os.listdir(path)
            if f.lower().endswith('.png')
        ])
        if not files:
            print(f"目录下没有 PNG 文件: {path}")
            sys.exit(1)
        # 只检查前几张 + 中间一张，避免太多
        if len(files) > 5:
            mid = len(files) // 2
            files = [files[0], files[mid], files[-1]]
        else:
            files = files[:5]

    print(f"检查 {len(files)} 个文件: {[os.path.basename(f) for f in files]}\n")
    bit_depths = []
    max_vals = []
    min_vals = []

    for fp in files:
        # OpenCV 读 PNG: 8-bit 为 0-255，16-bit 为 0-65535
        im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if im is None:
            print(f"  跳过（无法读取）: {fp}")
            continue
        if im.ndim == 3 and im.shape[2] == 3:
            pass
        elif im.ndim == 2:
            im = np.stack([im] * 3, axis=-1)
        else:
            print(f"  跳过（通道数异常）: {fp} shape={im.shape}")
            continue

        dtype = im.dtype
        if dtype == np.uint8:
            bit_depths.append(8)
            mx, mn = float(im.max()), float(im.min())
            max_vals.append(mx / 255.0)
            min_vals.append(mn / 255.0)
        elif dtype == np.uint16:
            bit_depths.append(16)
            mx, mn = float(im.max()), float(im.min())
            # 归一化到 0-1（按 65535）
            max_vals.append(mx / 65535.0)
            min_vals.append(mn / 65535.0)
        else:
            print(f"  跳过（类型 {dtype}）: {fp}")
            continue
        print(f"  {os.path.basename(fp)}: {im.shape} {dtype}, 原始 min={im.min()}, max={im.max()}, 归一化 [0,1] 后 min={min_vals[-1]:.4f}, max={max_vals[-1]:.4f}")

    if not max_vals:
        print("没有可用的 PNG 数据")
        sys.exit(1)

    max_norm = max(max_vals)
    min_norm = min(min_vals)
    all_8bit = all(b == 8 for b in bit_depths)
    any_16bit = any(b == 16 for b in bit_depths)

    print("\n" + "=" * 50)
    print("结论（经验规则，非绝对）:")
    print("=" * 50)
    print(f"  位深: {'全部 8-bit' if all_8bit else '存在 16-bit'}")
    print(f"  归一化后数值范围: [{min_norm:.4f}, {max_norm:.4f}]")
    if all_8bit and max_norm <= 1.0 and min_norm >= 0:
        print("  → 判定: 大概率 SDR（8-bit，且值在 0–1 内）")
        print("  建议: 使用 --dynamic_range sdr")
    elif any_16bit and max_norm > 1.01:
        print("  → 判定: 可能为 HDR 或高动态范围（16-bit 且归一化后 max > 1）")
        print("  建议: 若来源是 HDR 管线，可尝试 --dynamic_range hdr")
    elif any_16bit and max_norm <= 1.0:
        print("  → 判定: 可能是高精度 SDR（16-bit 但值仍在 0–1）")
        print("  建议: 使用 --dynamic_range sdr")
    else:
        print("  → 判定: 倾向 SDR（值在常规 0–1 范围）")
        print("  建议: 使用 --dynamic_range sdr")
    print("=" * 50)

if __name__ == "__main__":
    main()
