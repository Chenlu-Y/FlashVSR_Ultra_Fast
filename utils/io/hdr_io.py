#!/usr/bin/env python3
"""
HDR 输入读取模块

支持：
1. 10-bit DPX 图片序列读取（保持 HDR 信息）
2. HDR 视频读取（H.265/HEVC with HDR10/HLG，保持 HDR 信息）
"""

import os
import subprocess
import numpy as np
import torch
from typing import Tuple, Optional
import tempfile


def read_dpx_frame(dpx_path: str) -> np.ndarray:
    """读取单帧 10-bit DPX 文件，保持 HDR 信息。
    
    Args:
        dpx_path: DPX 文件路径
    
    Returns:
        frame: (H, W, 3) float RGB，值可能 > 1.0（HDR）
    """
    # 先使用 ffprobe 获取图像尺寸
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "default=noprint_wrappers=1:nokey=1",
        dpx_path
    ]
    
    try:
        probe_result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        lines = probe_result.stdout.decode().strip().split('\n')
        w, h = int(lines[0]), int(lines[1])
    except Exception as e:
        raise RuntimeError(f"无法获取 DPX 图像尺寸: {e}")
    
    # 使用 FFmpeg 读取 DPX，保持原始位深
    # 输出为 float32，值范围取决于原始 DPX 的位深和编码方式
    cmd = [
        "ffmpeg", "-y",
        "-i", dpx_path,
        "-f", "rawvideo",
        "-pix_fmt", "rgb48le",  # 16-bit RGB，可以承载 10-bit 或 12-bit 数据
        "-",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        # 读取原始数据
        raw_data = result.stdout
        expected_size = h * w * 3 * 2  # rgb48le: 每个通道 16-bit (2 bytes)
        
        if len(raw_data) != expected_size:
            raise RuntimeError(f"DPX 数据大小不匹配: 期望 {expected_size}, 实际 {len(raw_data)}")
        
        # 转换为 numpy array
        frame_uint16 = np.frombuffer(raw_data, dtype=np.uint16).reshape(h, w, 3)
        
        # 转换为 float
        # DPX 10-bit: 值范围 0-1023，存储在 16-bit 容器中
        # 对于 HDR DPX，可能需要根据实际编码方式调整
        # 这里假设是标准的 10-bit DPX，值在 0-1023 范围内
        # 转换为线性 float，范围 [0, 1] 或更大（取决于 HDR）
        frame_float = frame_uint16.astype(np.float32) / 1023.0
        
        # 注意：如果原始 DPX 是 HDR，值可能已经超出标准范围
        # 这里我们保持原始值，不强制限制到 [0, 1]
        # 如果值 > 1.0，说明是 HDR 内容
        
        return frame_float
    
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode('utf-8', errors='ignore')
        raise RuntimeError(f"FFmpeg 读取 DPX 失败: {stderr[:500]}")
    except Exception as e:
        raise RuntimeError(f"读取 DPX 文件失败: {e}")


def read_hdr_video_frame_range(video_path: str, start_idx: int, end_idx: int) -> Tuple[np.ndarray, float]:
    """读取 HDR 视频的指定帧范围，保持 HDR 信息。
    
    Args:
        video_path: HDR 视频文件路径
        start_idx: 起始帧索引（包含）
        end_idx: 结束帧索引（不包含）
    
    Returns:
        (frames, fps): frames 是 (N, H, W, 3) float，值可能 > 1.0（HDR）；fps 是帧率
    """
    # 使用 FFmpeg 读取 HDR 视频，保持原始位深和颜色空间
    # 输出为 float32，值范围取决于 HDR 编码方式（PQ/HLG）
    
    # 先获取视频信息
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    try:
        probe_result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        lines = probe_result.stdout.decode().strip().split('\n')
        w, h = int(lines[0]), int(lines[1])
        fps_str = lines[2]  # 格式: "30/1" 或 "29.97/1"
        fps = eval(fps_str) if '/' in fps_str else float(fps_str)
    except Exception as e:
        raise RuntimeError(f"无法获取 HDR 视频信息: {e}")
    
    # 读取指定范围的帧
    num_frames = end_idx - start_idx
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_idx / fps),  # 跳转到起始帧
        "-i", video_path,
        "-frames:v", str(num_frames),
        "-f", "rawvideo",
        "-pix_fmt", "rgb48le",  # 16-bit RGB，可以承载 HDR 数据
        "-",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        # 读取原始数据
        raw_data = result.stdout
        expected_size = num_frames * h * w * 3 * 2  # rgb48le: 每个通道 16-bit
        
        if len(raw_data) < expected_size:
            # 可能读取的帧数少于请求的帧数
            actual_frames = len(raw_data) // (h * w * 3 * 2)
            if actual_frames == 0:
                raise RuntimeError(f"未读取到任何帧")
            num_frames = actual_frames
        
        # 转换为 numpy array
        frames_uint16 = np.frombuffer(raw_data[:num_frames * h * w * 3 * 2], dtype=np.uint16)
        frames_uint16 = frames_uint16.reshape(num_frames, h, w, 3)
        
        # 转换为 float
        # HDR 视频（H.265/HEVC）通常使用 10-bit 或 12-bit 编码
        # 这里假设是 10-bit，值范围 0-1023
        # 对于 PQ/HLG 编码的 HDR，值可能需要特殊处理
        # 这里先简单转换为 [0, 1] 范围，但允许 > 1.0（HDR）
        frames_float = frames_uint16.astype(np.float32) / 1023.0
        
        # 注意：HDR 视频的值可能已经超出标准范围
        # 我们保持原始值，不强制限制到 [0, 1]
        
        return frames_float, fps
    
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode('utf-8', errors='ignore')
        raise RuntimeError(f"FFmpeg 读取 HDR 视频失败: {stderr[:500]}")
    except Exception as e:
        raise RuntimeError(f"读取 HDR 视频失败: {e}")


def detect_hdr_input(input_path: str, hdr_mode: bool = False) -> bool:
    """检测输入是否为 HDR 格式。
    
    Args:
        input_path: 输入路径（视频文件或图片序列目录）
        hdr_mode: 是否启用了 HDR 模式
    
    Returns:
        bool: 如果检测到 HDR 格式，返回 True
    """
    if not hdr_mode:
        return False
    
    if os.path.isfile(input_path):
        # 视频文件：检查扩展名和可能的 HDR 元数据
        ext = os.path.splitext(input_path)[1].lower()
        # HDR 视频通常是 .mp4, .mkv, .mov 等，但需要检查编码格式
        # 这里简单检查扩展名，实际应该用 ffprobe 检查编码格式
        return ext in {'.mp4', '.mkv', '.mov', '.mxf'}
    elif os.path.isdir(input_path):
        # 图片序列：检查是否有 .dpx 文件
        files = os.listdir(input_path)
        dpx_files = [f for f in files if f.lower().endswith('.dpx')]
        return len(dpx_files) > 0
    
    return False
