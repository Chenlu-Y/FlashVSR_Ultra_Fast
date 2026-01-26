#!/usr/bin/env python3
"""批量处理视频脚本

使用方法：
    python batch_process.py --input_dir /app/input --output_dir /app/output [其他参数...]

示例：
    python batch_process.py \
      --input_dir /app/input \
      --output_dir /app/output \
      --mode tiny \
      --scale 4 \
      --tiled_dit True \
      --multi_gpu
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List

# 支持的视频格式
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}

def find_video_files(input_dir: str) -> List[str]:
    """查找输入目录中的所有视频文件"""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(input_path.glob(f'*{ext}'))
        video_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    # 去重并排序
    video_files = sorted(list(set(video_files)))
    return [str(f) for f in video_files]

def get_output_path(input_file: str, output_dir: str, scale: int = 4) -> str:
    """根据输入文件生成输出路径"""
    input_path = Path(input_file)
    output_path = Path(output_dir)
    
    # 生成输出文件名：原文件名_4x.mp4
    output_filename = f"{input_path.stem}_{scale}x{input_path.suffix}"
    return str(output_path / output_filename)

def process_video(input_file: str, output_file: str, args: argparse.Namespace) -> bool:
    """处理单个视频文件"""
    print(f"\n{'='*80}")
    print(f"处理视频: {os.path.basename(input_file)}")
    print(f"输出文件: {os.path.basename(output_file)}")
    print(f"{'='*80}\n")
    
    # 构建infer_video.py的命令行参数
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'infer_video.py'),
        '--input', input_file,
        '--output', output_file,
    ]
    
    # 添加其他参数（排除input_dir和output_dir）
    exclude_args = {'input_dir', 'output_dir', 'input', 'output', 'skip_existing', 'continue_on_error'}
    for key, value in vars(args).items():
        if key not in exclude_args and value is not None:
            # 对于布尔参数，如果为True则添加flag，如果为False则不添加
            if isinstance(value, bool):
                if value:
                    cmd.append(f'--{key}')
            # 对于action='store_true'的参数，如果为True则添加flag
            elif key in ['multi_gpu', 'adaptive_batch_size', 'segmented', 'resume', 'clean_checkpoint']:
                if value:
                    cmd.append(f'--{key}')
            elif isinstance(value, (list, tuple)):
                for v in value:
                    cmd.append(f'--{key}')
                    cmd.append(str(v))
            else:
                cmd.append(f'--{key}')
                cmd.append(str(value))
    
    # 执行命令
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        if result.returncode == 0:
            print(f"\n✓ 成功处理: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
            return True
        else:
            print(f"\n✗ 处理失败: {os.path.basename(input_file)} (退出码: {result.returncode})")
            return False
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 处理失败: {os.path.basename(input_file)} (错误: {e})")
        return False
    except Exception as e:
        print(f"\n✗ 处理失败: {os.path.basename(input_file)} (异常: {e})")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='批量处理视频文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础批量处理
  python batch_process.py --input_dir /app/input --output_dir /app/output --scale 4
  
  # 多GPU批量处理
  python batch_process.py --input_dir /app/input --output_dir /app/output --scale 4 --multi_gpu --tiled_dit True
  
  # 低显存批量处理
  python batch_process.py --input_dir /app/input --output_dir /app/output --mode tiny-long --scale 4 --tiled_dit True --tile_size 128
        """
    )
    
    # 必需参数
    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入视频目录路径')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出视频目录路径')
    
    # 布尔参数转换函数（与infer_video.py保持一致）
    def str_to_bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')
    
    # 从infer_video.py继承的参数
    parser.add_argument('--mode', type=str, default='tiny',
                       choices=['tiny', 'tiny-long', 'full'],
                       help='运行模式 (default: tiny)')
    parser.add_argument('--scale', type=int, default=4,
                       choices=[2, 3, 4],
                       help='超分倍数 (default: 4)')
    parser.add_argument('--tiled_dit', type=str_to_bool, default=False,
                       help='启用DiT分块计算 (True/False, default: False)')
    parser.add_argument('--tile_size', type=int, default=256,
                       help='分块大小 (default: 256)')
    parser.add_argument('--tile_overlap', type=int, default=24,
                       help='分块重叠大小 (default: 24)')
    parser.add_argument('--tiled_vae', type=str_to_bool, default=True,
                       help='启用VAE分块解码 (True/False, default: True)')
    parser.add_argument('--unload_dit', type=str_to_bool, default=False,
                       help='解码前卸载DiT模型 (True/False, default: False)')
    parser.add_argument('--multi_gpu', action='store_true',
                       help='启用多GPU并行处理')
    parser.add_argument('--adaptive_batch_size', action='store_true',
                       help='启用自适应批处理大小')
    parser.add_argument('--segmented', action='store_true',
                       help='启用分段处理模式')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='指定使用的GPU设备 (default: cuda:0)')
    parser.add_argument('--precision', type=str, default='bf16',
                       choices=['fp16', 'bf16', 'fp32'],
                       help='计算精度 (default: bf16)')
    parser.add_argument('--color_fix', type=str_to_bool, default=True,
                       help='颜色修正 (True/False, default: True)')
    parser.add_argument('--sparse_ratio', type=float, default=2.0,
                       help='稀疏比率 (default: 2.0)')
    parser.add_argument('--kv_ratio', type=float, default=3.0,
                       help='KV缓存比率 (default: 3.0)')
    parser.add_argument('--local_range', type=int, default=11,
                       choices=[9, 11],
                       help='局部范围 (default: 11)')
    parser.add_argument('--attention_mode', type=str, default='sparse_sage_attention',
                       help='注意力模式 (default: sparse_sage_attention)')
    parser.add_argument('--model_dir', type=str, default='/app/models',
                       help='模型目录路径 (default: /app/models)')
    parser.add_argument('--seed', type=int, default=0,
                       help='随机种子 (default: 0)')
    parser.add_argument('--resume', action='store_true',
                       help='启用断点续传')
    parser.add_argument('--clean_checkpoint', action='store_true',
                       help='清理checkpoint并重新开始')
    
    # 批量处理特定参数
    parser.add_argument('--skip_existing', action='store_true',
                       help='跳过已存在的输出文件')
    parser.add_argument('--continue_on_error', action='store_true',
                       help='遇到错误时继续处理下一个文件')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有视频文件
    print(f"扫描输入目录: {args.input_dir}")
    video_files = find_video_files(args.input_dir)
    
    if not video_files:
        print(f"错误: 在 {args.input_dir} 中未找到任何视频文件")
        print(f"支持的格式: {', '.join(VIDEO_EXTENSIONS)}")
        return 1
    
    print(f"找到 {len(video_files)} 个视频文件:")
    for i, vf in enumerate(video_files, 1):
        print(f"  {i}. {os.path.basename(vf)}")
    
    # 处理每个视频
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for i, input_file in enumerate(video_files, 1):
        output_file = get_output_path(input_file, args.output_dir, args.scale)
        
        # 检查是否已存在
        if args.skip_existing and os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            if file_size > 0:
                print(f"\n[{i}/{len(video_files)}] 跳过已存在的文件: {os.path.basename(output_file)}")
                skip_count += 1
                continue
        
        print(f"\n[{i}/{len(video_files)}] 开始处理...")
        
        success = process_video(input_file, output_file, args)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            if not args.continue_on_error:
                print(f"\n处理失败，停止批量处理")
                break
    
    # 输出统计信息
    print(f"\n{'='*80}")
    print(f"批量处理完成!")
    print(f"  总计: {len(video_files)} 个文件")
    print(f"  成功: {success_count} 个")
    print(f"  失败: {fail_count} 个")
    print(f"  跳过: {skip_count} 个")
    print(f"{'='*80}\n")
    
    return 0 if fail_count == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
