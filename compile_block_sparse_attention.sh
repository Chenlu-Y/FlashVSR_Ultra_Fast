#!/bin/bash
# 在 flashvsr_ultra_fast 容器内重新编译 block-sparse-attention
# 使用方法：
#   1. 进入容器：docker exec -it flashvsr_ultra_fast bash
#   2. 运行脚本：bash /app/FlashVSR_Ultra_Fast/compile_block_sparse_attention.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始编译 block-sparse-attention"
echo "=========================================="

# 检查是否在容器内
if [ ! -d "/app" ]; then
    echo "警告: 未检测到 /app 目录，可能不在容器内"
    echo "请确保在 flashvsr_ultra_fast 容器内运行此脚本"
fi

# 设置工作目录
WORK_DIR="/tmp/block_sparse_attention_build"
REPO_URL="https://github.com/smthemex/Block-Sparse-Attention.git"

echo "1. 检查 Python 环境..."
python --version
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "2. 清理旧的构建目录..."
rm -rf "$WORK_DIR"

echo ""
echo "3. 克隆 block-sparse-attention 仓库..."
git clone "$REPO_URL" "$WORK_DIR"
cd "$WORK_DIR"

echo ""
echo "4. 检查仓库状态..."
git log --oneline -5

echo ""
echo "5. 安装编译依赖..."
# 确保有必要的编译工具
pip install -q ninja packaging wheel

echo ""
echo "6. 开始编译安装..."
# 使用 pip 安装（会自动编译）
pip install -v --no-build-isolation .

# 或者使用 setup.py（如果 pip 安装失败）
# python setup.py install

echo ""
echo "7. 验证安装..."
python -c "
try:
    from block_sparse_attn import block_sparse_attn_func
    print('✓ block-sparse-attention 安装成功！')
    print('✓ block_sparse_attn_func 可以正常导入')
except ImportError as e:
    print('✗ 导入失败:', e)
    exit(1)
"

echo ""
echo "=========================================="
echo "编译完成！"
echo "=========================================="
echo ""
echo "现在可以在代码中使用 block_sparse_attention 了："
echo "  from block_sparse_attn import block_sparse_attn_func"
echo ""
echo "清理临时文件..."
rm -rf "$WORK_DIR"

echo "完成！"

