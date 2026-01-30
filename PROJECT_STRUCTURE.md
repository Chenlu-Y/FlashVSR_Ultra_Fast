# 项目结构重组说明

## 新的目录结构

```
FlashVSR_Ultra_Fast/
├── README.md, README_zh.md, LICENSE, requirements.txt  # 根目录文档
│
├── scripts/                    # 主要推理脚本（入口点）
│   ├── infer_video.py              # 推理入口（参数解析、流程编排、单/多 GPU 启动）
│   └── inference_runner.py         # 推理执行核心（单卡/多卡、分片、tile、pipeline 加载与运行）
│
├── tools/                      # 工具脚本
│   ├── batch_process.py
│   ├── find_unmerged.py
│   ├── recover_distributed_inference.py
│   └── recover_all_segments.py
│
├── utils/                      # 工具函数模块
│   ├── __init__.py
│   ├── inference_support/      # 推理支持（tensor/video、tile 几何、显存估算等）
│   ├── io/                     # 输入输出相关
│   │   ├── __init__.py
│   │   ├── inference_io.py     # 推理 I/O 与 HDR/SDR 编排（帧输入、Tone Mapping、保存、分布式合并）
│   │   ├── video_io.py         # 原 save_recovered_video.py
│   │   └── hdr_io.py           # 原 read_hdr_input.py
│   └── hdr/                    # HDR 相关工具
│       ├── __init__.py
│       └── tone_mapping.py     # 原 hdr_tone_mapping.py
│
├── tests/                      # 测试文件
│   ├── test_hdr_integration.py
│   ├── test_hdr_support.py
│   └── test_tone_mapping.py
│
├── docs/                       # 文档
│   ├── HDR_USAGE.md
│   ├── HDR_NORMALIZATION_EXPLANATION.md
│   └── hdr_integration_plan.md
│
├── data/                       # 数据文件
│   └── posi_prompt.pth
│
└── src/                        # 核心代码（保持不变）
    ├── models/
    ├── pipelines/
    ├── schedulers/
    └── ...
```

## 文件移动清单

### 已移动的文件

1. **主要推理脚本** → `scripts/`
   - 入口脚本命名为 `scripts/infer_video.py`（参数解析、流程编排、单/多 GPU 启动）
   - `scripts/inference_runner.py`：推理执行核心（单卡/多卡、分片、tile、pipeline 加载与运行）
   - I/O 与 HDR/SDR 编排已迁至 `utils/io/inference_io.py`，入口与 inference_runner 均依赖该模块

2. **工具脚本** → `tools/`
   - `batch_process.py` → `tools/batch_process.py`
   - `find_unmerged.py` → `tools/find_unmerged.py`
   - `recover_distributed_inference.py` → `tools/recover_distributed_inference.py`
   - `recover_all_segments.py` → `tools/recover_all_segments.py`

3. **推理支持** → `utils/`
   - `utils/inference_support/`：tensor/video 转换、tile 几何、显存估算等纯函数（包）
   - `utils/io/inference_io.py`：推理 I/O 与 HDR/SDR 编排（帧输入、Tone Mapping、保存、分布式合并）

4. **HDR 相关** → `utils/hdr/` 和 `utils/io/`
   - `hdr_tone_mapping.py` → `utils/hdr/tone_mapping.py`
   - `read_hdr_input.py` → `utils/io/hdr_io.py`
   - `save_recovered_video.py` → `utils/io/video_io.py`

5. **测试文件** → `tests/`
   - `test_hdr_integration.py` → `tests/test_hdr_integration.py`
   - `test_hdr_support.py` → `tests/test_hdr_support.py`
   - `test_tone_mapping.py` → `tests/test_tone_mapping.py`

6. **文档** → `docs/`
   - `HDR_USAGE.md` → `docs/HDR_USAGE.md`
   - `HDR_NORMALIZATION_EXPLANATION.md` → `docs/HDR_NORMALIZATION_EXPLANATION.md`
   - `hdr_integration_plan.md` → `docs/hdr_integration_plan.md`

7. **数据文件** → `data/`
   - `posi_prompt.pth` → `data/posi_prompt.pth`

## 更新的导入路径

所有文件中的导入路径已更新：

### 旧导入 → 新导入

```python
# HDR Tone Mapping
from hdr_tone_mapping import ... 
→ from utils.hdr.tone_mapping import ...

# HDR 输入读取
from read_hdr_input import ...
→ from utils.io.hdr_io import ...

# 视频保存
from save_recovered_video import ...
→ from utils.io.video_io import ...
```

## 使用方法更新

### 旧方式
```bash
python infer_video_distributed.py --input video.mp4 --output output.mp4
python recover_distributed_inference.py --checkpoint_dir ...
```

### 新方式
```bash
python scripts/infer_video.py --input video.mp4 --output output.mp4
python tools/recover_distributed_inference.py --checkpoint_dir ...
```

## 优势

1. **清晰的模块划分**：scripts/、tools/、utils/、tests/、docs/ 各司其职
2. **易于维护**：相关文件集中管理
3. **符合 Python 项目规范**：使用 utils/ 模块化工具函数
4. **便于扩展**：新功能可以轻松添加到对应目录
5. **保持兼容性**：核心代码（src/）保持不变

## 注意事项

- 所有脚本已自动添加项目根目录到 `sys.path`，确保可以正确导入 `utils` 模块
- 测试文件已更新 `sys.path` 设置，指向项目根目录
- Shell 脚本（.sh）保留在根目录，便于直接执行
