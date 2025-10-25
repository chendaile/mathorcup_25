# MathOrCup 2025

基于人工智能的表面缺陷检测和图像增强工具集，为 MathOrCup 2025 竞赛开发。

## 项目概述

本项目提供以下命令行工具：
- **YOLO 模型训练**：训练目标检测模型以识别表面缺陷，包括：
  - **Dent（凹痕）**：表面凹陷
  - **Hole（孔洞）**：表面孔洞
  - **Rusty（锈蚀）**：锈蚀损伤
- **图像增强对比**：对比不同生成对抗网络模型的超分辨率效果

## 功能特性

### YOLO 训练
- 易用的命令行界面，用于 YOLO 模型训练
- 支持限时训练会话
- 自动记录训练进度日志
- 支持中断后恢复训练
- 可自定义批次大小和缓存选项
- 自动生成基于时间戳的结果命名

### 图像增强对比
- 并排对比多个基于 GAN 的超分辨率模型
- 支持 RealESR-AnimeVideoV3、RealESRGAN-x4plus 和 RealESRGAN-x4plus-anime
- 生成带模型标签的 2x2 对比网格图
- 批量处理整个图像目录

## 安装

### 前置要求

- Python >= 3.10
- Poetry（推荐）或 pip

### 使用 Poetry 安装

```bash
poetry install
```

### 使用 pip 安装

```bash
pip install -e .
```

### Real-ESRGAN 工具下载（用于图像超分辨率）

#### Windows 系统

1. 下载预编译的可执行文件：
   ```
   https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip
   ```

2. 解压缩下载的 zip 文件到项目目录

3. 使用方法：
   ```bash
   # 基本用法 - 使用 RealESRGAN-x4plus 模型
   realesrgan-ncnn-vulkan.exe -i input.jpg -o output.jpg

   # 使用 RealESRGAN-x4plus-anime 模型（适合动漫风格图像）
   realesrgan-ncnn-vulkan.exe -i input.jpg -o output.jpg -n realesrgan-x4plus-anime

   # 使用 RealESR-AnimeVideoV3 模型
   realesrgan-ncnn-vulkan.exe -i input.jpg -o output.jpg -n realesr-animevideov3

   # 批量处理文件夹中的图像
   realesrgan-ncnn-vulkan.exe -i data-bin\ori-data\images\test -o output-bin\pic_realesrgan-x4plus -n realesrgan-x4plus
   ```

4. 可用的模型选项：
   - `realesrgan-x4plus`（默认）：通用4倍超分辨率模型
   - `realesrgan-x4plus-anime`：动漫风格图像专用模型
   - `realesr-animevideov3`：动漫视频优化模型

## 使用方法

### YOLO 训练

#### 基础训练

```bash
mathorcup-train-yolo --time 2 --project my-project
```

#### 使用自定义参数进行高级训练

```bash
mathorcup-train-yolo \
  --model yolo11n.pt \
  --data data-bin/yaml-data/data.yaml \
  --batch 0.9 \
  --cache True \
  --time 2.5 \
  --project defect-detection \
  --name experiment-001
```

#### 恢复训练

```bash
mathorcup-train-yolo \
  --time 1 \
  --project my-project \
  --resume path/to/last.pt
```

### 图像增强对比

对比不同 GAN 模型的超分辨率结果：

```bash
show-diff
```

该命令将：
- 从原始测试集和三个增强输出目录中读取图像
- 创建 2x2 对比网格，显示原始图像、RealESR-AnimeVideoV3、RealESRGAN-x4plus 和 RealESRGAN-x4plus-anime 的结果
- 将对比图像保存到 `output-bin/pic_show_diff/` 目录

## 命令行参数

### YOLO 训练参数 (`mathorcup-train-yolo`)

| 参数 | 简写 | 类型 | 默认值 | 必需 | 说明 |
|----------|-------|------|---------|----------|-------------|
| `--model` | `-m` | str | `yolo11n.pt` | 否 | YOLO 模型文件路径 |
| `--data` | - | str | `data.yaml` | 否 | 数据集 YAML 文件路径 |
| `--batch` | - | float | `0.9` | 否 | 训练批次大小百分比 |
| `--cache` | - | bool | `True` | 否 | 是否缓存图像以加快训练速度 |
| `--time` | `-t` | float | - | **是** | 训练时间限制（小时） |
| `--name` | - | str | `{时间戳}` | 否 | 保存训练结果的名称 |
| `--project` | `-P` | str | - | **是** | 保存结果的项目目录 |
| `--resume` | `-R` | str | `False` | 否 | 恢复训练的检查点路径 |

### 图像对比工具 (`show-diff`)

无需参数。该工具会自动处理配置路径中的图像。

## 项目结构

```
mathorcup_25/
├── src/
│   └── mathorcup_25/
│       ├── __init__.py
│       ├── train_yolo.py     # YOLO 训练脚本
│       └── showdiff_GenerativeAdversarial_models.py  # 图像对比工具
├── data-bin/                 # 数据集目录
│   ├── ori-data/
│   │   └── images/test/      # 原始测试图像
│   └── yaml-data/
│       └── data.yaml         # 数据集配置
├── output-bin/               # 训练和处理输出
│   ├── pic_realesr-animevideov3/     # RealESR-AnimeVideoV3 结果
│   ├── pic_realesrgan-x4plus/        # RealESRGAN-x4plus 结果
│   ├── pic_realesrgan-x4plus-anime/  # RealESRGAN-x4plus-anime 结果
│   └── pic_show_diff/                # 对比网格图输出
├── Question/                 # 竞赛材料
├── pyproject.toml            # 项目配置
└── README.md                 # 本文件
```

## 数据集配置

数据集在 `data.yaml` 中配置：

```yaml
path: .
train: train/images
val: validation/images
nc: 3
names: ["Dent", "Hole", "Rusty"]
```

## 训练输出

训练结果保存到：
```
output-bin/{project}/{name}/
```

日志写入到：
```
training-log.log
```

## 示例

### YOLO 训练示例

#### 快速开始（2小时）

```bash
mathorcup-train-yolo -t 2 -P quick-test
```

#### 生产环境训练（8小时，使用自定义模型）

```bash
mathorcup-train-yolo \
  -m yolo11m.pt \
  --data data-bin/yaml-data/data.yaml \
  -t 8 \
  -P production \
  --name run-001
```

#### 恢复之前的训练

```bash
mathorcup-train-yolo \
  -t 2 \
  -P production \
  -R output-bin/production/run-001/weights/last.pt
```

### 图像增强对比示例

```bash
show-diff
```

这将为测试目录中的所有图像生成对比网格，每个网格显示原始图像以及三种不同超分辨率模型的输出结果。

## 依赖要求

- Python >= 3.10
- ultralytics >= 8.3.221, < 9.0.0
- clearml >= 2.0.2, < 3.0.0
- matplotlib（用于图像对比）

## 作者

**chendaile**
- 邮箱：3185854290@qq.com

## 许可证

本项目为 MathOrCup 2025 竞赛创建。

## 注意事项

### YOLO 训练
- 训练日志自动保存到 `training-log.log`
- 默认启用模型性能分析
- 如果未指定名称，结果将使用自动生成的时间戳保存
- 工具使用 `exist_ok=True` 以避免重用项目名称时出现错误

### 图像增强对比
- 支持 `.jpg` 和 `.png` 图像格式
- 所有对比目录必须包含文件名匹配的图像
- 输出图像保存为 PNG 文件，尺寸为 15x10 英寸
- 工具会自动创建输出目录（如果不存在）
