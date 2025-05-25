# 使用自定义数据集训练YOLOv10模型

本项目用于将包含jpg图像和json标注的数据集转换为YOLO格式，并使用YOLOv10模型进行训练。

## 目录结构

```
├── datasets/                 # 数据集目录
│   ├── Data2/                # 原有数据集
│   └── custom_dataset/       # 转换后的自定义数据集
├── json_to_yolo.py           # 数据集转换脚本
├── train_custom_dataset.py   # 自定义数据集训练脚本
├── train_v10.py              # 原有训练脚本
└── yolov10n.pt               # 预训练模型
```

## 使用步骤

### 1. 准备数据集

将您的jpg图像文件和json标注文件分别放在指定目录中。

### 2. 转换数据集

**使用命令行参数指定数据集路径**：

```
python json_to_yolo.py --images_dir /path/to/your/images --json_dir /path/to/your/json --output_dir datasets/my_dataset
```

参数说明：
- `--images_dir`：图像文件夹路径（必需）
- `--json_dir`：JSON标注文件夹路径（必需）
- `--output_dir`：输出数据集路径（可选，默认为datasets/custom_dataset）
- `--train_ratio`：训练集比例（可选，默认为0.8）
- `--val_ratio`：验证集比例（可选，默认为0.1）
- `--test_ratio`：测试集比例（可选，默认为0.1）
- `--class_names`：类别名称列表（可选）

例如，如果您想指定类别名称：
```
python json_to_yolo.py --images_dir /path/to/images --json_dir /path/to/json --class_names strain flower berry
```

转换完成后，在指定的输出目录下将生成以下结构：

```
my_dataset/
├── train/
│   ├── images/  # 训练图像
│   └── labels/  # 训练标注
├── valid/
│   ├── images/  # 验证图像
│   └── labels/  # 验证标注
├── test/
│   ├── images/  # 测试图像
│   └── labels/  # 测试标注
└── data.yaml    # 数据集配置文件
```

### 3. 训练模型

**使用命令行参数指定数据集配置文件路径**：

```
python train_custom_dataset.py --data_yaml datasets/my_dataset/data.yaml
```

参数说明：
- `--data_yaml`：数据集配置文件路径（必需）
- `--model_yaml`：模型配置文件路径（可选）
- `--pre_model`：预训练模型路径（可选，默认为yolov10n.pt）
- `--epochs`：训练轮数（可选，默认为150）
- `--batch`：批次大小（可选，默认为4）
- `--name`：保存结果的文件夹名称（可选，默认为custom_train）
- `--imgsz`：图像尺寸（可选，默认为640）
- `--device`：训练设备（可选，默认为0）

例如，如果您想使用较小的批次大小和更多的训练轮数：
```
python train_custom_dataset.py --data_yaml datasets/my_dataset/data.yaml --batch 2 --epochs 200 --name strawberry_model
```

训练完成后，模型将保存在`runs/detect/strawberry_model`目录下。

## 自定义训练参数

除了上述命令行参数外，您也可以直接编辑`train_custom_dataset.py`脚本来修改更多高级训练参数：

```python
results = model.train(
    # ... 其他参数 ...
    lr0=0.01,               # 初始学习率
    lrf=0.01,               # 最终学习率因子
    warmup_epochs=3         # 热身训练轮数
)
```

## 注意事项

1. JSON格式标注需要转换为YOLO格式（类别索引 + 边界框坐标）
2. YOLO格式的边界框坐标为归一化坐标，范围为0-1
3. 训练前请确保模型配置文件路径正确
4. 如果GPU内存不足，可以减小批次大小(batch)和图像尺寸(imgsz)
5. 如果训练过程中出现过拟合，可以增加数据增强或减小模型容量 