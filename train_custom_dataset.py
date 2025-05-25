# coding: utf-8
from ultralytics import YOLO
import matplotlib
matplotlib.use('TkAgg')
import os
import argparse

# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='使用YOLOv10训练自定义数据集')
    parser.add_argument('--data_yaml', type=str, required=True,
                        help='数据集配置文件路径 (data.yaml)')
    parser.add_argument('--model_yaml', type=str, 
                        default="ultralytics/cfg/models/v10/yolov10n.yaml",
                        help='模型配置文件路径')
    parser.add_argument('--pre_model', type=str, default='yolov10n.pt',
                        help='预训练模型路径')
    parser.add_argument('--epochs', type=int, default=150,
                        help='训练轮数 (默认: 150)')
    parser.add_argument('--batch', type=int, default=4,
                        help='批次大小 (默认: 4)')
    parser.add_argument('--name', type=str, default='custom_train',
                        help='保存结果的文件夹名称 (默认: custom_train)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='图像尺寸 (默认: 640)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='训练设备 (默认: cpu)')
    return parser.parse_args()

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    
    # 获取配置参数
    model_yaml_path = args.model_yaml
    data_yaml_path = args.data_yaml
    pre_model_name = args.pre_model
    
    # 检查数据集路径是否存在
    if not os.path.exists(data_yaml_path):
        print(f"错误: 数据集配置文件 {data_yaml_path} 不存在!")
        print("请先运行 json_to_yolo.py 脚本转换数据集")
        exit(1)
        
    # 创建 YOLO 模型实例
    model = YOLO(model_yaml_path)

    # 加载预训练模型
    model.load(pre_model_name)

    # 训练模型
    results = model.train(
        data=data_yaml_path,
        epochs=args.epochs,     # 训练轮数
        batch=args.batch,       # batch大小
        name=args.name,         # 保存结果的文件夹名称
        optimizer='SGD',        # 优化器
        imgsz=args.imgsz,       # 图像尺寸
        patience=30,            # 早停参数
        save=True,              # 保存最佳模型
        device=args.device,     # 使用GPU设备
        workers=4,              # 数据加载线程数
        pretrained=True,        # 使用预训练权重
        lr0=0.01,               # 初始学习率
        lrf=0.01,               # 最终学习率因子
        warmup_epochs=3         # 热身训练轮数
    )
    
    # 训练完成后进行评估
    print("训练完成，开始评估模型...")
    model.val()
    
    print("模型训练和评估完成！")
    print(f"训练结果保存在 runs/detect/{args.name} 目录下") 