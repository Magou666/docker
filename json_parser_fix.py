import os
import json
from pathlib import Path


# 自定义JSON解析函数，请根据您的JSON格式进行修改
def parse_custom_json(json_file, img_width, img_height):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    annotations = []
    
    # 在这里解析您的JSON格式并构建YOLO格式的标注
    # YOLO格式: [class_id] [x_center] [y_center] [width] [height]
    # 所有值都应该归一化到0-1范围内
    
    # 示例：假设您的JSON中包含一个"objects"列表
    # 每个对象都有"class_id"和"bbox"字段
    # "bbox"格式为[x, y, width, height]
    
    if "objects" in data:
        for obj in data["objects"]:
            class_id = obj["class_id"]
            bbox = obj["bbox"]
            
            # 转换为YOLO格式
            x_center = (bbox[0] + bbox[2] / 2) / img_width
            y_center = (bbox[1] + bbox[3] / 2) / img_height
            width = bbox[2] / img_width
            height = bbox[3] / img_height
            
            annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return annotations

# 将这个函数添加到json_to_yolo.py文件中，然后修改convert_json_to_yolo函数
# 将对应的解析函数替换掉现有的代码
