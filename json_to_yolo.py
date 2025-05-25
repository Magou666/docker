import os
import json
import shutil
from pathlib import Path
import cv2
import argparse
from PIL import Image  # 使用PIL库代替OpenCV读取图像

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='将JSON格式标注转换为YOLO格式')
    parser.add_argument('--images_dir', type=str, required=True, 
                        help='存放jpg图像的文件夹路径')
    parser.add_argument('--json_dir', type=str, required=True, 
                        help='存放json标注的文件夹路径')
    parser.add_argument('--output_dir', type=str, default='datasets/custom_dataset', 
                        help='输出YOLO格式数据集的文件夹路径 (默认: datasets/custom_dataset)')
    parser.add_argument('--train_ratio', type=float, default=0.8, 
                        help='训练集比例 (默认: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1, 
                        help='验证集比例 (默认: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1, 
                        help='测试集比例 (默认: 0.1)')
    parser.add_argument('--class_names', type=str, nargs='+', 
                        help='类别名称列表 (如不指定，将尝试从JSON文件中获取或使用默认值)')
    parser.add_argument('--json_format', type=str, default='auto',
                        choices=['auto', 'coco', 'labelme', 'custom', 'strawberry'],
                        help='JSON格式类型 (默认: auto)')
    return parser.parse_args()

# 获取类别名称列表 (需要根据您的JSON格式进行调整)
def get_classes_from_json(json_dir, json_format, user_classes=None):
    # 如果用户指定了类别，直接返回
    if user_classes and len(user_classes) > 0:
        print(f"使用用户指定的类别: {user_classes}")
        return user_classes
    
    # 否则尝试从JSON文件中获取
    classes = []
    json_files = list(Path(json_dir).glob("*.json"))
    
    if len(json_files) > 0:
        with open(json_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 根据JSON格式获取类别
            if json_format == 'auto' or json_format == 'coco':
                # 尝试COCO格式
                if "categories" in data:
                    categories = data["categories"]
                    classes = [cat["name"] for cat in categories]
                    print("从COCO格式JSON中获取类别名称")
            
            if (json_format == 'auto' or json_format == 'labelme') and not classes:
                # 尝试LabelMe格式
                if "shapes" in data:
                    shapes = data.get("shapes", [])
                    class_set = set()
                    for shape in shapes:
                        if "label" in shape:
                            class_set.add(shape["label"])
                    classes = list(class_set)
                    print("从LabelMe格式JSON中获取类别名称")
            
            if (json_format == 'auto' or json_format == 'strawberry') and not classes:
                # 尝试用户的草莓数据集格式
                if "labels" in data:
                    labels = data.get("labels", [])
                    class_set = set()
                    for label in labels:
                        if "name" in label:
                            class_set.add(label["name"])
                    classes = list(class_set)
                    print("从草莓数据集JSON中获取类别名称")
    
    # 如果无法从JSON获取类别，使用默认值
    if not classes:
        classes = ['mature', 'growth', 'flower']  # 草莓成熟度数据集的完整类别列表
        print("使用默认类别名称")
    
    return classes

# 解析COCO格式JSON
def parse_coco_json(json_file, img_width, img_height):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    annotations = []
    
    if "annotations" in data:
        for anno in data["annotations"]:
            try:
                category_id = anno["category_id"]
                bbox = anno["bbox"]  # [x, y, width, height] in COCO format
                
                # 转换为YOLO格式 [class_id, x_center, y_center, width, height]
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width = bbox[2] / img_width
                height = bbox[3] / img_height
                
                # 确保值在0-1范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                annotations.append(f"{category_id} {x_center} {y_center} {width} {height}")
            except Exception as e:
                print(f"处理COCO标注时出错: {e}")
    
    return annotations

# 解析LabelMe格式JSON
def parse_labelme_json(json_file, img_width, img_height, class_map=None):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    annotations = []
    
    if "shapes" in data:
        for shape in data["shapes"]:
            try:
                if shape["shape_type"] != "rectangle":
                    continue  # 只处理矩形标注
                
                label = shape["label"]
                points = shape["points"]
                
                # 如果提供了类别映射，使用映射的ID
                if class_map and label in class_map:
                    class_id = class_map[label]
                else:
                    # 尝试将类别名称转换为ID
                    try:
                        class_id = int(label)
                    except:
                        # 如果label不是数字，使用默认值0
                        class_id = 0
                        print(f"警告: 非数字标签 '{label}' 使用默认类别ID 0")
                
                # LabelMe中的点是[左上角, 右下角]
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # 计算YOLO格式 [class_id, x_center, y_center, width, height]
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = abs(x2 - x1) / img_width
                height = abs(y2 - y1) / img_height
                
                # 确保值在0-1范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
            except Exception as e:
                print(f"处理LabelMe标注时出错: {e}")
    
    return annotations

# 解析草莓数据集JSON格式
def parse_strawberry_json(json_file, class_map=None):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    annotations = []
    
    # 如果未提供类别映射，先尝试从文件中收集所有类别
    if not class_map:
        class_set = set()
        if "labels" in data:
            for label in data["labels"]:
                if "name" in label:
                    class_set.add(label["name"])
            
            # 创建一个简单的类别映射
            class_map = {name: i for i, name in enumerate(sorted(class_set))}
            print(f"从JSON文件自动检测到类别: {list(class_map.keys())}")
    
    if "labels" in data:
        for label in data["labels"]:
            try:
                # 获取类别名称和映射到ID
                name = label["name"]
                if class_map and name in class_map:
                    class_id = class_map[name]
                else:
                    # 如果类别映射中没有，使用默认ID 0
                    class_id = 0
                    print(f"警告: 类别名称 '{name}' 不在类别映射中，使用默认ID 0")
                
                # 获取边界框坐标
                x1 = label["x1"]
                y1 = label["y1"]
                x2 = label["x2"]
                y2 = label["y2"]
                
                # 获取图像尺寸
                img_width = label["size"]["width"]
                img_height = label["size"]["height"]
                
                # 计算YOLO格式的坐标
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = abs(x2 - x1) / img_width
                height = abs(y2 - y1) / img_height
                
                # 确保值在0-1范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
            except Exception as e:
                print(f"处理草莓数据集标注时出错: {e}, label={label}")
    
    return annotations

# 解析单个JSON文件并转换为YOLO格式
def convert_json_to_yolo(json_file, images_dir, classes, json_format='auto'):
    # 创建类别名称到ID的映射
    class_map = {name: i for i, name in enumerate(classes)}
    
    # 如果是草莓数据集格式，可以直接解析
    if json_format == 'strawberry':
        annotations = parse_strawberry_json(json_file, class_map)
        if annotations:
            # 获取图像名称 (不带扩展名)
            img_name = os.path.splitext(os.path.basename(json_file))[0]
            
            # 查找对应的图像文件
            img_path = os.path.join(images_dir, f"{img_name}.jpg")
            if not os.path.exists(img_path):
                print(f"找不到图像文件: {img_path}")
                return None
            
            return {
                "img_path": img_path,
                "annotations": annotations
            }
    
    # 读取JSON文件
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"无法读取JSON文件 {json_file}: {e}")
        return None
    
    # 获取图像名称 (不带扩展名)
    img_name = os.path.splitext(os.path.basename(json_file))[0]
    
    # 查找对应的图像文件
    img_path = os.path.join(images_dir, f"{img_name}.jpg")
    if not os.path.exists(img_path):
        print(f"找不到图像文件: {img_path}")
        return None
    
    # 使用PIL库读取图像获取宽度和高度
    try:
        img = Image.open(img_path)
        img_width, img_height = img.size
    except Exception as e:
        print(f"无法读取图像: {img_path}, 错误: {e}")
        return None
    
    # 根据JSON格式解析标注
    annotations = []
    
    if json_format == 'auto':
        # 尝试不同格式
        if "labels" in data:
            # 尝试草莓数据集格式
            annotations = parse_strawberry_json(json_file, class_map)
            if annotations:
                print(f"使用草莓数据集格式解析 {json_file}")
        elif "annotations" in data and "categories" in data:
            annotations = parse_coco_json(json_file, img_width, img_height)
            if annotations:
                print(f"使用COCO格式解析 {json_file}")
        elif "shapes" in data:
            annotations = parse_labelme_json(json_file, img_width, img_height, class_map)
            if annotations:
                print(f"使用LabelMe格式解析 {json_file}")
        else:
            print(f"无法识别JSON格式: {json_file}")
    elif json_format == 'coco':
        annotations = parse_coco_json(json_file, img_width, img_height)
    elif json_format == 'labelme':
        annotations = parse_labelme_json(json_file, img_width, img_height, class_map)
    
    # 如果没有找到任何标注，尝试简单解析
    if not annotations:
        print(f"尝试简单解析 {json_file}")
        try:
            # 在JSON的第一层直接查找关键字段
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    # 检查这个列表是否可能包含边界框
                    for item in value:
                        if isinstance(item, dict):
                            # 查找可能的边界框字段
                            bbox = None
                            class_id = 0
                            
                            # 寻找可能的类别ID字段
                            for id_key in ['category_id', 'class_id', 'label', 'class']:
                                if id_key in item:
                                    try:
                                        if isinstance(item[id_key], int):
                                            class_id = item[id_key]
                                            break
                                        elif isinstance(item[id_key], str):
                                            # 如果是类别名称，尝试从class_map中获取ID
                                            if item[id_key] in class_map:
                                                class_id = class_map[item[id_key]]
                                                break
                                    except:
                                        pass
                            
                            # 寻找可能的边界框字段
                            for bbox_key in ['bbox', 'bounding_box', 'rect', 'box']:
                                if bbox_key in item and isinstance(item[bbox_key], (list, tuple)) and len(item[bbox_key]) >= 4:
                                    bbox = item[bbox_key]
                                    break
                            
                            if bbox:
                                # 假设格式为 [x, y, width, height] 或 [x1, y1, x2, y2]
                                if len(bbox) == 4:
                                    # 判断是否为 [x1, y1, x2, y2] 格式
                                    is_xyxy = False
                                    for known_key in item.keys():
                                        if "xyxy" in known_key.lower() or "corner" in known_key.lower():
                                            is_xyxy = True
                                            break
                                    
                                    if is_xyxy:
                                        # [x1, y1, x2, y2] -> [x_center, y_center, width, height]
                                        x1, y1, x2, y2 = bbox
                                        x_center = ((x1 + x2) / 2) / img_width
                                        y_center = ((y1 + y2) / 2) / img_height
                                        width = abs(x2 - x1) / img_width
                                        height = abs(y2 - y1) / img_height
                                    else:
                                        # [x, y, width, height] -> [x_center, y_center, width, height]
                                        x, y, width, height = bbox
                                        x_center = (x + width / 2) / img_width
                                        y_center = (y + height / 2) / img_height
                                        width = width / img_width
                                        height = height / img_height
                                    
                                    # 确保值在0-1范围内
                                    x_center = max(0, min(1, x_center))
                                    y_center = max(0, min(1, y_center))
                                    width = max(0, min(1, width))
                                    height = max(0, min(1, height))
                                    
                                    annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
        except Exception as e:
            print(f"简单解析时出错: {e}")
    
    if not annotations:
        print(f"警告: 未找到任何标注 {json_file}")
    
    return {
        "img_path": img_path,
        "annotations": annotations
    }

# 主函数
def main():
    # 解析命令行参数
    args = parse_args()
    
    # 获取配置参数
    SOURCE_IMAGES_DIR = args.images_dir
    SOURCE_JSON_DIR = args.json_dir
    OUTPUT_DIR = args.output_dir
    TRAIN_RATIO = args.train_ratio
    VAL_RATIO = args.val_ratio
    TEST_RATIO = args.test_ratio
    JSON_FORMAT = args.json_format
    
    # 创建输出目录结构
    os.makedirs(f"{OUTPUT_DIR}/train/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/train/labels", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/valid/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/valid/labels", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/test/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/test/labels", exist_ok=True)
    
    # 获取类别名称
    classes = get_classes_from_json(SOURCE_JSON_DIR, JSON_FORMAT, args.class_names)
    print(f"识别到的类别: {classes}")
    
    # 获取所有JSON文件
    json_files = list(Path(SOURCE_JSON_DIR).glob("*.json"))
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 随机分配数据集
    import random
    random.shuffle(json_files)
    
    train_count = int(len(json_files) * TRAIN_RATIO)
    val_count = int(len(json_files) * VAL_RATIO)
    
    train_files = json_files[:train_count]
    val_files = json_files[train_count:train_count+val_count]
    test_files = json_files[train_count+val_count:]
    
    # 处理训练集
    process_dataset(train_files, "train", classes, SOURCE_IMAGES_DIR, OUTPUT_DIR, JSON_FORMAT)
    # 处理验证集
    process_dataset(val_files, "valid", classes, SOURCE_IMAGES_DIR, OUTPUT_DIR, JSON_FORMAT)
    # 处理测试集
    process_dataset(test_files, "test", classes, SOURCE_IMAGES_DIR, OUTPUT_DIR, JSON_FORMAT)
    
    # 创建data.yaml文件
    create_data_yaml(classes, OUTPUT_DIR)
    
    print(f"转换完成！数据集已保存到 {OUTPUT_DIR} 文件夹")

# 处理数据集
def process_dataset(json_files, dataset_type, classes, images_dir, output_dir, json_format):
    print(f"处理{dataset_type}集...")
    processed_count = 0
    empty_count = 0
    
    for json_file in json_files:
        result = convert_json_to_yolo(json_file, images_dir, classes, json_format)
        if result:
            # 检查是否有标注
            if not result["annotations"]:
                empty_count += 1
                print(f"警告: {json_file} 未找到任何标注，跳过")
                continue
                
            # 复制图像文件
            img_name = os.path.basename(result["img_path"])
            dst_img_path = f"{output_dir}/{dataset_type}/images/{img_name}"
            shutil.copy2(result["img_path"], dst_img_path)
            
            # 创建标注文件
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            dst_txt_path = f"{output_dir}/{dataset_type}/labels/{txt_name}"
            with open(dst_txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(result["annotations"]))
            
            processed_count += 1
    
    print(f"{dataset_type}集处理完成，成功转换 {processed_count} 个样本，跳过 {empty_count} 个空标注样本")

# 创建data.yaml文件
def create_data_yaml(classes, output_dir):
    yaml_content = f"""
train: {os.path.abspath(f'{output_dir}/train')}
val: {os.path.abspath(f'{output_dir}/valid')}
test: {os.path.abspath(f'{output_dir}/test')}

nc: {len(classes)}
names: {classes}
"""
    
    with open(f"{output_dir}/data.yaml", 'w', encoding='utf-8') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    main() 