import Config
import numpy as np

# 参考分辨率基准值（可调整）
REFERENCE_WIDTH = 1920
REFERENCE_HEIGHT = 1080

# 自适应比例因子，根据图像尺寸动态调整
adaptive_scale_factor = 1.0

def update_scale_factor(image_width, image_height):
    """
    根据当前图像尺寸更新比例因子
    较大分辨率的图像使用较小的比例因子，较小分辨率的图像使用较大的比例因子
    """
    global adaptive_scale_factor
    
    # 计算当前图像与参考分辨率的比例关系
    width_ratio = REFERENCE_WIDTH / max(image_width, 1)
    height_ratio = REFERENCE_HEIGHT / max(image_height, 1)
    
    # 取平均作为最终比例因子
    adaptive_scale_factor = (width_ratio + height_ratio) / 2
    
    print(f"图像尺寸: {image_width}x{image_height}, 自适应比例因子: {adaptive_scale_factor:.4f}")
    return adaptive_scale_factor

def pixels_to_cm(pixel_value, use_adaptive=True):
    """
    将像素值转换为厘米值，可选使用自适应比例
    """
    # 先转换为毫米，再转换为厘米
    if use_adaptive:
        # 使用自适应比例，保证不同分辨率图像计算结果一致性
        cm_value = pixel_value / Config.pixels_per_mm / 10 * adaptive_scale_factor
    else:
        # 使用固定比例
        cm_value = pixel_value / Config.pixels_per_mm / 10
    return cm_value

def get_physical_dimensions(box, use_adaptive=True):
    """
    计算目标的实际物理尺寸（厘米）
    """
    width_pixels = box[2] - box[0]
    height_pixels = box[3] - box[1]
    
    width_cm = pixels_to_cm(width_pixels, use_adaptive)
    height_cm = pixels_to_cm(height_pixels, use_adaptive)
    
    return width_cm, height_cm 

def get_normalized_coordinates(box, image_width, image_height):
    """
    获取归一化的坐标（0-1范围内），便于不同分辨率图像间比较
    """
    x_min = box[0] / image_width
    y_min = box[1] / image_height
    x_max = box[2] / image_width
    y_max = box[3] / image_height
    
    return [x_min, y_min, x_max, y_max] 