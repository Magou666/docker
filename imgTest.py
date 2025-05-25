#coding:utf-8
from ultralytics import YOLO
import cv2
import os

# 所需加载的模型目录
path = r'D:\Strawberry_detect\models\best.pt'
# 需要检测的图片地址
img_path = r"D:\Strawberry_detect\TestFiles\4.jpg"
output_dir = r"D:\Strawberry_detect\Results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 加载预训练模型
model = YOLO(path, task='detect')

# 设置置信度阈值
confidence_threshold = 0.10

# 检测图片，并设置参数
results = model.predict(source=img_path, conf=confidence_threshold)

# 获取第一个结果并绘制
res = results[0].plot()
res = cv2.resize(res, dsize=None, fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)

# 显示结果图像
cv2.imshow("YOLOv10 Detection", res)
cv2.waitKey(0)