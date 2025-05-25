# 用于划分数据集的脚本
import os
import random
import shutil

# 设置目录路径
image_dir = r'D:\Strawberry_detect\sourse_data\pictures_berry'  # 改成你自己的原图片目录
label_dir = r'D:\Strawberry_detect\sourse_data\labels_berry'  # 改成你自己的原标签目录

# 获取图片和txt文件列表
images = os.listdir(image_dir)
labels = os.listdir(label_dir)

# 随机打乱图片列表
random.shuffle(images)

# 计算训练集、验证集和测试集的数量
total_images = len(images)
train_count = int(total_images * 0.7)
val_count = int(total_images * 0.2)
test_count = total_images - train_count - val_count

# 分配文件到训练集、验证集和测试集
train_images = images[:train_count]
val_images = images[train_count:train_count + val_count]
test_images = images[train_count + val_count:]

# 移动文件到对应的目录
for image in train_images:
    # 移动图片和标签到训练集目录
    shutil.move(os.path.join(image_dir, image), r'D:\Strawberry_detect\datasets\Data3/train/images') # 请改成你自己的训练集存放图片的文件夹目录
    shutil.move(os.path.join(label_dir, image[:-4]+'.txt'), r'D:\Strawberry_detect\datasets\Data3/train/labels') # 请改成你自己的训练集存放标签的文件夹目录

for image in val_images:
    # 移动图片和标签到验证集目录
    shutil.move(os.path.join(image_dir, image), r'D:\Strawberry_detect\datasets\Data3\valid/images')# 请改成你自己的验证集存放图片的文件夹目录
    shutil.move(os.path.join(label_dir, image[:-4] + '.txt'), r'D:\Strawberry_detect\datasets\Data3/valid/labels') # 请改成你自己的验证集存放标签的文件夹目录

for image in test_images:
    # 移动图片和标签到测试集目录
    shutil.move(os.path.join(image_dir, image), r'D:\Strawberry_detect\datasets\Data3/test/images')# 请改成你自己的测试集存放图片的文件夹目录
    shutil.move(os.path.join(label_dir, image[:-4] + '.txt'), r'D:\Strawberry_detect\datasets\Data3\test/labels')# 请改成你自己的测试集存放标签的文件夹目录
