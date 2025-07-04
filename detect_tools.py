# encoding:utf-8
import cv2
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import csv
import os

# fontC = ImageFont.truetype("Font/platech.ttf", 20, 0)

# 绘图展示
def cv_show(name,img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def drawRectBox(image, rect, addText, fontC, color):
    """
    绘制矩形框与结果
    :param image: 原始图像
    :param rect: 矩形框坐标, int类型
    :param addText: 类别名称
    :param fontC: 字体
    :param color: 颜色
    :return:
    """
    # 绘制位置方框
    cv2.rectangle(image, (rect[0], rect[1]),
                 (rect[2], rect[3]),
                 color, 2)

    # 绘制字体背景框 - 修改到方框下方
    cv2.rectangle(image, (rect[0] - 1, rect[3]), (rect[0] + 100, rect[3] + 25), color, -1, cv2.LINE_AA)
    
    # 使用PIL绘制中文文本 - 修改到方框下方
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((rect[0]+2, rect[3]+2), addText, (255, 255, 255), font=fontC)
    imagex = np.array(img)
    return imagex


def img_cvread(path):
    # 读取含中文名的图片文件
    # img = cv2.imread(path)
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def draw_boxes(img, boxes):
    for each in boxes:
        x1 = each[0]
        y1 = each[1]
        x2 = each[2]
        y2 = each[3]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img



def cvimg_to_qpiximg(cvimg):
    height, width, depth = cvimg.shape
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    qimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)
    qpix_img = QPixmap(qimg)
    return qpix_img


def save_video():
    # VideoCapture方法是cv2库提供的读取视频方法
    cap = cv2.VideoCapture('C:\\Users\\xxx\\Desktop\\sweet.mp4')
    # 设置需要保存视频的格式"xvid"
    # 该参数是MPEG-4编码类型，文件名后缀为.avi
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 设置视频帧频
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 设置视频大小
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # VideoWriter方法是cv2库提供的保存视频方法
    # 按照设置的格式来out输出
    out = cv2.VideoWriter('C:\\Users\\xxx\\Desktop\\out.avi', fourcc, fps, size)

    # 确定视频打开并循环读取
    while (cap.isOpened()):
        # 逐帧读取，ret返回布尔值
        # 参数ret为True 或者False,代表有没有读取到图片
        # frame表示截取到一帧的图片
        ret, frame = cap.read()
        if ret == True:
            # 垂直翻转矩阵
            frame = cv2.flip(frame, 0)

            out.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # 释放资源
    cap.release()
    out.release()
    # 关闭窗口
    cv2.destroyAllWindows()


# 封装函数:图片上显示中文
def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=50):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def insert_rows(path, lines ,header):
    """
    将n行数据写入csv文件
    :param path:
    :param lines:
    :return:
    """
    no_header = False
    if not os.path.exists(path):
        no_header = True
        start_num = 1
    else:
        start_num = len(open(path).readlines())

    csv_head = header
    with open(path, 'a', newline='') as f:
        csv_write = csv.writer(f)
        if no_header:
            csv_write.writerow(csv_head)  # 写入表头

        for each_list in lines:
            # 添加序号
            each_list = [start_num] + each_list
            csv_write.writerow(each_list)
            # 序号 + 1
            start_num += 1

class Colors:
    # 用于绘制不同颜色
    def __init__(self):
        """Initialize colors for different classes."""
        # 修改颜色定义，使类别对应我们想要的颜色:
        # 0:好果-红色, 1:坏果-绿色, 2:好花-粉色, 3:坏花-黄色, 4:截点-黑色, 5:无效-灰色
        hexs = ('FF0000', '00FF00', 'FFC0CB', 'FFFF00', '000000', '808080',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array([[0, 0, 255],     # 0: 好果-红色
                                      [0, 255, 0],     # 1: 坏果-绿色
                                      [203, 192, 255], # 2: 好花-粉色
                                      [0, 255, 255],   # 3: 坏花-黄色
                                      [0, 0, 0],       # 4: 截点-黑色
                                      [128, 128, 128], # 5: 无效-灰色
                                      [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0],
                                      [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255]],
                                     dtype=np.uint8)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to rgb values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def yolo_to_location(w,h,yolo_data):
    # yolo文件转两点坐标，注意画图坐标要转换成int格式
    x_, y_, w_, h_ = yolo_data
    x1 = int(w * x_ - 0.5 * w * w_)
    x2 = int(w * x_ + 0.5 * w * w_)
    y1 = int(h * y_ - 0.5 * h * h_)
    y2 = int(h * y_ + 0.5 * h * h_)
    # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0))
    return [x1,y1,x2,y2]

def location_to_yolo(w, h, locations):
    # x1,y1左上角坐标，x2,y2右上角坐标
    x1, y1, x2, y2 = locations
    x_ = (x1 + x2) / 2 / w
    x_ = float('%.5f' % x_)
    y_ = (y1 + y2) / 2 / h
    y_ = float('%.5f' % y_)
    w_ = (x2 - x1) / w
    w_ = float('%.5f' % w_)
    h_ = (y2 - y1) / h
    h_ = float('%.5f' % h_)
    return [x_,y_,w_,h_]

def draw_yolo_data(img_path, yolo_file_path):
    # 读取yolo标注数据并显示
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    print(img.shape)
    # yolo标注数据文件名为786_rgb_0616.txt
    with open(yolo_file_path, 'r') as f:
        data = f.readlines()
        for each in data:
            temp = each.split()
            # ['1', '0.43906', '0.52083', '0.34687', '0.15']
            # YOLO转换为两点坐标x1, x2, y1, y2
            x_, y_, w_, h_ = eval(temp[1]), eval(temp[2]), eval(temp[3]), eval(temp[4])
            x1, y1, x2, y2 = yolo_to_location(w,h,[x_, y_, w_, h_])
            # 画图验证框是否正确
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.imshow('windows', img)
    cv2.waitKey(0)


def analyze_ripeness(img, location):
    """
    分析草莓成熟度的函数
    :param img: 原始图像
    :param location: 草莓位置 [x1, y1, x2, y2]
    :return: 成熟度类型（"成熟草莓" 或 "未成熟草莓"）和成熟度比例
    """
    x1, y1, x2, y2 = location
    # 确保坐标在图像范围内
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # 提取草莓区域
    strawberry_roi = img[y1:y2, x1:x2]
    if strawberry_roi.size == 0:  # 防止空区域
        return "未知", 0
    
    # 转换到HSV颜色空间更容易分析颜色
    hsv = cv2.cvtColor(strawberry_roi, cv2.COLOR_BGR2HSV)
    
    # 定义红色范围（成熟草莓通常是红色）
    # HSV中红色有两个范围
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # 创建红色掩码
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    # 计算红色区域比例
    total_pixels = strawberry_roi.shape[0] * strawberry_roi.shape[1]
    red_pixels = np.sum(red_mask > 0)
    red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0
    
    # 基于红色比例判断成熟度
    if red_ratio > 0.4:  # 可以根据实际情况调整阈值
        return "成熟果子", red_ratio
    else:
        return "未成熟果子", red_ratio



