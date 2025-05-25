# -*- coding: utf-8 -*-
import time
from PyQt5.QtWidgets import QApplication , QMainWindow, QFileDialog, \
    QMessageBox,QHeaderView,QTableWidgetItem, QAbstractItemView
import sys
import os
from ultralytics import YOLO
sys.path.append('UIProgram')
from UIProgram.UiMain import Ui_MainWindow
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal,QCoreApplication
import detect_tools as tools
import cv2
import Config
from UIProgram.QssLoader import QSSLoader
from UIProgram.precess_bar import ProgressBar
import numpy as np
import torch
import csv
import warnings
from PIL import ImageFont
import subprocess
warnings.filterwarnings('ignore')

# 导入修改ultralytics库的颜色映射
from ultralytics.utils.plotting import Colors as UltraColors

class MainWindow(QMainWindow):
    def __init__(self, parent=None):

        super(QMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # 加载css渲染效果
        style_file = 'UIProgram/style.css'
        qssStyleSheet = QSSLoader.read_qss_file(style_file)
        self.setStyleSheet(qssStyleSheet)

        self.conf = 0.25   # 设置检测置信度值
        self.iou = 0.5     # 设置检测IOU值

        self.initMain()
        self.signalconnect()

    def signalconnect(self):
        self.ui.PicBtn.clicked.connect(self.open_img)
        self.ui.comboBox.activated.connect(self.combox_change)
        self.ui.VideoBtn.clicked.connect(self.vedio_show)
        self.ui.CapBtn.clicked.connect(self.camera_show)
        self.ui.SaveBtn.clicked.connect(self.save_detect_result)
        self.ui.ExitBtn.clicked.connect(self.open_train_module)
        self.ui.FilesBtn.clicked.connect(self.detact_batch_imgs)
        self.ui.doubleSpinBox.valueChanged.connect(self.conf_value_change)
        self.ui.doubleSpinBox_2.valueChanged.connect(self.iou_value_change)
        self.ui.tableWidget.cellClicked.connect(self.on_cell_clicked)

    def initMain(self):
        self.show_width = 770
        self.show_height = 480
        self.org_path = None
        self.is_camera_open = False
        self.cap = None
        self.device = 0 if torch.cuda.is_available() else 'cpu'

        # 加载检测模型
        self.model = YOLO(Config.model_path, task='detect')
        self.model(np.zeros((48, 48, 3)).astype(np.uint8), device=self.device)  #预先加载推理模型

        # 用于绘制不同颜色矩形框
        self.colors = tools.Colors()
        
        # 初始化中文字体，用于绘制中文标签
        self.fontC = ImageFont.truetype("simsun.ttc", 20, encoding="utf-8")

        # 更新视频图像
        self.timer_camera = QTimer()

        # 更新检测信息表格
        self.timer_info = QTimer()
        # 保存视频
        self.timer_save_video = QTimer()

        # 表格
        self.ui.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.ui.tableWidget.verticalHeader().setDefaultSectionSize(40)
        self.ui.tableWidget.setColumnWidth(0, 80)  # 设置列宽
        self.ui.tableWidget.setColumnWidth(1, 200)
        self.ui.tableWidget.setColumnWidth(2, 80)
        self.ui.tableWidget.setColumnWidth(3, 150)
        self.ui.tableWidget.setColumnWidth(4, 90)
        self.ui.tableWidget.setColumnWidth(5, 250)
        # self.ui.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 表格铺满
        # self.ui.tableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        # self.ui.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 设置表格不可编辑
        self.ui.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置表格整行选中
        self.ui.tableWidget.verticalHeader().setVisible(False)  # 隐藏列标题
        self.ui.tableWidget.setAlternatingRowColors(True)  # 表格背景交替

        self.ui.doubleSpinBox.setValue(self.conf)
        self.ui.doubleSpinBox_2.setValue(self.iou)
        
        # 添加设置像素到厘米转换比例的SpinBox
        try:
            # 如果界面中有名为doubleSpinBox_3的控件，则使用它
            if hasattr(self.ui, 'doubleSpinBox_3'):
                self.ui.doubleSpinBox_3.setRange(0.1, 100.0)
                self.ui.doubleSpinBox_3.setSingleStep(0.1)
                self.ui.doubleSpinBox_3.setValue(Config.pixels_per_mm * 10)  # 转换为厘米单位
                self.ui.doubleSpinBox_3.valueChanged.connect(self.pixels_per_cm_change)
                
                # 添加说明标签，指明单位是厘米
                if hasattr(self.ui, 'label_pix_ratio'):
                    self.ui.label_pix_ratio.setText("像素/厘米比例:")
            else:
                # 如果没有现成的控件，添加一个标签说明
                from PyQt5.QtWidgets import QLabel, QDoubleSpinBox, QHBoxLayout, QWidget
                pixels_per_mm_layout = QHBoxLayout()
                pixels_per_mm_label = QLabel("像素/厘米比例:")
                self.pixels_per_mm_spinbox = QDoubleSpinBox()
                self.pixels_per_mm_spinbox.setRange(0.1, 100.0)
                self.pixels_per_mm_spinbox.setSingleStep(0.1)
                self.pixels_per_mm_spinbox.setValue(Config.pixels_per_mm * 10)  # 转换为厘米单位
                self.pixels_per_mm_spinbox.valueChanged.connect(self.pixels_per_cm_change)
                pixels_per_mm_layout.addWidget(pixels_per_mm_label)
                pixels_per_mm_layout.addWidget(self.pixels_per_mm_spinbox)
                pixels_per_mm_widget = QWidget()
                pixels_per_mm_widget.setLayout(pixels_per_mm_layout)
                
                # 将新控件添加到界面的适当位置
                if hasattr(self.ui, 'verticalLayout_3'):
                    self.ui.verticalLayout_3.addWidget(pixels_per_mm_widget)
        except Exception as e:
            print(f"设置像素/厘米控件时出错: {e}")

        # 更新CSV文件标题行，添加厘米单位的列
        self.csv_header = ['文件路径', '目标编号', '类别', '置信度', '坐标位置(像素)', '中心点(厘米)', '物体尺寸(厘米)', '归一化坐标']
        if not os.path.exists(Config.csv_save_path):
            with open(Config.csv_save_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_header)
                writer.writeheader()
        
        # 添加状态栏用于显示厘米坐标信息
        self.statusBar().showMessage("准备就绪 - 当前像素/厘米比例: " + str(Config.pixels_per_mm * 10) + "  (单位: 厘米)")

        # 设置默认图片
        default_img_path = "UIProgram/ui_imgs/11.jpg"
        if os.path.exists(default_img_path):
            import cv2
            from PyQt5.QtGui import QPixmap, QImage
            img = cv2.imread(default_img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.ui.label_show.setPixmap(pixmap.scaled(self.show_width, self.show_height, Qt.KeepAspectRatio))
            self.ui.label_show.setAlignment(Qt.AlignCenter)

    def conf_value_change(self):
        # 改变置信度值
        cur_conf = round(self.ui.doubleSpinBox.value(), 2)
        self.conf = cur_conf

    def iou_value_change(self):
        # 改变iou值
        cur_iou = round(self.ui.doubleSpinBox_2.value(), 2)
        self.iou = cur_iou
        
    def pixels_per_cm_change(self):
        # 改变像素到厘米的转换比例
        try:
            if hasattr(self.ui, 'doubleSpinBox_3'):
                Config.pixels_per_mm = round(self.ui.doubleSpinBox_3.value() / 10, 2)  # 从厘米转回毫米
            else:
                Config.pixels_per_mm = round(self.pixels_per_mm_spinbox.value() / 10, 2)  # 从厘米转回毫米
        except Exception as e:
            print(f"获取像素/厘米比例控件时出错: {e}")
            
        print(f"更新像素到厘米转换比例: {Config.pixels_per_mm * 10} 像素/厘米")
        # 更新状态栏显示
        self.statusBar().showMessage(f"当前像素/厘米比例: {Config.pixels_per_mm * 10} (单位: 厘米)")
        
        # 如果当前有图像显示，重新处理以更新坐标
        if hasattr(self, 'results') and self.results is not None:
            self.combox_change()  # 重新应用当前选择，更新显示

    def open_img(self):
        if self.cap:
            # 打开图片前关闭摄像头
            self.video_stop()
            self.is_camera_open = False
            self.ui.CapBtn.setText('打开摄像头')
            self.ui.VideoBtn.setText('打开视频')
            self.cap = None

        # 弹出的窗口名称：'打开图片'
        # 默认打开的目录：'./'
        # 只能打开*.jpg *.jpeg *.png *.bmp结尾的图片文件
        file_path, _ = QFileDialog.getOpenFileName(None, '打开图片', './', "Image files (*.jpg *.jpeg *.png *.bmp)")
        if not file_path:
            return

        self.ui.comboBox.setDisabled(False)
        self.org_path = file_path
        self.org_img = tools.img_cvread(self.org_path)  # 保存原始图像用于成熟度分析
        
        # 更新自适应比例因子
        try:
            import utils
            # 获取图像尺寸并更新比例因子
            img_height, img_width = self.org_img.shape[:2]
            utils.update_scale_factor(img_width, img_height)
            # 在状态栏显示当前比例信息
            self.statusBar().showMessage(f"图像尺寸: {img_width}x{img_height}, 自适应比例因子: {utils.adaptive_scale_factor:.4f}")
        except Exception as e:
            print(f"更新自适应比例因子时出错: {e}")

        # 目标检测
        t1 = time.time()
        self.results = self.model(self.org_path, conf=self.conf, iou=self.iou)[0]
        t2 = time.time()
        take_time_str = '{:.3f} s'.format(t2 - t1)
        self.ui.time_lb.setText(take_time_str)

        location_list = self.results.boxes.xyxy.tolist()
        self.location_list = [list(map(int, e)) for e in location_list]
        cls_list = self.results.boxes.cls.tolist()
        self.cls_list = [int(i) for i in cls_list]
        self.conf_list = self.results.boxes.conf.tolist()
        self.conf_list = ['%.2f %%' % (each*100) for each in self.conf_list]
        self.id_list = [i for i in range(len(self.location_list))]

        # 使用中文标签进行绘制
        # 替换检测结果的names字典，确保使用中文标签
        self.results.names = Config.CH_names
        
        # 获取检测结果图像 - 不显示置信度
        if self.ui.show_labels_and_conf.isChecked():
            now_img = self.results.plot(conf=False)
        else:
            now_img = self.results.plot(labels=False, conf=False)
            
        self.draw_img = now_img
        # 获取缩放后的图片尺寸
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img,(self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

        # 目标数目
        target_nums = len(self.cls_list)
        self.ui.label_nums.setText(str(target_nums))

        # 设置目标选择下拉框
        choose_list = ['全部']
        target_names = [Config.CH_names[id]+ '_'+ str(index) for index,id in enumerate(self.cls_list)]
        # object_list = sorted(set(self.cls_list))
        # for each in object_list:
        #     choose_list.append(Config.CH_names[each])
        choose_list = choose_list + target_names

        self.ui.comboBox.clear()
        self.ui.comboBox.addItems(choose_list)

        if target_nums >= 1:
            self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.ui.label_conf.setText(str(self.conf_list[0]))
        #   默认显示第一个目标框坐标
        #   设置坐标位置值
            self.ui.label_xmin.setText(str(self.location_list[0][0]))
            self.ui.label_ymin.setText(str(self.location_list[0][1]))
            self.ui.label_xmax.setText(str(self.location_list[0][2]))
            self.ui.label_ymax.setText(str(self.location_list[0][3]))
        else:
            self.ui.type_lb.setText('')
            self.ui.label_conf.setText('')
            self.ui.label_xmin.setText('')
            self.ui.label_ymin.setText('')
            self.ui.label_xmax.setText('')
            self.ui.label_ymax.setText('')

        # # 删除表格所有行
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, self.id_list, path=self.org_path)

    def detact_batch_imgs(self):
        if self.cap:
            # 打开图片前关闭摄像头
            self.video_stop()
            self.is_camera_open = False
            self.ui.CapBtn.setText('打开摄像头')
            self.ui.VideoBtn.setText('打开视频')
            self.cap = None
        directory = QFileDialog.getExistingDirectory(self,
                                                      "选取文件夹",
                                                      "./")  # 起始路径
        if not  directory:
            return
        self.ui.comboBox.setDisabled(False)
        self.org_path = directory
        img_suffix = ['jpg','png','jpeg','bmp']
        for file_name in os.listdir(directory):
            full_path = os.path.join(directory,file_name)
            if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                img_path = full_path
                self.org_img = tools.img_cvread(img_path)  # 保存原始图像用于成熟度分析
                
                # 更新自适应比例因子
                try:
                    import utils
                    img_height, img_width = self.org_img.shape[:2]
                    utils.update_scale_factor(img_width, img_height)
                    # 在状态栏显示当前比例信息
                    self.statusBar().showMessage(f"图像尺寸: {img_width}x{img_height}, 自适应比例因子: {utils.adaptive_scale_factor:.4f}")
                except Exception as e:
                    print(f"更新自适应比例因子时出错: {e}")
                
                # 目标检测
                t1 = time.time()
                self.results = self.model(img_path,conf=self.conf, iou=self.iou)[0]
                t2 = time.time()
                take_time_str = '{:.3f} s'.format(t2 - t1)
                self.ui.time_lb.setText(take_time_str)

                location_list = self.results.boxes.xyxy.tolist()
                self.location_list = [list(map(int, e)) for e in location_list]
                cls_list = self.results.boxes.cls.tolist()
                self.cls_list = [int(i) for i in cls_list]
                self.conf_list = self.results.boxes.conf.tolist()
                self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]
                self.id_list = [i for i in range(len(self.location_list))]

                # 使用中文标签进行绘制
                # 替换检测结果的names字典，确保使用中文标签
                self.results.names = Config.CH_names
                
                if self.ui.show_labels_and_conf.isChecked():
                    now_img = self.results.plot(conf=False)
                else:
                    now_img = self.results.plot(labels=False,conf=False)
                    
                self.draw_img = now_img
                # 获取缩放后的图片尺寸
                self.img_width, self.img_height = self.get_resize_size(now_img)
                resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
                pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
                self.ui.label_show.setPixmap(pix_img)
                self.ui.label_show.setAlignment(Qt.AlignCenter)

                # 目标数目
                target_nums = len(self.cls_list)
                self.ui.label_nums.setText(str(target_nums))

                # 设置目标选择下拉框
                choose_list = ['全部']
                target_names = [Config.CH_names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
                choose_list = choose_list + target_names

                self.ui.comboBox.clear()
                self.ui.comboBox.addItems(choose_list)

                if target_nums >= 1:
                    self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
                    self.ui.label_conf.setText(str(self.conf_list[0]))
                    #   默认显示第一个目标框坐标
                    #   设置坐标位置值
                    self.ui.label_xmin.setText(str(self.location_list[0][0]))
                    self.ui.label_ymin.setText(str(self.location_list[0][1]))
                    self.ui.label_xmax.setText(str(self.location_list[0][2]))
                    self.ui.label_ymax.setText(str(self.location_list[0][3]))
                else:
                    self.ui.type_lb.setText('')
                    self.ui.label_conf.setText('')
                    self.ui.label_xmin.setText('')
                    self.ui.label_ymin.setText('')
                    self.ui.label_xmax.setText('')
                    self.ui.label_ymax.setText('')

                # # 删除表格所有行
                # self.ui.tableWidget.setRowCount(0)
                # self.ui.tableWidget.clearContents()
                self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, self.id_list, path=img_path)

                self.ui.tableWidget.scrollToBottom()
                QApplication.processEvents()  #刷新页面

    def draw_rect_and_tabel(self, results, img):
        now_img = img.copy()
        location_list = results.boxes.xyxy.tolist()
        self.location_list = [list(map(int, e)) for e in location_list]
        cls_list = results.boxes.cls.tolist()
        self.cls_list = [int(i) for i in cls_list]
        self.conf_list = results.boxes.conf.tolist()
        self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

        for loacation, type_id, conf in zip(self.location_list, self.cls_list, self.conf_list):
            type_id = int(type_id)
            color = self.colors(int(type_id), True)
            # cv2.rectangle(now_img, (int(x1), int(y1)), (int(x2), int(y2)), colors(int(type_id), True), 3)
            now_img = tools.drawRectBox(now_img, loacation, Config.CH_names[type_id], self.fontC, color)

        # 获取缩放后的图片尺寸
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

        # 目标数目
        target_nums = len(self.cls_list)
        self.ui.label_nums.setText(str(target_nums))
        if target_nums >= 1:
            self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.ui.label_conf.setText(str(self.conf_list[0]))
            self.ui.label_xmin.setText(str(self.location_list[0][0]))
            self.ui.label_ymin.setText(str(self.location_list[0][1]))
            self.ui.label_xmax.setText(str(self.location_list[0][2]))
            self.ui.label_ymax.setText(str(self.location_list[0][3]))
        else:
            self.ui.type_lb.setText('')
            self.ui.label_conf.setText('')
            self.ui.label_xmin.setText('')
            self.ui.label_ymin.setText('')
            self.ui.label_xmax.setText('')
            self.ui.label_ymax.setText('')

        # 删除表格所有行
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=self.org_path)
        return now_img

    def combox_change(self):
        com_text = self.ui.comboBox.currentText()
        if com_text == '全部':
            cur_box = self.location_list
            if self.ui.show_labels_and_conf.isChecked():
                cur_img = self.results.plot(conf=False)
            else:
                cur_img = self.results.plot(labels=False, conf=False)
            self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.ui.label_conf.setText(str(self.conf_list[0]))
        else:
            index = int(com_text.split('_')[-1])
            cur_box = [self.location_list[index]]
            if self.ui.show_labels_and_conf.isChecked():
                cur_img = self.results[index].plot(conf=False)
            else:
                cur_img = self.results[index].plot(labels=False, conf=False)
            self.ui.type_lb.setText(Config.CH_names[self.cls_list[index]])
            self.ui.label_conf.setText(str(self.conf_list[index]))

        # 设置坐标位置值
        self.ui.label_xmin.setText(str(cur_box[0][0]))
        self.ui.label_ymin.setText(str(cur_box[0][1]))
        self.ui.label_xmax.setText(str(cur_box[0][2]))
        self.ui.label_ymax.setText(str(cur_box[0][3]))

        resize_cvimg = cv2.resize(cur_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.clear()
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

    def get_video_path(self):
        file_path, _ = QFileDialog.getOpenFileName(None, '打开视频', './', "Image files (*.avi *.mp4 *.wmv *.mkv)")
        if not file_path:
            return None
        self.org_path = file_path
        return file_path

    def video_start(self):
        # 删除表格所有行
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()

        # 清空下拉框
        self.ui.comboBox.clear()

        # 定时器开启，每隔一段时间，读取一帧
        self.timer_camera.start(1)
        self.timer_camera.timeout.connect(self.open_frame)

    def tabel_info_show(self, locations, clses, confs, target_ids, path=None):
        path = path
        if self.is_camera_open:
            path = 'Camera'

        # 获取当前显示的图像用于成熟度分析
        current_img = self.org_img if hasattr(self, 'org_img') and self.org_img is not None else None

        for location, cls, conf, target_id in zip(locations, clses, confs, target_ids):
            row_count = self.ui.tableWidget.rowCount()  # 返回当前行数(尾部)
            self.ui.tableWidget.insertRow(row_count)  # 尾部插入一行
            item_id = QTableWidgetItem(str(row_count+1))  # 序号
            item_id.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中
            item_path = QTableWidgetItem(str(path))  # 路径
            # item_path.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

            # 计算物理尺寸（厘米）和中心点坐标，使用自适应比例
            try:
                import utils
                width_cm, height_cm = utils.get_physical_dimensions(location, use_adaptive=True)
                center_x = (location[0] + location[2]) / 2
                center_y = (location[1] + location[3]) / 2
                center_x_cm = utils.pixels_to_cm(center_x, use_adaptive=True)
                center_y_cm = utils.pixels_to_cm(center_y, use_adaptive=True)
                
                # 如果有图像尺寸信息，也计算归一化坐标（0-1范围）
                if current_img is not None:
                    img_height, img_width = current_img.shape[:2]
                    normalized_coords = utils.get_normalized_coordinates(location, img_width, img_height)
                    normalized_text = f" 归一化坐标: ({normalized_coords[0]:.3f},{normalized_coords[1]:.3f},{normalized_coords[2]:.3f},{normalized_coords[3]:.3f})"
                else:
                    normalized_text = ""
            except Exception as e:
                print(f"计算物理尺寸时出错: {e}")
                # 如果导入失败，直接计算
                width_pixels = location[2] - location[0]
                height_pixels = location[3] - location[1]
                width_cm = width_pixels / Config.pixels_per_mm / 10
                height_cm = height_pixels / Config.pixels_per_mm / 10
                center_x = (location[0] + location[2]) / 2
                center_y = (location[1] + location[3]) / 2
                center_x_cm = center_x / Config.pixels_per_mm / 10
                center_y_cm = center_y / Config.pixels_per_mm / 10
                normalized_text = ""
                
            # 将物理尺寸和厘米坐标添加到位置信息中，去掉cm单位标记
            # 注意：将像素坐标单独放在一行，确保正则表达式能完整匹配
            location_text = f"{location}\n"
            
            # 根据类别不同显示不同的信息
            if Config.CH_names[cls] == '截点' or Config.CH_names[cls] == '坏花':
                # 对于截点和坏花，只显示坐标数字
                location_cm_text = f"坐标: ({center_x_cm:.2f}, {center_y_cm:.2f})"
                location_text = location_text + location_cm_text
            else:
                # 其他目标显示完整信息
                location_cm_text = f"中心点: ({center_x_cm:.2f}, {center_y_cm:.2f}), 尺寸: {width_cm:.2f}x{height_cm:.2f}"
                location_text = location_text + location_cm_text + normalized_text
                
            item_location = QTableWidgetItem(location_text) # 目标框位置
            # item_location.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            item_target_id = QTableWidgetItem(str(target_id))
            item_target_id.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            self.ui.tableWidget.setItem(row_count, 0, item_id)
            self.ui.tableWidget.setItem(row_count, 1, item_path)
            self.ui.tableWidget.setItem(row_count, 2, item_target_id)
            self.ui.tableWidget.setItem(row_count, 3, QTableWidgetItem(str(Config.CH_names[cls])))
            self.ui.tableWidget.setItem(row_count, 4, QTableWidgetItem(str(conf)))
            self.ui.tableWidget.setItem(row_count, 5, item_location)
        self.ui.tableWidget.scrollToBottom()

    def video_stop(self):
        self.cap.release()
        self.timer_camera.stop()
        # self.timer_info.stop()

    def open_frame(self):
        ret, now_img = self.cap.read()
        if ret:
            self.org_img = now_img.copy()  # 保存原始图像用于成熟度分析
            
            # 更新自适应比例因子
            try:
                import utils
                img_height, img_width = self.org_img.shape[:2]
                utils.update_scale_factor(img_width, img_height)
                # 在状态栏显示当前比例信息
                self.statusBar().showMessage(f"图像尺寸: {img_width}x{img_height}, 自适应比例因子: {utils.adaptive_scale_factor:.4f}")
            except Exception as e:
                print(f"更新自适应比例因子时出错: {e}")
            
            # 目标检测
            t1 = time.time()
            results = self.model(now_img,conf=self.conf, iou=self.iou)[0]
            t2 = time.time()
            take_time_str = '{:.3f} s'.format(t2 - t1)
            self.ui.time_lb.setText(take_time_str)

            location_list = results.boxes.xyxy.tolist()
            self.location_list = [list(map(int, e)) for e in location_list]
            cls_list = results.boxes.cls.tolist()
            self.cls_list = [int(i) for i in cls_list]
            self.conf_list = results.boxes.conf.tolist()
            self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]
            self.id_list = [i for i in range(len(self.location_list))]

            # 使用中文标签进行绘制
            # 替换检测结果的names字典，确保使用中文标签
            results.names = Config.CH_names
            
            # 在绘制检测结果时，也可以添加成熟度信息
            if self.ui.show_labels_and_conf.isChecked():
                now_img = results.plot(conf=False)
            else:
                now_img = results.plot(labels=False, conf=False)
                
            self.draw_img = now_img
            # 获取缩放后的图片尺寸
            self.img_width, self.img_height = self.get_resize_size(now_img)
            resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
            pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
            self.ui.label_show.setPixmap(pix_img)
            self.ui.label_show.setAlignment(Qt.AlignCenter)

            # 目标数目
            target_nums = len(self.cls_list)
            self.ui.label_nums.setText(str(target_nums))

            # 设置目标选择下拉框
            choose_list = ['全部']
            target_names = [Config.CH_names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
            choose_list = choose_list + target_names

            self.ui.comboBox.clear()
            self.ui.comboBox.addItems(choose_list)

            if target_nums >= 1:
                self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
                self.ui.label_conf.setText(str(self.conf_list[0]))
                #   默认显示第一个目标框坐标
                #   设置坐标位置值
                self.ui.label_xmin.setText(str(self.location_list[0][0]))
                self.ui.label_ymin.setText(str(self.location_list[0][1]))
                self.ui.label_xmax.setText(str(self.location_list[0][2]))
                self.ui.label_ymax.setText(str(self.location_list[0][3]))
            else:
                self.ui.type_lb.setText('')
                self.ui.label_conf.setText('')
                self.ui.label_xmin.setText('')
                self.ui.label_ymin.setText('')
                self.ui.label_xmax.setText('')
                self.ui.label_ymax.setText('')

            # 添加当前帧的检测信息到表格
            self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, self.id_list, path=self.org_path)

        else:
            self.cap.release()
            self.timer_camera.stop()
            self.ui.VideoBtn.setText('打开视频')

    def vedio_show(self):
        if self.is_camera_open:
            self.is_camera_open = False
            self.ui.CapBtn.setText('打开摄像头')
            if self.cap and self.cap.isOpened():
                self.cap.release()
                cv2.destroyAllWindows()

        if self.cap and self.cap.isOpened():
            # 关闭视频
            self.ui.VideoBtn.setText('打开视频')
            self.ui.label_show.setText('')
            self.cap.release()
            cv2.destroyAllWindows()
            self.ui.label_show.clear()
            return

        video_path = self.get_video_path()
        if not video_path:
            return None

        self.ui.VideoBtn.setText('关闭视频')
        self.cap = cv2.VideoCapture(video_path)
        self.video_start()
        self.ui.comboBox.setDisabled(True)

    def camera_show(self):
        self.is_camera_open = not self.is_camera_open
        if self.is_camera_open:
            self.ui.VideoBtn.setText('打开视频')
            self.ui.CapBtn.setText('关闭摄像头')
            self.cap = cv2.VideoCapture(0)
            self.video_start()
            self.ui.comboBox.setDisabled(True)
        else:
            self.ui.CapBtn.setText('打开摄像头')
            self.ui.label_show.setText('')
            if self.cap:
                self.cap.release()
                cv2.destroyAllWindows()
            self.ui.label_show.clear()

    def get_resize_size(self, img):
        _img = img.copy()
        img_height, img_width , depth= _img.shape
        ratio = img_width / img_height
        if ratio >= self.show_width / self.show_height:
            self.img_width = self.show_width
            self.img_height = int(self.img_width / ratio)
        else:
            self.img_height = self.show_height
            self.img_width = int(self.img_height * ratio)
        return self.img_width, self.img_height

    def save_detect_result(self):
        """
        保存检测结果
        """
        if self.cap is None and not self.org_path:
            QMessageBox.about(self, '提示', '当前没有可保存信息，请先打开图片或视频！')
            return

        if self.is_camera_open:
            QMessageBox.about(self, '提示', '摄像头视频无法保存!')
            return

        if self.cap:
            res = QMessageBox.information(self, '提示', '保存视频检测结果可能需要较长时间，请确认是否继续保存？',QMessageBox.Yes | QMessageBox.No ,  QMessageBox.Yes)
            if res == QMessageBox.Yes:
                self.video_stop()
                self.ui.VideoBtn.setText('打开视频')
                com_text = self.ui.comboBox.currentText()
                self.btn2Thread_object = btn2Thread(self.org_path, self.model, com_text,self.conf,self.iou, self.ui.show_labels_and_conf.isChecked())
                self.btn2Thread_object.start()
                self.btn2Thread_object.update_ui_signal.connect(self.update_process_bar)
            else:
                return
        else:
            # 创建csv文件
            if not os.path.exists(Config.csv_save_path):
                with open(Config.csv_save_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.csv_header)
                    writer.writeheader()
            if os.path.isfile(self.org_path):
                fileName = os.path.basename(self.org_path)
                name , end_name= fileName.rsplit(".",1)
                save_name = name + '_detect_result.' + end_name
                save_img_path = os.path.join(Config.save_path, save_name)
                # 保存图片
                cv2.imwrite(save_img_path, self.draw_img)
                # 保存csv文件
                self.save_to_csv(self.org_path, self.id_list, self.cls_list, self.conf_list, self.location_list)
                QMessageBox.about(self, '提示', '图片保存成功!\n文件路径:{}'.format(save_img_path))
            else:
                img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
                for file_name in os.listdir(self.org_path):
                    full_path = os.path.join(self.org_path, file_name)
                    if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                        name, end_name = file_name.rsplit(".",1)
                        save_name = name + '_detect_result.' + end_name
                        save_img_path = os.path.join(Config.save_path, save_name)
                        self.results = self.model(full_path,conf=self.conf, iou=self.iou)[0]
                        if self.ui.show_labels_and_conf.isChecked():
                            now_img = self.results.plot()
                        else:
                            now_img = self.results.plot(labels=False,conf=False)
                        # 保存图片
                        cv2.imwrite(save_img_path, now_img)

                        # 保存csv文件
                        location_list = self.results.boxes.xyxy.tolist()
                        self.location_list = [list(map(int, e)) for e in location_list]
                        cls_list = self.results.boxes.cls.tolist()
                        self.cls_list = [int(i) for i in cls_list]
                        self.conf_list = self.results.boxes.conf.tolist()
                        self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]
                        self.id_list = [i for i in range(len(self.location_list))]
                        self.save_to_csv(full_path, self.id_list, self.cls_list, self.conf_list, self.location_list)

                QMessageBox.about(self, '提示', '图片保存成功!\n文件路径:{}'.format(Config.save_path))


    def update_process_bar(self,cur_num, total):
        if cur_num == 1:
            self.progress_bar = ProgressBar(self)
            self.progress_bar.show()
        if cur_num >= total:
            self.progress_bar.close()
            QMessageBox.about(self, '提示', '视频保存成功!\n文件在{}目录下'.format(Config.save_path))
            return
        if self.progress_bar.isVisible() is False:
            # 点击取消保存时，终止进程
            self.btn2Thread_object.stop()
            return
        value = int(cur_num / total *100)
        self.progress_bar.setValue(cur_num, total, value)
        QApplication.processEvents()

    def on_cell_clicked(self, row, column):
        """
        鼠标点击表格触发，界面显示当前行内容
        """
        if self.cap:
            # 视频或摄像头不支持表格选择
            return
        img_path = self.ui.tableWidget.item(row, 1).text()
        target_id = int(self.ui.tableWidget.item(row, 2).text())
        now_type = self.ui.tableWidget.item(row, 3).text()
        conf_value = self.ui.tableWidget.item(row, 4).text()
        
        # 修复坐标解析问题
        location_text = self.ui.tableWidget.item(row, 5).text()
        try:
            # 尝试从文本中提取完整的坐标列表
            import re
            match = re.search(r'\[.*?\]', location_text)
            if match:
                location_str = match.group(0)
                location_value = eval(location_str)
            else:
                # 如果没找到坐标格式，尝试使用旧方式
                location_value = eval(location_text.split()[0])
        except Exception as e:
            print(f"解析坐标时出错: {e}")
            # 提供备用错误处理方案
            QMessageBox.warning(self, "坐标解析错误", f"无法解析坐标信息，请重新选择目标。错误: {e}")
            return

        self.ui.type_lb.setText(now_type)
        self.ui.label_conf.setText(str(conf_value))
        self.ui.label_xmin.setText(str(location_value[0]))
        self.ui.label_ymin.setText(str(location_value[1]))
        self.ui.label_xmax.setText(str(location_value[2]))
        self.ui.label_ymax.setText(str(location_value[3]))

        # 计算并显示厘米单位坐标，使用自适应比例
        try:
            import utils
            center_x = (location_value[0] + location_value[2]) / 2
            center_y = (location_value[1] + location_value[3]) / 2
            center_x_cm = utils.pixels_to_cm(center_x, use_adaptive=True)
            center_y_cm = utils.pixels_to_cm(center_y, use_adaptive=True)
            width_cm, height_cm = utils.get_physical_dimensions(location_value, use_adaptive=True)
            
            # 如果有图像尺寸信息，也计算归一化坐标（0-1范围）
            if hasattr(self, 'org_img') and self.org_img is not None:
                img_height, img_width = self.org_img.shape[:2]
                normalized_coords = utils.get_normalized_coordinates(location_value, img_width, img_height)
                normalized_text = f"归一化坐标: ({normalized_coords[0]:.3f},{normalized_coords[1]:.3f},{normalized_coords[2]:.3f},{normalized_coords[3]:.3f})"
            else:
                normalized_text = ""
                
        except Exception as e:
            print(f"计算物理尺寸时出错: {e}")
            center_x = (location_value[0] + location_value[2]) / 2
            center_y = (location_value[1] + location_value[3]) / 2
            center_x_cm = center_x / Config.pixels_per_mm / 10
            center_y_cm = center_y / Config.pixels_per_mm / 10
            width_cm = (location_value[2] - location_value[0]) / Config.pixels_per_mm / 10
            height_cm = (location_value[3] - location_value[1]) / Config.pixels_per_mm / 10
            normalized_text = ""
        
        # 在状态栏显示厘米坐标信息，根据类别不同显示不同内容
        is_cut_point = "截点" in now_type
        is_bad_flower = "坏花" in now_type
        
        if is_cut_point or is_bad_flower:
            # 对于截点和坏花只显示简单坐标
            status_text = f"像素坐标: {location_value}, 厘米坐标: ({center_x_cm:.2f}, {center_y_cm:.2f})"
        else:
            # 其他类别显示完整信息
            status_text = f"像素坐标: {location_value}, 厘米坐标: 中心点({center_x_cm:.2f}, {center_y_cm:.2f}), 尺寸: {width_cm:.2f}x{height_cm:.2f} (单位: 厘米)"
            if normalized_text:
                status_text += f", {normalized_text}"
        
        self.statusBar().showMessage(status_text)

        cur_commbox_text = now_type + '_' + str(target_id)

        now_img = tools.img_cvread(img_path)
        # 目标检测
        t1 = time.time()
        self.results = self.model(now_img, conf=self.conf, iou=self.iou)[0]
        t2 = time.time()
        take_time_str = '{:.3f} s'.format(t2 - t1)
        self.ui.time_lb.setText(take_time_str)

        location_list = self.results.boxes.xyxy.tolist()
        self.location_list = [list(map(int, e)) for e in location_list]
        cls_list = self.results.boxes.cls.tolist()
        self.cls_list = [int(i) for i in cls_list]
        self.conf_list = self.results.boxes.conf.tolist()
        self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

        # 设置目标选择下拉框
        choose_list = ['全部']
        target_names = [Config.CH_names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
        choose_list = choose_list + target_names
        self.ui.comboBox.clear()
        self.ui.comboBox.addItems(choose_list)
        self.ui.comboBox.setCurrentText(cur_commbox_text)

        # 绘制窗口图片
        if self.ui.show_labels_and_conf.isChecked():
            now_img = self.results[target_id].plot(conf=False)
        else:
            now_img = self.results[target_id].plot(labels=False, conf=False)
        self.draw_img = now_img
        # 获取缩放后的图片尺寸
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

        # 目标数目
        target_nums = len(self.cls_list)
        self.ui.label_nums.setText(str(target_nums))

    def save_to_csv(self, file_path, res_id_list, cls_list, confidence_list, location_list):
        """
        保存检测结果为csv文件格式
        """
        self.csv_header = ['文件路径', '目标编号', '类别', '置信度', '坐标位置(像素)', '中心点(厘米)', '物体尺寸(厘米)', '归一化坐标']
        if not os.path.exists(Config.csv_save_path):
            with open(Config.csv_save_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_header)
                writer.writeheader()
        else:
            # 检查现有文件的表头是否与新表头一致
            try:
                with open(Config.csv_save_path, 'r') as f:
                    first_line = f.readline().strip()
                    headers = [h.strip() for h in first_line.split(',')]
                    if headers != self.csv_header:
                        # 表头不一致，备份旧文件并创建新文件
                        import shutil
                        backup_path = Config.csv_save_path + '.bak'
                        shutil.copy2(Config.csv_save_path, backup_path)
                        with open(Config.csv_save_path, 'w', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=self.csv_header)
                            writer.writeheader()
            except Exception as e:
                print(f"检查CSV文件表头时出错: {e}")
                # 创建新文件
                with open(Config.csv_save_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.csv_header)
                    writer.writeheader()
        
        # 获取当前图像以计算归一化坐标
        current_img = None
        if hasattr(self, 'org_img') and self.org_img is not None:
            current_img = self.org_img
        
        # 写入数据
        with open(Config.csv_save_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_header)
            res_list = []
            for recognition_id, cls, confidence, location in zip(res_id_list, cls_list, confidence_list, location_list):
                # 检查是否为截点或坏花
                is_cut_point = Config.CH_names[cls] == '截点'
                is_bad_flower = Config.CH_names[cls] == '坏花'
                
                # 计算物理尺寸和中心点坐标，使用自适应比例
                try:
                    import utils
                    width_cm, height_cm = utils.get_physical_dimensions(location, use_adaptive=True)
                    center_x = (location[0] + location[2]) / 2
                    center_y = (location[1] + location[3]) / 2
                    center_x_cm = utils.pixels_to_cm(center_x, use_adaptive=True)
                    center_y_cm = utils.pixels_to_cm(center_y, use_adaptive=True)
                    
                    # 如果有图像，计算归一化坐标
                    normalized_coords = ""
                    if current_img is not None:
                        img_height, img_width = current_img.shape[:2]
                        norm_coords = utils.get_normalized_coordinates(location, img_width, img_height)
                        normalized_coords = f"({norm_coords[0]:.3f},{norm_coords[1]:.3f},{norm_coords[2]:.3f},{norm_coords[3]:.3f})"
                except Exception as e:
                    print(f"计算物理尺寸时出错: {e}")
                    width_cm = (location[2] - location[0]) / Config.pixels_per_mm / 10
                    height_cm = (location[3] - location[1]) / Config.pixels_per_mm / 10
                    center_x = (location[0] + location[2]) / 2
                    center_y = (location[1] + location[3]) / 2
                    center_x_cm = center_x / Config.pixels_per_mm / 10
                    center_y_cm = center_y / Config.pixels_per_mm / 10
                    normalized_coords = ""
                
                # 根据类别不同，使用不同的CSV记录格式
                if is_cut_point or is_bad_flower:
                    # 对于截点和坏花只保存简单坐标
                    data = {
                        '文件路径': file_path,
                        '目标编号': recognition_id,
                        '类别': Config.CH_names[cls],
                        '置信度': confidence,
                        '坐标位置(像素)': location,
                        '中心点(厘米)': f"({center_x_cm:.2f}, {center_y_cm:.2f})",
                        '物体尺寸(厘米)': "",  # 不保存尺寸
                        '归一化坐标': ""  # 不保存归一化坐标
                    }
                else:
                    # 其他类别保存完整信息
                    data = {
                        '文件路径': file_path,
                        '目标编号': recognition_id,
                        '类别': Config.CH_names[cls],
                        '置信度': confidence,
                        '坐标位置(像素)': location,
                        '中心点(厘米)': f"({center_x_cm:.2f}, {center_y_cm:.2f})",
                        '物体尺寸(厘米)': f"{width_cm:.2f}x{height_cm:.2f}",
                        '归一化坐标': normalized_coords
                    }
                
                res_list.append(data)
            writer.writerows(res_list)

    def open_train_module(self):
        """打开草莓截点训练模块"""
        try:
            # 检查训练模块文件是否存在
            if not os.path.exists("strawberry_jiedian_main.py"):
                QMessageBox.critical(self, "错误", "训练模块文件不存在: strawberry_jiedian_main.py")
                return
                
            # 运行训练模块
            QMessageBox.information(self, "提示", "正在启动模型训练模块，请稍候...")
            subprocess.Popen([sys.executable, "strawberry_jiedian_main.py"])
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动训练模块时出错: {str(e)}")


class btn2Thread(QThread):
    """
    进行检测后的视频保存
    """
    # 声明一个信号
    update_ui_signal = pyqtSignal(int,int)

    def __init__(self, path, model, com_text,conf,iou,show_label_and_conf):
        super(btn2Thread, self).__init__()
        self.org_path = path
        self.model = model
        self.com_text = com_text
        self.conf = conf
        self.iou = iou
        self.show_label_and_conf = show_label_and_conf
        # 用于绘制不同颜色矩形框
        self.colors = tools.Colors()
        self.is_running = True  # 标志位，表示线程是否正在运行

    def run(self):
        # VideoCapture方法是cv2库提供的读取视频方法
        cap = cv2.VideoCapture(self.org_path)
        # 设置需要保存视频的格式"xvid"
        # 该参数是MPEG-4编码类型，文件名后缀为.avi
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # 设置视频帧频
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 设置视频大小
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # VideoWriter方法是cv2库提供的保存视频方法
        # 按照设置的格式来out输出
        fileName = os.path.basename(self.org_path)
        name, end_name = fileName.split('.')
        save_name = name + '_detect_result.avi'
        save_video_path = os.path.join(Config.save_path, save_name)
        out = cv2.VideoWriter(save_video_path, fourcc, fps, size)

        prop = cv2.CAP_PROP_FRAME_COUNT
        total = int(cap.get(prop))
        print("[INFO] 视频总帧数：{}".format(total))
        cur_num = 0

        # 确定视频打开并循环读取
        while (cap.isOpened() and self.is_running):
            cur_num += 1
            print('当前第{}帧，总帧数{}'.format(cur_num, total))
            # 逐帧读取，ret返回布尔值
            # 参数ret为True 或者False,代表有没有读取到图片
            # frame表示截取到一帧的图片
            ret, frame = cap.read()
            if ret == True:
                # 检测
                results = self.model(frame,conf=self.conf,iou=self.iou)[0]
                
                # 使用中文标签进行绘制
                # 替换检测结果的names字典，确保使用中文标签
                results.names = Config.CH_names
                
                if self.show_label_and_conf:
                    frame = results.plot(conf=False)
                else:
                    frame = results.plot(labels=False, conf=False)
                    
                # 添加成熟度标签
                location_list = results.boxes.xyxy.tolist()
                location_list = [list(map(int, e)) for e in location_list]
                cls_list = results.boxes.cls.tolist()
                cls_list = [int(i) for i in cls_list]
                
                # 为果子添加成熟度标签
                for location, cls in zip(location_list, cls_list):
                    if Config.CH_names[cls] == '果子':
                        try:
                            ripeness_type, ripeness_ratio = tools.analyze_ripeness(frame, location)
                            # 在图像上添加成熟度标签，不显示百分比
                            label_text = f"{ripeness_type}"
                            
                            # 定义标签位置 - 在目标框上方
                            text_position = (location[0], location[1] - 30)  # 位置调高一些，给中文腾出空间
                            
                            # 根据成熟度设置颜色和字体大小
                            if "成熟" in ripeness_type:
                                color = (0, 255, 0)  # 绿色表示成熟
                                text_size = 20
                            else:
                                color = (0, 0, 255)  # 红色表示未成熟
                                text_size = 30  # 增大未成熟果子的字体
                            
                            # 为未成熟果子添加白色描边以增强可见性
                            if "未成熟" in ripeness_type:
                                # 先用白色背景增强对比度
                                outline_color = (255, 255, 255)
                                offset = 2  # 描边宽度
                                for dx, dy in [(-offset, -offset), (-offset, 0), (-offset, offset), 
                                             (0, -offset), (0, offset), 
                                             (offset, -offset), (offset, 0), (offset, offset)]:
                                    outline_pos = (text_position[0] + dx, text_position[1] + dy)
                                    frame = tools.cv2AddChineseText(img=frame, 
                                                                  text=label_text, 
                                                                  position=outline_pos, 
                                                                  textColor=outline_color, 
                                                                  textSize=text_size)
                            
                            # 使用cv2AddChineseText函数显示中文
                            frame = tools.cv2AddChineseText(img=frame, 
                                                           text=label_text, 
                                                           position=text_position, 
                                                           textColor=color, 
                                                           textSize=text_size)
                        except Exception as e:
                            print(f"视频保存时添加成熟度标签出错: {e}")
                
                out.write(frame)
                self.update_ui_signal.emit(cur_num, total)
            else:
                break
        # 释放资源
        cap.release()
        out.release()

    def stop(self):
        self.is_running = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
