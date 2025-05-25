import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
import time
from pathlib import Path
from ultralytics import YOLO
import yaml
from custom_loss import filter_jiedian_boxes

class StrawberryJiedianDetectGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("草莓截点关联检测工具")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # 创建主框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        self.control_frame = ttk.LabelFrame(self.main_frame, text="检测控制")
        self.control_frame.pack(side=tk.LEFT, fill="y", padx=5, pady=5)
        
        # 模型文件
        ttk.Label(self.control_frame, text="模型文件:").grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)
        self.model_var = tk.StringVar()
        self.model_var.set("runs/detect/strawberry_jiedian_train/weights/best.pt")
        model_entry = ttk.Entry(self.control_frame, textvariable=self.model_var, width=30)
        model_entry.grid(column=0, row=1, padx=10, pady=5, sticky=tk.W)
        ttk.Button(self.control_frame, text="浏览...", command=self.browse_model).grid(column=1, row=1, padx=5, pady=5)
        
        # 输入源选择
        ttk.Label(self.control_frame, text="输入源:").grid(column=0, row=2, padx=10, pady=10, sticky=tk.W)
        self.source_type_var = tk.StringVar()
        self.source_type_var.set("图像")
        source_types = ["图像", "视频", "摄像头"]
        self.source_type_combo = ttk.Combobox(self.control_frame, textvariable=self.source_type_var, values=source_types, state="readonly", width=10)
        self.source_type_combo.grid(column=0, row=3, padx=10, pady=5, sticky=tk.W)
        self.source_type_combo.bind("<<ComboboxSelected>>", self.update_source_ui)
        
        # 源文件路径
        ttk.Label(self.control_frame, text="源文件:").grid(column=0, row=4, padx=10, pady=10, sticky=tk.W)
        self.source_var = tk.StringVar()
        self.source_entry = ttk.Entry(self.control_frame, textvariable=self.source_var, width=30)
        self.source_entry.grid(column=0, row=5, padx=10, pady=5, sticky=tk.W)
        self.browse_btn = ttk.Button(self.control_frame, text="浏览...", command=self.browse_source)
        self.browse_btn.grid(column=1, row=5, padx=5, pady=5)
        
        # 摄像头设备ID
        ttk.Label(self.control_frame, text="摄像头ID:").grid(column=0, row=6, padx=10, pady=10, sticky=tk.W)
        self.camera_id_var = tk.StringVar()
        self.camera_id_var.set("0")
        self.camera_id_entry = ttk.Entry(self.control_frame, textvariable=self.camera_id_var, width=5)
        self.camera_id_entry.grid(column=0, row=7, padx=10, pady=5, sticky=tk.W)
        self.camera_id_entry.config(state="disabled")
        
        # 检测参数
        param_frame = ttk.LabelFrame(self.control_frame, text="检测参数")
        param_frame.grid(column=0, row=8, columnspan=2, padx=10, pady=10, sticky="we")
        
        # 置信度阈值
        ttk.Label(param_frame, text="置信度阈值:").grid(column=0, row=0, padx=10, pady=5, sticky=tk.W)
        self.conf_var = tk.StringVar()
        self.conf_var.set("0.25")
        ttk.Entry(param_frame, textvariable=self.conf_var, width=5).grid(column=1, row=0, padx=5, pady=5, sticky=tk.W)
        
        # IoU阈值
        ttk.Label(param_frame, text="IoU阈值:").grid(column=0, row=1, padx=10, pady=5, sticky=tk.W)
        self.iou_var = tk.StringVar()
        self.iou_var.set("0.45")
        ttk.Entry(param_frame, textvariable=self.iou_var, width=5).grid(column=1, row=1, padx=5, pady=5, sticky=tk.W)
        
        # 图像尺寸
        ttk.Label(param_frame, text="图像尺寸:").grid(column=0, row=2, padx=10, pady=5, sticky=tk.W)
        self.imgsz_var = tk.StringVar()
        self.imgsz_var.set("640")
        ttk.Entry(param_frame, textvariable=self.imgsz_var, width=5).grid(column=1, row=2, padx=5, pady=5, sticky=tk.W)
        
        # 截点关联设置
        jiedian_frame = ttk.LabelFrame(self.control_frame, text="截点关联设置")
        jiedian_frame.grid(column=0, row=9, columnspan=2, padx=10, pady=10, sticky="we")
        
        # 使用自定义截点处理
        self.use_custom_filter_var = tk.BooleanVar()
        self.use_custom_filter_var.set(True)
        ttk.Checkbutton(jiedian_frame, text="截点关联处理", 
                         variable=self.use_custom_filter_var).grid(column=0, row=0, columnspan=2, padx=10, pady=5, sticky=tk.W)
        
        # 坏果类别ID
        ttk.Label(jiedian_frame, text="坏果类别ID:").grid(column=0, row=1, padx=10, pady=5, sticky=tk.W)
        self.bad_fruit_id_var = tk.StringVar()
        self.bad_fruit_id_var.set("1")
        ttk.Entry(jiedian_frame, textvariable=self.bad_fruit_id_var, width=5).grid(column=1, row=1, padx=5, pady=5, sticky=tk.W)
        
        # 截点类别ID
        ttk.Label(jiedian_frame, text="截点类别ID:").grid(column=0, row=2, padx=10, pady=5, sticky=tk.W)
        self.jiedian_id_var = tk.StringVar()
        self.jiedian_id_var.set("4")
        ttk.Entry(jiedian_frame, textvariable=self.jiedian_id_var, width=5).grid(column=1, row=2, padx=5, pady=5, sticky=tk.W)
        
        # IoU关联阈值
        ttk.Label(jiedian_frame, text="关联IoU阈值:").grid(column=0, row=3, padx=10, pady=5, sticky=tk.W)
        self.relation_iou_var = tk.StringVar()
        self.relation_iou_var.set("0.5")
        ttk.Entry(jiedian_frame, textvariable=self.relation_iou_var, width=5).grid(column=1, row=3, padx=5, pady=5, sticky=tk.W)
        
        # 截点置信度比例
        ttk.Label(jiedian_frame, text="置信度比例:").grid(column=0, row=4, padx=10, pady=5, sticky=tk.W)
        self.conf_ratio_var = tk.StringVar()
        self.conf_ratio_var.set("0.8")
        ttk.Entry(jiedian_frame, textvariable=self.conf_ratio_var, width=5).grid(column=1, row=4, padx=5, pady=5, sticky=tk.W)
        
        # 输出设置
        output_frame = ttk.LabelFrame(self.control_frame, text="输出设置")
        output_frame.grid(column=0, row=10, columnspan=2, padx=10, pady=10, sticky="we")
        
        # 保存结果
        self.save_results_var = tk.BooleanVar()
        self.save_results_var.set(True)
        ttk.Checkbutton(output_frame, text="保存检测结果", 
                         variable=self.save_results_var).grid(column=0, row=0, columnspan=2, padx=10, pady=5, sticky=tk.W)
        
        # 结果保存目录
        ttk.Label(output_frame, text="保存目录:").grid(column=0, row=1, padx=10, pady=5, sticky=tk.W)
        self.save_dir_var = tk.StringVar()
        self.save_dir_var.set("Results/jiedian_detection")
        ttk.Entry(output_frame, textvariable=self.save_dir_var, width=20).grid(column=0, row=2, padx=10, pady=5, sticky=tk.W)
        ttk.Button(output_frame, text="浏览...", command=self.browse_save_dir).grid(column=1, row=2, padx=5, pady=5)
        
        # 输出文件名前缀
        ttk.Label(output_frame, text="文件名前缀:").grid(column=0, row=3, padx=10, pady=5, sticky=tk.W)
        self.save_prefix_var = tk.StringVar()
        self.save_prefix_var.set("jiedian_result")
        ttk.Entry(output_frame, textvariable=self.save_prefix_var, width=15).grid(column=0, row=4, padx=10, pady=5, sticky=tk.W)
        
        # 显示类别名称
        self.show_names_var = tk.BooleanVar()
        self.show_names_var.set(True)
        ttk.Checkbutton(output_frame, text="显示类别名称", 
                         variable=self.show_names_var).grid(column=0, row=5, padx=10, pady=5, sticky=tk.W)
        
        # 显示置信度
        self.show_conf_var = tk.BooleanVar()
        self.show_conf_var.set(True)
        ttk.Checkbutton(output_frame, text="显示置信度", 
                         variable=self.show_conf_var).grid(column=1, row=5, padx=5, pady=5, sticky=tk.W)
        
        # 开始/停止检测按钮
        self.detect_btn = ttk.Button(self.control_frame, text="开始检测", command=self.start_detection)
        self.detect_btn.grid(column=0, row=11, padx=10, pady=15, sticky=tk.W)
        
        self.stop_btn = ttk.Button(self.control_frame, text="停止检测", command=self.stop_detection, state="disabled")
        self.stop_btn.grid(column=1, row=11, padx=5, pady=15, sticky=tk.W)
        
        # 右侧图像显示区域
        self.image_frame = ttk.LabelFrame(self.main_frame, text="检测结果")
        self.image_frame.pack(side=tk.RIGHT, fill="both", expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.image_frame, bg="gray")
        self.canvas.pack(fill="both", expand=True)
        
        # 底部状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 检测状态变量
        self.detecting = False
        self.cap = None
        self.model = None
        self.class_names = None
        
        # 初始化UI
        self.update_source_ui()
    
    def update_source_ui(self, event=None):
        """根据选择的输入源类型更新UI"""
        source_type = self.source_type_var.get()
        
        if source_type == "摄像头":
            self.source_entry.config(state="disabled")
            self.browse_btn.config(state="disabled")
            self.camera_id_entry.config(state="normal")
        else:
            self.source_entry.config(state="normal")
            self.browse_btn.config(state="normal")
            self.camera_id_entry.config(state="disabled")
    
    def browse_model(self):
        """浏览模型文件"""
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch 模型", "*.pt"), ("所有文件", "*.*")])
        if file_path:
            self.model_var.set(file_path)
    
    def browse_source(self):
        """浏览源文件"""
        source_type = self.source_type_var.get()
        
        if source_type == "图像":
            file_path = filedialog.askopenfilename(filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), 
                ("所有文件", "*.*")
            ])
        elif source_type == "视频":
            file_path = filedialog.askopenfilename(filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv"), 
                ("所有文件", "*.*")
            ])
        else:
            return
        
        if file_path:
            self.source_var.set(file_path)
    
    def browse_save_dir(self):
        """浏览保存目录"""
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.save_dir_var.set(dir_path)
    
    def load_model(self):
        """加载模型"""
        model_path = self.model_var.get()
        if not os.path.exists(model_path):
            messagebox.showerror("错误", f"模型文件不存在: {model_path}")
            return False
        
        try:
            self.status_var.set("正在加载模型...")
            self.root.update_idletasks()
            
            # 加载模型
            self.model = YOLO(model_path)
            
            # 获取类别名称
            yaml_file = self.model.names if hasattr(self.model, 'names') else None
            if isinstance(yaml_file, dict):
                self.class_names = yaml_file
            else:
                # 尝试从训练配置中获取类别名称
                try:
                    model_dir = os.path.dirname(os.path.dirname(model_path))
                    data_yaml = os.path.join(model_dir, "args.yaml")
                    if os.path.exists(data_yaml):
                        with open(data_yaml, 'r') as f:
                            args = yaml.safe_load(f)
                            if 'data' in args:
                                data_file = args['data']
                                with open(data_file, 'r') as f2:
                                    data_cfg = yaml.safe_load(f2)
                                    if 'names' in data_cfg:
                                        self.class_names = data_cfg['names']
                except Exception as e:
                    print(f"无法加载类别名称: {e}")
                    self.class_names = {i: f"class_{i}" for i in range(10)}
            
            self.status_var.set(f"模型加载成功: {os.path.basename(model_path)}")
            return True
            
        except Exception as e:
            messagebox.showerror("错误", f"加载模型时出错: {str(e)}")
            self.status_var.set("模型加载失败")
            return False
    
    def start_detection(self):
        """开始检测过程"""
        if self.detecting:
            return
        
        # 加载模型
        if self.model is None:
            if not self.load_model():
                return
        
        # 检查输入源
        source_type = self.source_type_var.get()
        
        if source_type == "图像":
            source = self.source_var.get()
            if not os.path.exists(source):
                messagebox.showerror("错误", f"图像文件不存在: {source}")
                return
        elif source_type == "视频":
            source = self.source_var.get()
            if not os.path.exists(source):
                messagebox.showerror("错误", f"视频文件不存在: {source}")
                return
        else:  # 摄像头
            try:
                camera_id = int(self.camera_id_var.get())
                source = camera_id
            except ValueError:
                messagebox.showerror("错误", "摄像头ID必须是整数")
                return
        
        # 准备检测参数
        self.detect_params = {
            'conf': float(self.conf_var.get()),
            'iou': float(self.iou_var.get()),
            'imgsz': int(self.imgsz_var.get()),
            'use_custom_filter': self.use_custom_filter_var.get(),
            'bad_fruit_id': int(self.bad_fruit_id_var.get()),
            'jiedian_id': int(self.jiedian_id_var.get()),
            'relation_iou': float(self.relation_iou_var.get()),
            'conf_ratio': float(self.conf_ratio_var.get()),
            'save_results': self.save_results_var.get(),
            'save_dir': self.save_dir_var.get(),
            'save_prefix': self.save_prefix_var.get(),
            'show_names': self.show_names_var.get(),
            'show_conf': self.show_conf_var.get()
        }
        
        # 创建保存目录
        if self.detect_params['save_results']:
            os.makedirs(self.detect_params['save_dir'], exist_ok=True)
        
        # 更新界面状态
        self.detecting = True
        self.detect_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        
        # 开始检测过程
        if source_type == "图像":
            self.detect_image(source)
        else:  # 视频或摄像头
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                messagebox.showerror("错误", f"无法打开视频源: {source}")
                self.reset_ui()
                return
            
            # 开始视频检测线程
            self.detect_thread = self.root.after(100, self.detect_video_frame)
    
    def detect_image(self, image_path):
        """检测单个图像"""
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                messagebox.showerror("错误", f"无法读取图像: {image_path}")
                self.reset_ui()
                return
            
            # 更新状态
            self.status_var.set(f"正在检测: {os.path.basename(image_path)}")
            self.root.update_idletasks()
            
            # 执行检测
            start_time = time.time()
            results = self.model(img, conf=self.detect_params['conf'], iou=self.detect_params['iou'], imgsz=self.detect_params['imgsz'])
            
            # 使用自定义截点关联处理
            if self.detect_params['use_custom_filter']:
                prediction = results[0].boxes.data
                filtered_results = filter_jiedian_boxes(
                    [prediction],
                    bad_fruit_id=self.detect_params['bad_fruit_id'],
                    jiedian_id=self.detect_params['jiedian_id'],
                    iou_threshold=self.detect_params['relation_iou'],
                    conf_ratio=self.detect_params['conf_ratio']
                )
                # 替换原始结果中的预测
                results[0].boxes.data = filtered_results[0]
            
            # 绘制结果
            result_img = self.draw_results(img, results[0])
            
            # 显示结果
            self.display_image(result_img)
            
            # 保存结果
            if self.detect_params['save_results']:
                save_name = f"{self.detect_params['save_prefix']}_{int(time.time())}.jpg"
                save_path = os.path.join(self.detect_params['save_dir'], save_name)
                cv2.imwrite(save_path, result_img)
                self.status_var.set(f"检测完成，结果已保存至: {save_path}")
            else:
                self.status_var.set(f"检测完成，耗时: {time.time() - start_time:.2f}秒")
        
        except Exception as e:
            messagebox.showerror("错误", f"检测过程中出错: {str(e)}")
            self.status_var.set("检测失败")
        
        finally:
            # 重置界面状态
            self.reset_ui()
    
    def detect_video_frame(self):
        """检测视频帧"""
        if not self.detecting or self.cap is None:
            return
        
        try:
            # 读取一帧
            ret, frame = self.cap.read()
            if not ret:
                # 视频结束
                self.status_var.set("视频检测完成")
                self.reset_ui()
                return
            
            # 执行检测
            results = self.model(frame, conf=self.detect_params['conf'], iou=self.detect_params['iou'], imgsz=self.detect_params['imgsz'])
            
            # 使用自定义截点关联处理
            if self.detect_params['use_custom_filter']:
                prediction = results[0].boxes.data
                filtered_results = filter_jiedian_boxes(
                    [prediction],
                    bad_fruit_id=self.detect_params['bad_fruit_id'],
                    jiedian_id=self.detect_params['jiedian_id'],
                    iou_threshold=self.detect_params['relation_iou'],
                    conf_ratio=self.detect_params['conf_ratio']
                )
                # 替换原始结果中的预测
                results[0].boxes.data = filtered_results[0]
            
            # 绘制结果
            result_img = self.draw_results(frame, results[0])
            
            # 显示结果
            self.display_image(result_img)
            
            # 保存结果
            if self.detect_params['save_results']:
                save_name = f"{self.detect_params['save_prefix']}_{int(time.time())}.jpg"
                save_path = os.path.join(self.detect_params['save_dir'], save_name)
                cv2.imwrite(save_path, result_img)
            
            # 设置下一帧检测
            self.detect_thread = self.root.after(10, self.detect_video_frame)
            
        except Exception as e:
            print(f"视频检测错误: {e}")
            self.status_var.set(f"检测错误: {str(e)}")
            self.reset_ui()
    
    def stop_detection(self):
        """停止检测过程"""
        if not self.detecting:
            return
        
        self.detecting = False
        
        # 停止视频检测线程
        if hasattr(self, 'detect_thread'):
            self.root.after_cancel(self.detect_thread)
        
        # 释放视频捕获资源
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.reset_ui()
        self.status_var.set("检测已停止")
    
    def reset_ui(self):
        """重置界面状态"""
        self.detecting = False
        self.detect_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
    
    def draw_results(self, img, result):
        """绘制检测结果"""
        # 创建一个副本，避免修改原始图像
        img_copy = img.copy()
        
        if len(result) == 0:
            return img_copy
        
        # 获取检测框
        boxes = result.boxes.data
        
        # 定义颜色映射 - 修改颜色方案
        colors = {
            0: (0, 0, 255),    # 好果: 红色
            1: (147, 20, 255), # 坏果: 粉色(已修改)
            2: (255, 0, 0),    # 好花: 蓝色
            3: (255, 0, 255),  # 坏花: 紫色
            4: (147, 20, 255), # 截点: 粉色
            5: (128, 128, 128) # 无效: 灰色
        }
        
        # 绘制每个检测框
        for box in boxes:
            # 获取坐标和类别信息
            x1, y1, x2, y2 = map(int, box[:4])
            conf = float(box[4])
            cls_id = int(box[5])
            
            # 获取颜色
            color = colors.get(cls_id, (255, 255, 0))
            
            # 绘制框
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # 准备标签文本
            label_parts = []
            if self.detect_params['show_names'] and self.class_names is not None:
                class_name = self.class_names.get(cls_id, f"class_{cls_id}")
                label_parts.append(class_name)
            
            if self.detect_params['show_conf']:
                label_parts.append(f"{conf:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # 计算标签大小
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                
                # 绘制标签背景 - 修改为方框下方
                cv2.rectangle(img_copy, (x1, y2), (x1+text_width, y2+text_height+5), color, -1)
                
                # 绘制标签文本 - 修改为方框下方
                cv2.putText(img_copy, label, (x1, y2+text_height+2), font, font_scale, (255, 255, 255), thickness)
        
        return img_copy
    
    def display_image(self, img):
        """在界面上显示图像"""
        # 获取画布尺寸
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # 如果画布尚未完全初始化，使用默认值
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 640
            canvas_height = 480
        
        # 调整图像大小以适应画布
        h, w = img.shape[:2]
        scale = min(canvas_width/w, canvas_height/h)
        new_size = (int(w*scale), int(h*scale))
        
        # 调整图像大小
        resized_img = cv2.resize(img, new_size)
        
        # 转换颜色空间从BGR到RGB
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        # 转换为PhotoImage
        pil_img = Image.fromarray(rgb_img)
        self.photo = ImageTk.PhotoImage(image=pil_img)
        
        # 更新画布
        self.canvas.config(width=new_size[0], height=new_size[1])
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

if __name__ == "__main__":
    # 创建主窗口
    root = tk.Tk()
    app = StrawberryJiedianDetectGUI(root)
    root.mainloop() 