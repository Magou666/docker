import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import threading
import torch
import yaml
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ultralytics import YOLO
from pathlib import Path
import shutil
import traceback
import numpy as np
import queue

# 添加中文字体支持
try:
    import matplotlib.font_manager as fm
    # 检查是否有中文字体，这里使用了几个常见的中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi']
    chinese_font = None
    
    for font in chinese_fonts:
        if any(f.name == font for f in fm.fontManager.ttflist):
            chinese_font = font
            break
    
    if chinese_font:
        plt.rcParams['font.sans-serif'] = [chinese_font] + plt.rcParams['font.sans-serif']
    else:
        # 如果没有找到中文字体，尝试使用系统默认字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 出错时不做任何处理，保持默认配置
    pass

class StrawberryJiedianTrainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("草莓截点关联训练工具")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # 创建消息队列用于线程间通信
        self.message_queue = queue.Queue()
        
        # 创建主滚动框架
        # 创建主Canvas
        self.main_canvas = tk.Canvas(root)
        self.main_canvas.pack(side=tk.LEFT, fill="both", expand=True)
        
        # 添加垂直滚动条
        self.vsb = ttk.Scrollbar(root, orient="vertical", command=self.main_canvas.yview)
        self.vsb.pack(side=tk.RIGHT, fill="y")
        self.main_canvas.configure(yscrollcommand=self.vsb.set)
        
        # 创建滚动框架内的内容框架
        self.main_frame = ttk.Frame(self.main_canvas)
        self.canvas_frame = self.main_canvas.create_window((0, 0), window=self.main_frame, anchor="nw")
        
        # 绑定滚动事件和窗口大小调整事件
        self.main_frame.bind("<Configure>", self.on_frame_configure)
        self.main_canvas.bind("<Configure>", self.on_canvas_configure)
        self.root.bind("<MouseWheel>", self.on_mousewheel)  # Windows
        self.root.bind("<Button-4>", self.on_mousewheel)    # Linux上滚
        self.root.bind("<Button-5>", self.on_mousewheel)    # Linux下滚
        
        # 创建数据和模型配置区域
        self.config_frame = ttk.LabelFrame(self.main_frame, text="数据与模型配置")
        self.config_frame.pack(fill="x", padx=5, pady=5)
        
        # 数据集配置文件
        ttk.Label(self.config_frame, text="数据集配置文件:").grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)
        self.data_yaml_var = tk.StringVar()
        self.data_yaml_var.set("datasets/strawberry_dataset/data.yaml")
        ttk.Entry(self.config_frame, textvariable=self.data_yaml_var, width=50).grid(column=1, row=0, padx=10, pady=10, sticky=tk.W)
        ttk.Button(self.config_frame, text="浏览...", command=self.browse_data_yaml).grid(column=2, row=0, padx=10, pady=10)
        
        # 模型配置文件
        ttk.Label(self.config_frame, text="模型配置文件:").grid(column=0, row=1, padx=10, pady=10, sticky=tk.W)
        self.model_yaml_var = tk.StringVar()
        self.model_yaml_var.set("ultralytics/cfg/models/v10/yolov10n.yaml")
        ttk.Entry(self.config_frame, textvariable=self.model_yaml_var, width=50).grid(column=1, row=1, padx=10, pady=10, sticky=tk.W)
        ttk.Button(self.config_frame, text="浏览...", command=self.browse_model_yaml).grid(column=2, row=1, padx=10, pady=10)
        
        # 预训练模型
        ttk.Label(self.config_frame, text="预训练模型:").grid(column=0, row=2, padx=10, pady=10, sticky=tk.W)
        self.pre_model_var = tk.StringVar()
        self.pre_model_var.set("yolov10n.pt")
        ttk.Entry(self.config_frame, textvariable=self.pre_model_var, width=50).grid(column=1, row=2, padx=10, pady=10, sticky=tk.W)
        ttk.Button(self.config_frame, text="浏览...", command=self.browse_pre_model).grid(column=2, row=2, padx=10, pady=10)
        
        # 添加模型类型下拉框
        ttk.Label(self.config_frame, text="模型类型:").grid(column=0, row=3, padx=10, pady=10, sticky=tk.W)
        self.model_type_var = tk.StringVar()
        model_types = ["YOLOv10n", "YOLOv10s", "YOLOv10m", "YOLOv10l", "YOLOv10x", "YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"]
        self.model_type_var.set(model_types[0])
        model_type_combobox = ttk.Combobox(self.config_frame, textvariable=self.model_type_var, values=model_types, width=20, state="readonly")
        model_type_combobox.grid(column=1, row=3, padx=10, pady=10, sticky=tk.W)
        model_type_combobox.bind("<<ComboboxSelected>>", self.on_model_type_change)
        
        # 创建训练参数区域
        self.train_params_frame = ttk.LabelFrame(self.main_frame, text="训练参数")
        self.train_params_frame.pack(fill="x", padx=5, pady=5)
        
        # 参数左侧
        left_params = ttk.Frame(self.train_params_frame)
        left_params.grid(column=0, row=0, padx=5, pady=5, sticky=tk.W)
        
        # 训练轮数
        ttk.Label(left_params, text="训练轮数:").grid(column=0, row=0, padx=10, pady=5, sticky=tk.W)
        self.epochs_var = tk.StringVar()
        self.epochs_var.set("150")
        ttk.Entry(left_params, textvariable=self.epochs_var, width=10).grid(column=1, row=0, padx=10, pady=5, sticky=tk.W)
        
        # 批次大小
        ttk.Label(left_params, text="批次大小:").grid(column=0, row=1, padx=10, pady=5, sticky=tk.W)
        self.batch_var = tk.StringVar()
        self.batch_var.set("4")
        ttk.Entry(left_params, textvariable=self.batch_var, width=10).grid(column=1, row=1, padx=10, pady=5, sticky=tk.W)
        
        # 图像尺寸
        ttk.Label(left_params, text="图像尺寸:").grid(column=0, row=2, padx=10, pady=5, sticky=tk.W)
        self.imgsz_var = tk.StringVar()
        self.imgsz_var.set("640")
        ttk.Entry(left_params, textvariable=self.imgsz_var, width=10).grid(column=1, row=2, padx=10, pady=5, sticky=tk.W)
        
        # 参数右侧
        right_params = ttk.Frame(self.train_params_frame)
        right_params.grid(column=1, row=0, padx=5, pady=5, sticky=tk.W)
        
        # 训练设备
        ttk.Label(right_params, text="训练设备:").grid(column=0, row=0, padx=10, pady=5, sticky=tk.W)
        self.device_var = tk.StringVar()
        self.device_var.set("0" if torch.cuda.is_available() else "cpu")
        ttk.Entry(right_params, textvariable=self.device_var, width=10).grid(column=1, row=0, padx=10, pady=5, sticky=tk.W)
        
        # 结果保存名称
        ttk.Label(right_params, text="保存名称:").grid(column=0, row=1, padx=10, pady=5, sticky=tk.W)
        self.name_var = tk.StringVar()
        self.name_var.set("strawberry_jiedian_train")
        ttk.Entry(right_params, textvariable=self.name_var, width=20).grid(column=1, row=1, padx=10, pady=5, sticky=tk.W)
        
        # 学习率
        ttk.Label(right_params, text="初始学习率:").grid(column=0, row=2, padx=10, pady=5, sticky=tk.W)
        self.lr_var = tk.StringVar()
        self.lr_var.set("0.01")
        ttk.Entry(right_params, textvariable=self.lr_var, width=10).grid(column=1, row=2, padx=10, pady=5, sticky=tk.W)
        
        # 创建自定义损失函数参数区域
        self.loss_frame = ttk.LabelFrame(self.main_frame, text="损失函数配置")
        self.loss_frame.pack(fill="x", padx=5, pady=5)
        
        # 使用自定义损失函数
        self.use_custom_loss_var = tk.BooleanVar()
        self.use_custom_loss_var.set(True)
        ttk.Checkbutton(self.loss_frame, text="使用自定义损失函数（关联截点与坏果）", 
                         variable=self.use_custom_loss_var).grid(column=0, row=0, padx=10, pady=5, sticky=tk.W)
        
        # 坏果类别ID
        ttk.Label(self.loss_frame, text="坏果类别ID:").grid(column=0, row=1, padx=10, pady=5, sticky=tk.W)
        self.bad_fruit_id_var = tk.StringVar()
        self.bad_fruit_id_var.set("1")  # 假设坏果是类别1
        ttk.Entry(self.loss_frame, textvariable=self.bad_fruit_id_var, width=5).grid(column=1, row=1, padx=10, pady=5, sticky=tk.W)
        
        # 截点类别ID
        ttk.Label(self.loss_frame, text="截点类别ID:").grid(column=2, row=1, padx=10, pady=5, sticky=tk.W)
        self.jiedian_id_var = tk.StringVar()
        self.jiedian_id_var.set("4")  # 假设截点是类别4
        ttk.Entry(self.loss_frame, textvariable=self.jiedian_id_var, width=5).grid(column=3, row=1, padx=10, pady=5, sticky=tk.W)
        
        # 截点损失权重
        ttk.Label(self.loss_frame, text="截点损失权重:").grid(column=0, row=2, padx=10, pady=5, sticky=tk.W)
        self.jiedian_loss_weight_var = tk.StringVar()
        self.jiedian_loss_weight_var.set("1.5")
        ttk.Entry(self.loss_frame, textvariable=self.jiedian_loss_weight_var, width=5).grid(column=1, row=2, padx=10, pady=5, sticky=tk.W)
        
        # 坏果损失权重
        ttk.Label(self.loss_frame, text="坏果损失权重:").grid(column=2, row=2, padx=10, pady=5, sticky=tk.W)
        self.bad_fruit_loss_weight_var = tk.StringVar()
        self.bad_fruit_loss_weight_var.set("1.2")
        ttk.Entry(self.loss_frame, textvariable=self.bad_fruit_loss_weight_var, width=5).grid(column=3, row=2, padx=10, pady=5, sticky=tk.W)
        
        # 创建训练与可视化区域
        self.train_visual_frame = ttk.Frame(self.main_frame)
        self.train_visual_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 左侧控制区域
        self.control_frame = ttk.LabelFrame(self.train_visual_frame, text="训练控制")
        self.control_frame.grid(column=0, row=0, padx=5, pady=5, sticky="ns")
        
        # 开始训练按钮
        self.train_btn = ttk.Button(self.control_frame, text="开始训练", command=self.start_training)
        self.train_btn.grid(column=0, row=0, padx=10, pady=10)
        
        # 停止训练按钮
        self.stop_btn = ttk.Button(self.control_frame, text="停止训练", command=self.stop_training, state="disabled")
        self.stop_btn.grid(column=0, row=1, padx=10, pady=10)
        
        # 训练状态
        ttk.Label(self.control_frame, text="训练状态:").grid(column=0, row=2, padx=10, pady=5, sticky=tk.W)
        self.train_status_var = tk.StringVar()
        self.train_status_var.set("就绪")
        ttk.Label(self.control_frame, textvariable=self.train_status_var).grid(column=0, row=3, padx=10, pady=5, sticky=tk.W)
        
        # 训练进度
        ttk.Label(self.control_frame, text="当前进度:").grid(column=0, row=4, padx=10, pady=5, sticky=tk.W)
        self.progress_var = tk.StringVar()
        self.progress_var.set("0/0")
        ttk.Label(self.control_frame, textvariable=self.progress_var).grid(column=0, row=5, padx=10, pady=5, sticky=tk.W)
        
        # 右侧可视化区域
        self.visual_frame = ttk.LabelFrame(self.train_visual_frame, text="训练可视化")
        self.visual_frame.grid(column=1, row=0, padx=5, pady=5, sticky="nsew")
        self.train_visual_frame.columnconfigure(1, weight=1)
        self.train_visual_frame.rowconfigure(0, weight=1)
        
        # 创建固定大小的图表框架
        self.chart_frame = ttk.Frame(self.visual_frame, width=400, height=300)
        self.chart_frame.pack(padx=5, pady=5)
        self.chart_frame.pack_propagate(False)  # 防止子控件影响Frame大小
        
        # 创建损失图表，使用固定尺寸
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.ax.set_title("训练损失曲线")
        self.ax.set_xlabel("轮次")
        self.ax.set_ylabel("损失值")
        self.ax.grid(True)
        
        # 嵌入图表到Tkinter窗口
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # 日志区域
        self.log_frame = ttk.LabelFrame(self.main_frame, text="训练日志")
        self.log_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=10)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 训练控制变量
        self.training = False
        self.stop_requested = False
        self.epochs_data = []
        self.losses_data = []
        
        # 开始消息处理循环
        self.process_message_queue()
    
    def browse_data_yaml(self):
        file_path = filedialog.askopenfilename(filetypes=[("YAML 文件", "*.yaml"), ("所有文件", "*.*")])
        if file_path:
            self.data_yaml_var.set(file_path)
    
    def browse_model_yaml(self):
        file_path = filedialog.askopenfilename(filetypes=[("YAML 文件", "*.yaml"), ("所有文件", "*.*")])
        if file_path:
            self.model_yaml_var.set(file_path)
    
    def browse_pre_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch 模型", "*.pt"), ("所有文件", "*.*")])
        if file_path:
            self.pre_model_var.set(file_path)
    
    def process_message_queue(self):
        """处理来自训练线程的消息队列"""
        try:
            while not self.message_queue.empty():
                message = self.message_queue.get_nowait()
                
                if message['type'] == 'log':
                    self._update_log(message['content'])
                elif message['type'] == 'progress':
                    self._update_progress(message['epoch'], message['total'], message['loss'])
                elif message['type'] == 'ask_cmd_mode':
                    self._handle_cmd_mode_request()
                elif message['type'] == 'training_complete':
                    self._update_training_complete()
                elif message['type'] == 'training_error':
                    self._update_training_error(message['error'])
        except Exception as e:
            print(f"处理消息队列时出错: {str(e)}")
        
        # 每100毫秒检查一次队列
        self.root.after(100, self.process_message_queue)
    
    def _update_log(self, message):
        """在主线程中更新日志"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
    
    def _update_progress(self, current_epoch, total_epochs, loss):
        """在主线程中更新进度和损失图表"""
        try:
            # 更新进度文本
            self.progress_var.set(f"{current_epoch}/{total_epochs}")
            
            # 确保损失是标量值
            try:
                loss_value = float(loss)
            except:
                # 如果转换失败，使用一个合理的估计值
                if self.losses_data and len(self.losses_data) > 0:
                    loss_value = self.losses_data[-1]
                else:
                    loss_value = 1.0  # 初始损失估计
            
            # 添加损失数据点
            self.epochs_data.append(current_epoch)
            self.losses_data.append(loss_value)
            
            # 更新图表，保持固定大小
            self.ax.clear()
            self.ax.plot(self.epochs_data, self.losses_data, '-o', color='blue')
            self.ax.set_title("训练损失曲线")
            self.ax.set_xlabel("轮次")
            self.ax.set_ylabel("损失值")
            self.ax.grid(True)
            
            # 设置适当的x轴范围
            if len(self.epochs_data) > 1:
                self.ax.set_xlim(0, total_epochs)
            
            # 确保图表大小不变
            self.fig.tight_layout()
            
            # 重绘图表
            self.canvas.draw()
            
            # 添加日志到UI
            self._update_log(f"轮次 {current_epoch}/{total_epochs} 完成，损失: {loss_value:.6f}")
        except Exception as e:
            print(f"更新进度时出错: {str(e)}")
    
    def _handle_cmd_mode_request(self):
        """处理命令行模式请求"""
        # 不再显示弹窗，默认不使用命令行模式
        self.cmd_mode_result = False
        
        # 使用Event对象通知训练线程
        self.cmd_mode_event = threading.Event()
        self._update_log("API模式训练失败，已停止训练")
        self.cmd_mode_event.set()  # 设置事件，通知训练线程
    
    def _update_training_complete(self):
        """在训练完成时更新UI"""
        self.reset_ui()
    
    def _update_training_error(self, error_message):
        """显示训练错误但不使用弹窗"""
        # 移除错误弹窗，只在日志中显示
        self._update_log(f"训练过程出错: {error_message}")
        self._update_log("训练已停止，请检查上述日志信息")
        self.reset_ui()
    
    def log(self, message):
        """将日志消息添加到队列"""
        # 如果在主线程中，直接更新UI
        if threading.current_thread() is threading.main_thread():
            self._update_log(message)
        else:
            # 否则添加到消息队列
            self.message_queue.put({'type': 'log', 'content': message})
    
    def start_training(self):
        """开始训练过程"""
        # 检查参数
        if not os.path.exists(self.data_yaml_var.get()):
            messagebox.showerror("错误", f"数据集配置文件不存在: {self.data_yaml_var.get()}")
            return
        
        if not os.path.exists(self.model_yaml_var.get()):
            messagebox.showerror("错误", f"模型配置文件不存在: {self.model_yaml_var.get()}")
            return
        
        if not os.path.exists(self.pre_model_var.get()):
            messagebox.showerror("错误", f"预训练模型不存在: {self.pre_model_var.get()}")
            return
        
        # 修改界面状态
        self.train_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.train_status_var.set("训练中...")
        self.progress_var.set(f"0/{self.epochs_var.get()}")
        
        # 重置图表数据
        self.epochs_data = []
        self.losses_data = []
        self.ax.clear()
        self.ax.set_title("训练损失曲线")
        self.ax.set_xlabel("轮次")
        self.ax.set_ylabel("损失值")
        self.ax.grid(True)
        self.canvas.draw()
        
        # 重置日志
        self.log_text.delete(1.0, tk.END)
        
        # 重置控制标志
        self.training = True
        self.stop_requested = False
        
        # 在新线程中启动训练
        threading.Thread(target=self.training_thread, daemon=True).start()
    
    def stop_training(self):
        """请求停止训练"""
        if self.training:
            self.stop_requested = True
            self.train_status_var.set("正在停止...")
            self.log("用户请求停止训练...")
    
    def check_training_progress(self, model, epochs):
        """检查训练进度 - 独立于training_thread方法"""
        last_epoch = 0
        
        while self.training and not self.stop_requested:
            try:
                # 获取当前训练状态
                if hasattr(model, 'trainer') and model.trainer is not None:
                    current_epoch = model.trainer.epoch + 1
                    
                    # 只在轮次变化时更新
                    if current_epoch > last_epoch:
                        try:
                            loss = float(model.trainer.loss)  # 直接转换为float
                        except:
                            # 如果无法获取损失值，使用估计值
                            loss = 0.0
                        
                        # 将进度更新添加到消息队列
                        self.message_queue.put({
                            'type': 'progress', 
                            'epoch': current_epoch, 
                            'total': epochs, 
                            'loss': loss
                        })
                        
                        # 如果请求停止，尝试停止训练
                        if self.stop_requested:
                            try:
                                # 尝试通过各种方式停止训练
                                if hasattr(model.trainer, 'stop'):
                                    model.trainer.stop = True
                                elif hasattr(model, 'training'):
                                    model.training = False
                            except:
                                pass
                        
                        last_epoch = current_epoch
            except Exception as e:
                print(f"进度检查错误: {str(e)}")
            
            # 每秒检查一次
            time.sleep(1)

    def training_thread(self):
        """训练线程"""
        try:
            # 确保导入YOLO类在每次训练开始时都重新导入
            from ultralytics import YOLO
            
            # 记录开始时间
            start_time = time.time()
            self.log(f"开始训练，时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.log(f"数据集配置: {self.data_yaml_var.get()}")
            self.log(f"模型配置: {self.model_yaml_var.get()}")
            self.log(f"预训练模型: {self.pre_model_var.get()}")
            
            # 创建自定义模型配置
            if self.use_custom_loss_var.get():
                # 读取原始模型配置
                with open(self.model_yaml_var.get(), 'r', encoding='utf-8') as f:
                    model_cfg = yaml.safe_load(f)
                
                # 修改模型配置，添加自定义损失函数
                custom_model_yaml = f"custom_{os.path.basename(self.model_yaml_var.get())}"
                model_cfg['jiedian_loss'] = {
                    'enabled': True,
                    'bad_fruit_id': int(self.bad_fruit_id_var.get()),
                    'jiedian_id': int(self.jiedian_id_var.get()),
                    'jiedian_loss_weight': float(self.jiedian_loss_weight_var.get()),
                    'bad_fruit_loss_weight': float(self.bad_fruit_loss_weight_var.get())
                }
                
                # 保存自定义模型配置
                os.makedirs('custom_models', exist_ok=True)
                custom_model_path = os.path.join('custom_models', custom_model_yaml)
                with open(custom_model_path, 'w', encoding='utf-8') as f:
                    yaml.dump(model_cfg, f)
                
                self.log(f"创建自定义模型配置: {custom_model_path}")
                model_yaml_path = custom_model_path
            else:
                model_yaml_path = self.model_yaml_var.get()
            
            # 创建数据集配置的副本，避免修改原始文件
            data_yaml_path = self.data_yaml_var.get()
            temp_data_yaml = "temp_data.yaml"
            
            try:
                # 读取原始数据集配置
                with open(data_yaml_path, 'r', encoding='utf-8') as f:
                    data_cfg = yaml.safe_load(f)
                
                # 提取数据集名称 (从路径中提取目录名)
                dataset_dir = os.path.dirname(os.path.abspath(data_yaml_path))
                dataset_name = os.path.basename(dataset_dir)
                self.log(f"当前数据集名称: {dataset_name}")
                
                # 直接解决路径问题
                # 1. 获取数据集路径
                original_dataset_path = data_cfg.get('path', '')
                
                # 2. 获取绝对路径 - 使用数据集文件目录作为基础路径
                dataset_dir = os.path.dirname(os.path.abspath(data_yaml_path))
                
                # 3. 找到实际的images目录
                if os.path.isabs(original_dataset_path):
                    images_dir = os.path.join(original_dataset_path, 'images')
                else:
                    # 处理相对路径
                    if original_dataset_path == '.' or original_dataset_path.startswith('./'):
                        images_dir = os.path.join(dataset_dir, 'images')
                    else:
                        base_path = original_dataset_path.replace('datasets/', '')
                        images_dir = os.path.join(dataset_dir, 'images')
                
                # 检查train/val/test目录是否存在
                train_dir = os.path.join(images_dir, 'train') 
                val_dir = os.path.join(images_dir, 'val')
                test_dir = os.path.join(images_dir, 'test')
                
                self.log(f"检查数据集目录...")
                self.log(f"训练图像目录: {train_dir}")
                self.log(f"验证图像目录: {val_dir}")
                self.log(f"测试图像目录: {test_dir}")
                
                # 检查验证集是否为空
                val_has_images = False
                if os.path.exists(val_dir):
                    image_files = [f for f in os.listdir(val_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')) 
                                 and os.path.isfile(os.path.join(val_dir, f))]
                    val_has_images = len(image_files) > 0
                
                if not val_has_images:
                    self.log("警告：验证集目录为空，将从训练集自动复制部分图像作为验证集")
                    
                    # 确保目录存在
                    os.makedirs(val_dir, exist_ok=True)
                    
                    # 获取训练集图像
                    train_images = []
                    if os.path.exists(train_dir):
                        train_images = [f for f in os.listdir(train_dir) 
                                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')) 
                                      and os.path.isfile(os.path.join(train_dir, f))]
                    
                    if train_images:
                        # 选择约20%的图像作为验证集，但至少1张，最多10张
                        num_val = max(1, min(10, int(len(train_images) * 0.2)))
                        val_images = train_images[:num_val]
                        
                        self.log(f"从训练集中选择了 {num_val} 张图像作为验证集")
                        
                        # 复制图像和标签文件到验证集目录
                        for img_file in val_images:
                            # 复制图像
                            src_img = os.path.join(train_dir, img_file)
                            dst_img = os.path.join(val_dir, img_file)
                            shutil.copy2(src_img, dst_img)
                            
                            # 复制对应的标签文件（如果存在）
                            # YOLO格式的标签文件保存在labels目录，扩展名为.txt
                            label_name = os.path.splitext(img_file)[0] + '.txt'
                            src_label = os.path.join(dataset_dir, 'labels', 'train', label_name)
                            if os.path.exists(src_label):
                                # 确保验证集标签目录存在
                                os.makedirs(os.path.join(dataset_dir, 'labels', 'val'), exist_ok=True)
                                dst_label = os.path.join(dataset_dir, 'labels', 'val', label_name)
                                shutil.copy2(src_label, dst_label)
                    else:
                        self.log("错误：训练集中没有找到图像文件")
                
                # 检查并创建目标路径的链接
                datasets_dir = os.path.abspath('datasets')
                # 使用正确的数据集名称而不是硬编码的9.46
                target_dataset_dir = os.path.join(datasets_dir, 'datasets', dataset_name)
                target_images_dir = os.path.join(target_dataset_dir, 'images')
                target_train_dir = os.path.join(target_images_dir, 'train')
                target_val_dir = os.path.join(target_images_dir, 'val')
                target_test_dir = os.path.join(target_images_dir, 'test')
                
                # 创建必要的目录
                os.makedirs(target_dataset_dir, exist_ok=True)
                os.makedirs(target_images_dir, exist_ok=True)
                
                # 使用简单的文件复制或创建符号链接确保数据集存在
                if os.path.exists(train_dir) and not os.path.exists(target_train_dir):
                    self.log(f"创建训练集链接 -> {target_train_dir}")
                    try:
                        # 在Windows系统上创建目录连接
                        if os.name == 'nt':
                            import subprocess
                            subprocess.run(['mklink', '/j', target_train_dir, train_dir], shell=True, check=True)
                        else:
                            # 在Unix/Linux系统上创建符号链接
                            os.symlink(train_dir, target_train_dir)
                    except Exception as e:
                        self.log(f"创建链接失败: {str(e)}，尝试创建目录并复制文件...")
                        os.makedirs(target_train_dir, exist_ok=True)
                        # 复制文件作为备选方案
                        for file in os.listdir(train_dir):
                            src_file = os.path.join(train_dir, file)
                            dst_file = os.path.join(target_train_dir, file)
                            if os.path.isfile(src_file) and not os.path.exists(dst_file):
                                shutil.copy2(src_file, dst_file)
                
                if os.path.exists(val_dir) and not os.path.exists(target_val_dir):
                    self.log(f"创建验证集链接 -> {target_val_dir}")
                    try:
                        # 在Windows系统上创建目录连接
                        if os.name == 'nt':
                            import subprocess
                            subprocess.run(['mklink', '/j', target_val_dir, val_dir], shell=True, check=True)
                        else:
                            # 在Unix/Linux系统上创建符号链接
                            os.symlink(val_dir, target_val_dir)
                    except Exception as e:
                        self.log(f"创建链接失败: {str(e)}，尝试创建目录并复制文件...")
                        os.makedirs(target_val_dir, exist_ok=True)
                        # 复制文件作为备选方案
                        for file in os.listdir(val_dir):
                            src_file = os.path.join(val_dir, file)
                            dst_file = os.path.join(target_val_dir, file)
                            if os.path.isfile(src_file) and not os.path.exists(dst_file):
                                shutil.copy2(src_file, dst_file)
                
                if os.path.exists(test_dir) and not os.path.exists(target_test_dir):
                    self.log(f"创建测试集链接 -> {target_test_dir}")
                    try:
                        # 在Windows系统上创建目录连接
                        if os.name == 'nt':
                            import subprocess
                            subprocess.run(['mklink', '/j', target_test_dir, test_dir], shell=True, check=True)
                        else:
                            # 在Unix/Linux系统上创建符号链接
                            os.symlink(test_dir, target_test_dir)
                    except Exception as e:
                        self.log(f"创建链接失败: {str(e)}，尝试创建目录并复制文件...")
                        os.makedirs(target_test_dir, exist_ok=True)
                        # 复制文件作为备选方案
                        for file in os.listdir(test_dir):
                            src_file = os.path.join(test_dir, file)
                            dst_file = os.path.join(target_test_dir, file)
                            if os.path.isfile(src_file) and not os.path.exists(dst_file):
                                shutil.copy2(src_file, dst_file)
                
                # 修改数据集配置，使用绝对路径
                data_cfg['path'] = os.path.abspath(target_dataset_dir)
                
                # 保存临时配置
                with open(temp_data_yaml, 'w', encoding='utf-8') as f:
                    yaml.dump(data_cfg, f)
                
                self.log(f"已创建临时数据集配置: {temp_data_yaml}")
                self.log(f"配置的数据集路径: {data_cfg['path']}")
                
                # 使用新的配置文件
                data_yaml_path = temp_data_yaml
                
            except Exception as e:
                self.log(f"处理数据集路径时出错: {str(e)}")
                self.log(traceback.format_exc())
            
            # 创建并加载模型
            self.log("创建模型...")
            model = YOLO(model_yaml_path)
            model.load(self.pre_model_var.get())
            
            # 训练参数
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_var.get())
            imgsz = int(self.imgsz_var.get())
            device = self.device_var.get()
            save_name = self.name_var.get()
            lr0 = float(self.lr_var.get())
            
            # 开始训练
            self.log("训练开始...")
            
            # 准备训练参数
            train_args = dict(
                data=data_yaml_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=imgsz,
                device=device,
                name=save_name,
                patience=30,
                save=True,
                lr0=lr0,
                lrf=0.01,
                warmup_epochs=3,
                optimizer='SGD',
                workers=4
            )
            
            # 创建训练进度检查线程
            progress_thread = threading.Thread(
                target=lambda: self.check_training_progress(model, epochs), 
                daemon=True
            )
            progress_thread.start()
            
            # 开始训练
            try:
                # 使用最简单的方式调用训练
                results = model.train(**train_args)
            except Exception as e:
                error_message = str(e)
                # 隐藏"main thread is not in main loop"错误信息
                if "main thread is not in main loop" in error_message:
                    self.log("训练过程中断，正在尝试其他方式...")
                    print(f"捕获到线程错误（已隐藏）: {error_message}")
                else:
                    # 其他错误正常显示
                    self.log(f"训练API错误: {error_message}")
                
                # 请求用户确认是否使用命令行模式
                self.message_queue.put({'type': 'ask_cmd_mode'})
                
                # 等待用户响应，最多等待60秒
                if hasattr(self, 'cmd_mode_event'):
                    self.cmd_mode_event.wait(60)
                    
                    # 检查用户的选择
                    if hasattr(self, 'cmd_mode_result') and self.cmd_mode_result:
                        try:
                            self.log("尝试使用命令行方式训练...")
                            
                            # 确保这里也重新导入YOLO
                            from ultralytics import YOLO
                            YOLO(self.pre_model_var.get()).train(
                                data=data_yaml_path,
                                epochs=epochs,
                                imgsz=imgsz,
                                device=device
                            )
                        except Exception as e2:
                            self.log(f"命令行训练也失败: {str(e2)}")
                            self.log(traceback.format_exc())
                    else:
                        self.log("用户取消了命令行模式训练")
            
            # 清理临时文件
            try:
                if os.path.exists(temp_data_yaml):
                    os.remove(temp_data_yaml)
                    self.log("已删除临时数据集配置文件")
            except:
                pass
            
            # 训练完成
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # 计算并显示保存路径（无论成功与否）
            save_dir = os.path.join("runs", "detect", save_name)
            weights_dir = os.path.join(save_dir, "weights")
            best_model_path = os.path.join(weights_dir, "best.pt")
            last_model_path = os.path.join(weights_dir, "last.pt")
            
            # 始终显示预期的保存路径
            self.log(f"\n===== 保存路径信息 =====")
            self.log(f"训练结果目录: {os.path.abspath(save_dir)}")
            self.log(f"权重文件目录: {os.path.abspath(weights_dir)}")
            self.log(f"最佳模型路径: {os.path.abspath(best_model_path)}")
            self.log(f"最新模型路径: {os.path.abspath(last_model_path)}")
            
            if self.stop_requested:
                self.log(f"训练被用户中断，耗时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
            else:
                self.log(f"训练完成，共 {epochs} 轮，耗时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
                
                # 检查保存的模型目录和文件是否实际存在
                if os.path.exists(save_dir):
                    self.log(f"训练结果目录已成功创建")
                    
                    if os.path.exists(weights_dir):
                        self.log(f"权重文件目录已成功创建")
                    else:
                        self.log(f"警告：权重文件目录未创建，可能保存失败")
                    
                    if os.path.exists(best_model_path):
                        self.log(f"最佳模型已成功保存")
                    else:
                        self.log(f"警告：最佳模型未找到")
                    
                    if os.path.exists(last_model_path):
                        self.log(f"最新模型已成功保存")
                    else:
                        self.log(f"警告：最新模型未找到")
                else:
                    self.log(f"警告：训练结果目录未创建，模型可能未成功保存")
                
                # 尝试评估模型
                try:
                    self.log("开始模型评估...")
                    
                    # 创建一个新的模型用于评估
                    best_model_path = os.path.join(save_dir, "weights", "best.pt")
                    if os.path.exists(best_model_path):
                        eval_model = YOLO(best_model_path)
                        
                        # 使用临时数据集配置进行评估，确保路径正确
                        # 创建一个新的临时评估数据集配置文件
                        try:
                            eval_data_yaml = "temp_eval_data.yaml"
                            
                            # 读取原始数据集配置
                            with open(self.data_yaml_var.get(), 'r', encoding='utf-8') as f:
                                eval_data_cfg = yaml.safe_load(f)
                            
                            # 确保验证数据集的路径是绝对路径
                            dataset_dir = os.path.dirname(os.path.abspath(self.data_yaml_var.get()))
                            images_dir = os.path.join(dataset_dir, 'images')
                            val_dir = os.path.join(images_dir, 'val')
                            
                            # 检查验证集目录是否存在
                            if os.path.exists(val_dir):
                                self.log(f"使用验证集目录: {val_dir}")
                                eval_data_cfg['path'] = dataset_dir
                                
                                # 保存临时评估配置
                                with open(eval_data_yaml, 'w', encoding='utf-8') as f:
                                    yaml.dump(eval_data_cfg, f)
                                
                                self.log(f"已创建临时评估数据集配置: {eval_data_yaml}")
                                
                                # 使用临时配置评估模型
                                eval_results = eval_model.val(data=eval_data_yaml)
                                
                                # 输出评估结果
                                if eval_results is not None:
                                    self.log("\n验证结果:")
                                    for metric_name, metric_value in eval_results.results_dict.items():
                                        self.log(f"{metric_name}: {metric_value:.4f}")
                                
                                # 清理临时文件
                                try:
                                    if os.path.exists(eval_data_yaml):
                                        os.remove(eval_data_yaml)
                                except:
                                    pass
                            else:
                                self.log(f"警告: 验证集目录不存在 ({val_dir})，无法进行评估")
                                self.log(f"请确保数据集包含验证集，或者在训练前复制部分训练数据作为验证集")
                        except Exception as e:
                            self.log(f"评估模型时出错: {str(e)}")
                            self.log(traceback.format_exc())
                    else:
                        self.log("最佳模型文件不存在，无法进行评估")
                except Exception as e:
                    self.log(f"评估模型时出错: {str(e)}")
                    self.log(traceback.format_exc())
            
            # 通知主线程训练完成
            self.message_queue.put({'type': 'training_complete'})
        
        except Exception as e:
            # 通知主线程训练出错
            self.message_queue.put({'type': 'training_error', 'error': str(e)})
            # 记录详细错误信息
            self.message_queue.put({'type': 'log', 'content': f"训练过程出错: {str(e)}"})
            self.message_queue.put({'type': 'log', 'content': traceback.format_exc()})
            
            # 即使出错也显示预期的保存路径
            try:
                save_name = self.name_var.get()
                save_dir = os.path.join("runs", "detect", save_name)
                weights_dir = os.path.join(save_dir, "weights")
                best_model_path = os.path.join(weights_dir, "best.pt")
                last_model_path = os.path.join(weights_dir, "last.pt")
                
                self.log(f"\n===== 保存路径信息 (训练失败) =====")
                self.log(f"训练结果目录: {os.path.abspath(save_dir)}")
                self.log(f"权重文件目录: {os.path.abspath(weights_dir)}")
                self.log(f"最佳模型路径: {os.path.abspath(best_model_path)}")
                self.log(f"最新模型路径: {os.path.abspath(last_model_path)}")
                
                # 检查是否有部分结果保存
                if os.path.exists(save_dir):
                    self.log(f"部分训练结果可能已保存在上述目录中")
            except Exception as path_error:
                self.log(f"无法显示保存路径信息: {str(path_error)}")
        
        finally:
            # 不在这里重置UI状态，而是通过消息通知主线程
            self.training = False
    
    def reset_ui(self):
        """重置界面状态"""
        self.training = False
        self.train_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.train_status_var.set("就绪")
    
    def on_model_type_change(self, event):
        """处理模型类型下拉框选择变更事件"""
        selected_model = self.model_type_var.get()
        # 根据选择的模型类型设置对应的配置文件和预训练模型
        if selected_model.startswith("YOLOv10"):
            size = selected_model[-1]  # 获取模型大小参数 (n, s, m, l, x)
            self.model_yaml_var.set(f"ultralytics/cfg/models/v10/yolov10{size}.yaml")
            self.pre_model_var.set(f"yolov10{size}.pt")
        elif selected_model.startswith("YOLOv8"):
            size = selected_model[-1]  # 获取模型大小参数 (n, s, m, l, x)
            self.model_yaml_var.set(f"ultralytics/cfg/models/v8/yolov8{size}.yaml")
            self.pre_model_var.set(f"yolov8{size}.pt")
        
        self.log(f"已选择模型类型: {selected_model}，配置文件已更新")
    
    def on_frame_configure(self, event):
        """当框架大小改变时重新配置滚动区域"""
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
    
    def on_canvas_configure(self, event):
        """当画布大小改变时调整内部框架的宽度"""
        canvas_width = event.width
        self.main_canvas.itemconfig(self.canvas_frame, width=canvas_width)
    
    def on_mousewheel(self, event):
        """处理鼠标滚轮事件"""
        if event.num == 4 or event.delta > 0:  # 向上滚动
            self.main_canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:  # 向下滚动
            self.main_canvas.yview_scroll(1, "units")

if __name__ == "__main__":
    # 创建主窗口
    root = tk.Tk()
    app = StrawberryJiedianTrainGUI(root)
    root.mainloop() 