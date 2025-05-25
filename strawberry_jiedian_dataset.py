import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import threading
import shutil
import random
from pathlib import Path
import json

class StrawberryJiedianDatasetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("草莓截点关联数据集准备工具")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # 创建主框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 创建数据源配置区域
        self.source_frame = ttk.LabelFrame(self.main_frame, text="数据源配置")
        self.source_frame.pack(fill="x", padx=5, pady=5)
        
        # 图像文件夹
        ttk.Label(self.source_frame, text="图像文件夹:").grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)
        self.images_dir_var = tk.StringVar()
        ttk.Entry(self.source_frame, textvariable=self.images_dir_var, width=50).grid(column=1, row=0, padx=10, pady=10, sticky=tk.W)
        ttk.Button(self.source_frame, text="浏览...", command=self.browse_images_dir).grid(column=2, row=0, padx=10, pady=10)
        
        # TXT标签文件夹
        ttk.Label(self.source_frame, text="TXT标签文件夹:").grid(column=0, row=1, padx=10, pady=10, sticky=tk.W)
        self.txt_dir_var = tk.StringVar()
        ttk.Entry(self.source_frame, textvariable=self.txt_dir_var, width=50).grid(column=1, row=1, padx=10, pady=10, sticky=tk.W)
        ttk.Button(self.source_frame, text="浏览...", command=self.browse_txt_dir).grid(column=2, row=1, padx=10, pady=10)
        
        # 同步文件夹选择
        self.same_dir_var = tk.BooleanVar()
        self.same_dir_var.set(True)
        ttk.Checkbutton(self.source_frame, text="图像和标签在同一文件夹", variable=self.same_dir_var, 
                        command=self.sync_dirs).grid(column=1, row=2, padx=10, pady=5, sticky=tk.W)
        
        # 检查TXT格式按钮
        ttk.Button(self.source_frame, text="检查TXT格式", command=self.check_txt_format).grid(column=2, row=2, padx=10, pady=5)
        
        # 创建输出配置区域
        self.output_frame = ttk.LabelFrame(self.main_frame, text="输出配置")
        self.output_frame.pack(fill="x", padx=5, pady=5)
        
        # 输出文件夹
        ttk.Label(self.output_frame, text="输出数据集:").grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)
        self.output_dir_var = tk.StringVar()
        self.output_dir_var.set("datasets/strawberry_jiedian_dataset")
        ttk.Entry(self.output_frame, textvariable=self.output_dir_var, width=50).grid(column=1, row=0, padx=10, pady=10, sticky=tk.W)
        ttk.Button(self.output_frame, text="浏览...", command=self.browse_output_dir).grid(column=2, row=0, padx=10, pady=10)
        
        # 数据集划分
        ratio_frame = ttk.LabelFrame(self.output_frame, text="数据集划分比例")
        ratio_frame.grid(column=0, row=1, columnspan=3, padx=10, pady=10, sticky="we")
        
        ttk.Label(ratio_frame, text="训练集:").grid(column=0, row=0, padx=10, pady=5)
        self.train_ratio_var = tk.StringVar()
        self.train_ratio_var.set("0.8")
        ttk.Entry(ratio_frame, textvariable=self.train_ratio_var, width=10).grid(column=1, row=0, padx=10, pady=5)
        
        ttk.Label(ratio_frame, text="验证集:").grid(column=2, row=0, padx=10, pady=5)
        self.val_ratio_var = tk.StringVar()
        self.val_ratio_var.set("0.1")
        ttk.Entry(ratio_frame, textvariable=self.val_ratio_var, width=10).grid(column=3, row=0, padx=10, pady=5)
        
        ttk.Label(ratio_frame, text="测试集:").grid(column=4, row=0, padx=10, pady=5)
        self.test_ratio_var = tk.StringVar()
        self.test_ratio_var.set("0.1")
        ttk.Entry(ratio_frame, textvariable=self.test_ratio_var, width=10).grid(column=5, row=0, padx=10, pady=5)
        
        # 类别配置
        class_frame = ttk.LabelFrame(self.output_frame, text="类别配置")
        class_frame.grid(column=0, row=2, columnspan=3, padx=10, pady=10, sticky="we")
        
        # 类别名称
        ttk.Label(class_frame, text="类别名称:").grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)
        self.class_names_var = tk.StringVar()
        self.class_names_var.set("小麦细菌性叶斑病（黑秆病） 小麦穗病（麦穗霉病） 小麦叶锈病 小麦松秕病 小麦白粉病 小麦叶斑病（赤霉病） 小麦茎锈病 小麦条锈病")
        ttk.Entry(class_frame, textvariable=self.class_names_var, width=50).grid(column=1, row=0, padx=10, pady=10, sticky=tk.W)
        ttk.Label(class_frame, text="(空格分隔)").grid(column=2, row=0, padx=10, pady=10, sticky=tk.W)
        
        # 关联配置
        relation_frame = ttk.LabelFrame(self.output_frame, text="截点关联配置")
        relation_frame.grid(column=0, row=3, columnspan=3, padx=10, pady=10, sticky="we")
        
        # 使用截点关联处理
        self.use_jiedian_relation_var = tk.BooleanVar()
        self.use_jiedian_relation_var.set(True)
        ttk.Checkbutton(relation_frame, text="启用截点与坏果关联处理", 
                         variable=self.use_jiedian_relation_var).grid(column=0, row=0, columnspan=2, padx=10, pady=5, sticky=tk.W)
        
        # 坏果类别ID
        ttk.Label(relation_frame, text="坏果类别ID:").grid(column=0, row=1, padx=10, pady=5, sticky=tk.W)
        self.bad_fruit_id_var = tk.StringVar()
        self.bad_fruit_id_var.set("1")  # 假设坏果是类别1
        ttk.Entry(relation_frame, textvariable=self.bad_fruit_id_var, width=5).grid(column=1, row=1, padx=10, pady=5, sticky=tk.W)
        
        # 截点类别ID
        ttk.Label(relation_frame, text="截点类别ID:").grid(column=2, row=1, padx=10, pady=5, sticky=tk.W)
        self.jiedian_id_var = tk.StringVar()
        self.jiedian_id_var.set("4")  # 假设截点是类别4
        ttk.Entry(relation_frame, textvariable=self.jiedian_id_var, width=5).grid(column=3, row=1, padx=10, pady=5, sticky=tk.W)
        
        # 截点匹配距离阈值
        ttk.Label(relation_frame, text="截点匹配距离阈值:").grid(column=0, row=2, padx=10, pady=5, sticky=tk.W)
        self.jiedian_distance_var = tk.StringVar()
        self.jiedian_distance_var.set("0.2")  # 相对距离阈值
        ttk.Entry(relation_frame, textvariable=self.jiedian_distance_var, width=5).grid(column=1, row=2, padx=10, pady=5, sticky=tk.W)
        ttk.Label(relation_frame, text="(相对于图像尺寸)").grid(column=2, row=2, padx=10, pady=5, sticky=tk.W)
        
        # 过滤非关联截点
        self.filter_jiedian_var = tk.BooleanVar()
        self.filter_jiedian_var.set(True)
        ttk.Checkbutton(relation_frame, text="过滤未关联的截点", 
                         variable=self.filter_jiedian_var).grid(column=0, row=3, columnspan=2, padx=10, pady=5, sticky=tk.W)
        
        # 按钮区域
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(pady=10)
        
        # 准备数据集按钮
        self.convert_btn = ttk.Button(button_frame, text="准备数据集", command=self.prepare_dataset)
        self.convert_btn.grid(column=0, row=0, padx=10, pady=5)
        
        # 查看数据集按钮
        self.view_btn = ttk.Button(button_frame, text="查看数据集", command=self.view_dataset, state="disabled")
        self.view_btn.grid(column=1, row=0, padx=10, pady=5)
        
        # 日志区域
        self.log_frame = ttk.LabelFrame(self.main_frame, text="处理日志")
        self.log_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=15)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 初始化状态
        self.processing = False
    
    def browse_images_dir(self):
        """浏览图像文件夹"""
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.images_dir_var.set(dir_path)
            if self.same_dir_var.get():
                self.txt_dir_var.set(dir_path)
    
    def browse_txt_dir(self):
        """浏览TXT标签文件夹"""
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.txt_dir_var.set(dir_path)
            if self.same_dir_var.get():
                self.same_dir_var.set(False)
    
    def browse_output_dir(self):
        """浏览输出文件夹"""
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.output_dir_var.set(dir_path)
    
    def sync_dirs(self):
        """同步图像和标签文件夹"""
        if self.same_dir_var.get():
            self.txt_dir_var.set(self.images_dir_var.get())
    
    def log(self, message):
        """添加日志消息"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def set_status(self, message):
        """设置状态栏消息"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def check_txt_format(self):
        """检查TXT标签格式"""
        txt_dir = self.txt_dir_var.get()
        if not os.path.exists(txt_dir):
            messagebox.showerror("错误", f"TXT标签文件夹不存在: {txt_dir}")
            return
        
        self.set_status("正在检查TXT标签格式...")
        self.log("开始检查TXT标签格式...")
        
        txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]
        if not txt_files:
            messagebox.showwarning("警告", f"在 {txt_dir} 中未找到TXT文件")
            self.set_status("就绪")
            return
        
        # 随机选择5个文件进行检查（或者所有文件，如果少于5个）
        sample_files = random.sample(txt_files, min(5, len(txt_files)))
        
        format_issues = []
        class_stats = {}
        
        for txt_file in sample_files:
            file_path = os.path.join(txt_dir, txt_file)
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    format_issues.append(f"{txt_file}: 文件为空")
                    continue
                
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        format_issues.append(f"{txt_file}[行 {i+1}]: 数据不完整，至少需要5个值")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # 更新类别统计
                        class_stats[class_id] = class_stats.get(class_id, 0) + 1
                        
                        # 检查值的范围
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                            format_issues.append(f"{txt_file}[行 {i+1}]: 坐标或尺寸超出范围[0,1]")
                    
                    except ValueError:
                        format_issues.append(f"{txt_file}[行 {i+1}]: 无法解析数值")
            
            except Exception as e:
                format_issues.append(f"{txt_file}: 读取出错 - {str(e)}")
        
        # 显示检查结果
        self.log("TXT标签格式检查完成:")
        
        if format_issues:
            self.log("发现以下格式问题:")
            for issue in format_issues:
                self.log(f"- {issue}")
        else:
            self.log("未发现格式问题")
        
        if class_stats:
            self.log("\n类别统计:")
            class_names = self.class_names_var.get().split()
            for class_id, count in sorted(class_stats.items()):
                class_name = class_names[class_id] if class_id < len(class_names) else f"未知类别 {class_id}"
                self.log(f"- 类别 {class_id} ({class_name}): {count} 个标注")
        
        self.set_status("TXT标签检查完成")
    
    def prepare_dataset(self):
        """准备数据集"""
        if self.processing:
            return
        
        # 检查参数
        images_dir = self.images_dir_var.get()
        txt_dir = self.txt_dir_var.get()
        output_dir = self.output_dir_var.get()
        
        if not os.path.exists(images_dir):
            messagebox.showerror("错误", f"图像文件夹不存在: {images_dir}")
            return
        
        if not os.path.exists(txt_dir):
            messagebox.showerror("错误", f"TXT标签文件夹不存在: {txt_dir}")
            return
        
        try:
            train_ratio = float(self.train_ratio_var.get())
            val_ratio = float(self.val_ratio_var.get())
            test_ratio = float(self.test_ratio_var.get())
            
            if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
                messagebox.showerror("错误", "训练、验证和测试集比例之和必须为1")
                return
        except ValueError:
            messagebox.showerror("错误", "训练、验证和测试集比例必须是有效的浮点数")
            return
        
        # 开始处理
        self.processing = True
        self.convert_btn.config(state="disabled")
        self.view_btn.config(state="disabled")
        self.set_status("正在准备数据集...")
        self.log("开始准备数据集...")
        
        # 在新线程中处理数据集
        threading.Thread(target=self.process_dataset, args=(images_dir, txt_dir, output_dir,
                                                          train_ratio, val_ratio, test_ratio)).start()
    
    def process_dataset(self, images_dir, txt_dir, output_dir, train_ratio, val_ratio, test_ratio):
        """处理数据集（在子线程中运行）"""
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 创建数据集子目录
            train_img_dir = os.path.join(output_dir, "images", "train")
            val_img_dir = os.path.join(output_dir, "images", "val")
            test_img_dir = os.path.join(output_dir, "images", "test")
            
            train_label_dir = os.path.join(output_dir, "labels", "train")
            val_label_dir = os.path.join(output_dir, "labels", "val")
            test_label_dir = os.path.join(output_dir, "labels", "test")
            
            os.makedirs(train_img_dir, exist_ok=True)
            os.makedirs(val_img_dir, exist_ok=True)
            os.makedirs(test_img_dir, exist_ok=True)
            os.makedirs(train_label_dir, exist_ok=True)
            os.makedirs(val_label_dir, exist_ok=True)
            os.makedirs(test_label_dir, exist_ok=True)
            
            # 获取所有图像文件
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                image_files.extend([f for f in os.listdir(images_dir) if f.lower().endswith(ext)])
            
            if not image_files:
                self.log("错误: 在图像文件夹中未找到图像文件")
                self.set_status("处理失败")
                self.processing = False
                self.convert_btn.config(state="normal")
                return
            
            self.log(f"找到 {len(image_files)} 个图像文件")
            
            # 随机排序文件列表
            random.shuffle(image_files)
            
            # 计算各集合的数量
            n_files = len(image_files)
            n_train = int(n_files * train_ratio)
            n_val = int(n_files * val_ratio)
            n_test = n_files - n_train - n_val
            
            # 划分数据集
            train_files = image_files[:n_train]
            val_files = image_files[n_train:n_train+n_val]
            test_files = image_files[n_train+n_val:]
            
            self.log(f"数据集划分: 训练集 {n_train}，验证集 {n_val}，测试集 {n_test}")
            
            # 处理截点关联
            bad_fruit_id = int(self.bad_fruit_id_var.get())
            jiedian_id = int(self.jiedian_id_var.get())
            jiedian_distance = float(self.jiedian_distance_var.get())
            use_jiedian_relation = self.use_jiedian_relation_var.get()
            filter_jiedian = self.filter_jiedian_var.get()
            
            # 处理数据集
            self.log("处理训练集...")
            self._process_subset(train_files, "训练集", images_dir, txt_dir, train_img_dir, train_label_dir,
                               bad_fruit_id, jiedian_id, jiedian_distance, use_jiedian_relation, filter_jiedian)
            
            self.log("处理验证集...")
            self._process_subset(val_files, "验证集", images_dir, txt_dir, val_img_dir, val_label_dir,
                               bad_fruit_id, jiedian_id, jiedian_distance, use_jiedian_relation, filter_jiedian)
            
            self.log("处理测试集...")
            self._process_subset(test_files, "测试集", images_dir, txt_dir, test_img_dir, test_label_dir,
                               bad_fruit_id, jiedian_id, jiedian_distance, use_jiedian_relation, filter_jiedian)
            
            # 创建数据集配置文件
            class_names = self.class_names_var.get().split()
            self._create_data_yaml(output_dir, class_names)
            
            self.log("数据集准备完成")
            self.set_status("数据集准备完成")
            
            # 启用查看数据集按钮
            self.root.after(0, lambda: self.view_btn.config(state="normal"))
            
        except Exception as e:
            self.log(f"处理数据集时出错: {str(e)}")
            self.set_status("处理失败")
        finally:
            self.processing = False
            self.root.after(0, lambda: self.convert_btn.config(state="normal"))
    
    def _process_subset(self, files, subset_name, images_dir, txt_dir, img_output_dir, label_output_dir,
                      bad_fruit_id, jiedian_id, jiedian_distance, use_jiedian_relation, filter_jiedian):
        """处理数据集子集"""
        processed = 0
        skipped = 0
        
        for i, img_file in enumerate(files):
            # 更新状态
            if i % 10 == 0:
                self.set_status(f"正在处理{subset_name}: {i+1}/{len(files)}")
            
            # 构建文件路径
            img_path = os.path.join(images_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            txt_file = f"{base_name}.txt"
            txt_path = os.path.join(txt_dir, txt_file)
            
            # 检查标签文件是否存在
            if not os.path.exists(txt_path):
                skipped += 1
                continue
            
            # 复制图像文件
            shutil.copy2(img_path, os.path.join(img_output_dir, img_file))
            
            # 处理标签文件
            if use_jiedian_relation:
                # 读取标签文件
                annotations = []
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            annotations.append([class_id, x_center, y_center, width, height])
                
                # 分离坏果和截点标注
                bad_fruits = [anno for anno in annotations if anno[0] == bad_fruit_id]
                jiedian_annos = [anno for anno in annotations if anno[0] == jiedian_id]
                other_annos = [anno for anno in annotations if anno[0] != bad_fruit_id and anno[0] != jiedian_id]
                
                # 处理截点与坏果的关联
                valid_jiedian = []
                for jiedian in jiedian_annos:
                    j_x, j_y = jiedian[1], jiedian[2]
                    
                    # 寻找最近的坏果
                    min_dist = float('inf')
                    closest_bad_fruit = None
                    
                    for bad_fruit in bad_fruits:
                        bf_x, bf_y = bad_fruit[1], bad_fruit[2]
                        dist = ((j_x - bf_x) ** 2 + (j_y - bf_y) ** 2) ** 0.5
                        
                        if dist < min_dist:
                            min_dist = dist
                            closest_bad_fruit = bad_fruit
                    
                    # 判断是否满足距离阈值
                    if closest_bad_fruit is not None and min_dist <= jiedian_distance:
                        valid_jiedian.append(jiedian)
                
                # 根据过滤设置决定使用哪些截点
                if filter_jiedian:
                    # 只保留有效截点
                    processed_annotations = bad_fruits + valid_jiedian + other_annos
                else:
                    # 保留所有截点
                    processed_annotations = annotations
                
                # 写入处理后的标签文件
                with open(os.path.join(label_output_dir, txt_file), 'w') as f:
                    for anno in processed_annotations:
                        f.write(' '.join(map(str, anno)) + '\n')
            else:
                # 不处理截点关联，直接复制标签文件
                shutil.copy2(txt_path, os.path.join(label_output_dir, txt_file))
            
            processed += 1
        
        self.log(f"{subset_name}处理完成: 处理 {processed} 个文件，跳过 {skipped} 个文件")
    
    def _create_data_yaml(self, output_dir, class_names):
        """创建数据集配置文件"""
        yaml_path = os.path.join(output_dir, "data.yaml")
        
        data_config = {
            'path': output_dir,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write("# YOLO数据集配置文件\n")
            f.write(f"path: {data_config['path']}\n")
            f.write(f"train: {data_config['train']}\n")
            f.write(f"val: {data_config['val']}\n")
            f.write(f"test: {data_config['test']}\n")
            f.write(f"nc: {data_config['nc']}\n")
            f.write("names:\n")
            for i, name in enumerate(class_names):
                f.write(f"  {i}: '{name}'\n")
        
        self.log(f"创建数据集配置文件: {yaml_path}")
    
    def view_dataset(self):
        """查看准备好的数据集"""
        output_dir = self.output_dir_var.get()
        
        if not os.path.exists(output_dir):
            messagebox.showerror("错误", f"数据集目录不存在: {output_dir}")
            return
        
        yaml_path = os.path.join(output_dir, "data.yaml")
        if not os.path.exists(yaml_path):
            messagebox.showerror("错误", f"数据集配置文件不存在: {yaml_path}")
            return
        
        # 显示数据集统计信息
        train_img_dir = os.path.join(output_dir, "images", "train")
        val_img_dir = os.path.join(output_dir, "images", "val")
        test_img_dir = os.path.join(output_dir, "images", "test")
        
        train_label_dir = os.path.join(output_dir, "labels", "train")
        val_label_dir = os.path.join(output_dir, "labels", "val")
        test_label_dir = os.path.join(output_dir, "labels", "test")
        
        train_imgs = len([f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        val_imgs = len([f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        test_imgs = len([f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        
        train_labels = len([f for f in os.listdir(train_label_dir) if f.lower().endswith('.txt')])
        val_labels = len([f for f in os.listdir(val_label_dir) if f.lower().endswith('.txt')])
        test_labels = len([f for f in os.listdir(test_label_dir) if f.lower().endswith('.txt')])
        
        # 统计各类别的数量
        class_stats = {i: 0 for i in range(10)}  # 假设最多10个类别
        
        for label_dir in [train_label_dir, val_label_dir, test_label_dir]:
            for txt_file in os.listdir(label_dir):
                if txt_file.lower().endswith('.txt'):
                    try:
                        with open(os.path.join(label_dir, txt_file), 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    class_id = int(parts[0])
                                    class_stats[class_id] = class_stats.get(class_id, 0) + 1
                    except:
                        pass
        
        # 创建信息窗口
        info_window = tk.Toplevel(self.root)
        info_window.title("数据集信息")
        info_window.geometry("500x400")
        info_window.transient(self.root)
        info_window.grab_set()
        
        # 创建滚动文本区域
        info_text = scrolledtext.ScrolledText(info_window, wrap=tk.WORD)
        info_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 显示数据集信息
        info_text.insert(tk.END, f"数据集目录: {output_dir}\n\n")
        
        info_text.insert(tk.END, "数据集统计:\n")
        info_text.insert(tk.END, f"训练集: {train_imgs} 图像, {train_labels} 标签\n")
        info_text.insert(tk.END, f"验证集: {val_imgs} 图像, {val_labels} 标签\n")
        info_text.insert(tk.END, f"测试集: {test_imgs} 图像, {test_labels} 标签\n\n")
        
        info_text.insert(tk.END, "类别统计:\n")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ":" in line and "names" not in line and "path" not in line and "train" not in line and "val" not in line and "test" not in line and "nc" not in line:
                    parts = line.strip().split(":", 1)
                    if len(parts) == 2:
                        class_id = int(parts[0].strip())
                        class_name = parts[1].strip().strip("'\"")
                        count = class_stats.get(class_id, 0)
                        info_text.insert(tk.END, f"类别 {class_id} ({class_name}): {count} 个标注\n")
        
        # 添加打开目录按钮
        ttk.Button(
            info_window, 
            text="打开数据集目录", 
            command=lambda: os.startfile(output_dir) if os.name == 'nt' else os.system(f"xdg-open {output_dir}")
        ).pack(pady=10)

if __name__ == "__main__":
    # 创建主窗口
    root = tk.Tk()
    app = StrawberryJiedianDatasetGUI(root)
    root.mainloop() 