import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import subprocess
import threading
import json
from pathlib import Path
import shutil

class StrawberryDetectGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("草莓检测训练工具")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # 创建标签页
        self.tab_control = ttk.Notebook(root)
        
        # 数据集准备标签页
        self.convert_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.convert_tab, text="数据集准备")
        
        # 模型训练标签页
        self.train_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.train_tab, text="模型训练")
        
        # TXT调试标签页
        self.debug_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.debug_tab, text="TXT调试")
        
        self.tab_control.pack(expand=1, fill="both")
        
        # 创建数据集准备页面组件
        self.create_convert_tab()
        
        # 创建模型训练页面组件
        self.create_train_tab()
        
        # 创建TXT调试页面组件
        self.create_debug_tab()
        
        # 日志区域
        self.log_frame = ttk.LabelFrame(root, text="日志输出")
        self.log_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        self.log_text = tk.Text(self.log_frame, height=10)
        self.log_text.pack(pady=5, padx=5, fill="both", expand=True)
        
        # 滚动条
        scroll = ttk.Scrollbar(self.log_text)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        scroll.config(command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scroll.set)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_convert_tab(self):
        # 图像文件夹
        ttk.Label(self.convert_tab, text="图像文件夹:").grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)
        self.images_dir_var = tk.StringVar()
        ttk.Entry(self.convert_tab, textvariable=self.images_dir_var, width=50).grid(column=1, row=0, padx=10, pady=10, sticky=tk.W)
        ttk.Button(self.convert_tab, text="浏览...", command=self.browse_images_dir).grid(column=2, row=0, padx=10, pady=10)
        
        # TXT标签文件夹
        ttk.Label(self.convert_tab, text="TXT标签文件夹:").grid(column=0, row=1, padx=10, pady=10, sticky=tk.W)
        self.txt_dir_var = tk.StringVar()
        ttk.Entry(self.convert_tab, textvariable=self.txt_dir_var, width=50).grid(column=1, row=1, padx=10, pady=10, sticky=tk.W)
        ttk.Button(self.convert_tab, text="浏览...", command=self.browse_txt_dir).grid(column=2, row=1, padx=10, pady=10)
        
        # 同步文件夹选择
        self.same_dir_var = tk.BooleanVar()
        self.same_dir_var.set(True)
        ttk.Checkbutton(self.convert_tab, text="图像和标签在同一文件夹", variable=self.same_dir_var, 
                        command=self.sync_dirs).grid(column=1, row=2, padx=10, pady=5, sticky=tk.W)
        
        # 检查TXT格式按钮
        ttk.Button(self.convert_tab, text="检查TXT格式", command=self.check_txt_format).grid(column=2, row=2, padx=10, pady=5)
        
        # 输出文件夹
        ttk.Label(self.convert_tab, text="输出数据集:").grid(column=0, row=3, padx=10, pady=10, sticky=tk.W)
        self.output_dir_var = tk.StringVar()
        self.output_dir_var.set("datasets/strawberry_dataset")
        ttk.Entry(self.convert_tab, textvariable=self.output_dir_var, width=50).grid(column=1, row=3, padx=10, pady=10, sticky=tk.W)
        ttk.Button(self.convert_tab, text="浏览...", command=self.browse_output_dir).grid(column=2, row=3, padx=10, pady=10)
        
        # 数据集划分
        ratio_frame = ttk.LabelFrame(self.convert_tab, text="数据集划分比例")
        ratio_frame.grid(column=0, row=4, columnspan=3, padx=10, pady=10, sticky="we")
        
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
        
        # 类别名称
        ttk.Label(self.convert_tab, text="类别名称:").grid(column=0, row=5, padx=10, pady=10, sticky=tk.W)
        self.class_names_var = tk.StringVar()
        self.class_names_var.set("小麦细菌性叶斑病（黑秆病） 小麦穗病（麦穗霉病） 小麦叶锈病 小麦松秕病 小麦白粉病 小麦叶斑病（赤霉病） 小麦茎锈病 小麦条锈病")
        ttk.Entry(self.convert_tab, textvariable=self.class_names_var, width=50).grid(column=1, row=5, padx=10, pady=10, sticky=tk.W)
        ttk.Label(self.convert_tab, text="(空格分隔)").grid(column=2, row=5, padx=10, pady=10, sticky=tk.W)
        
        # 准备数据集按钮
        convert_btn = ttk.Button(self.convert_tab, text="准备数据集", command=self.prepare_dataset)
        convert_btn.grid(column=1, row=7, padx=10, pady=20)
    
    def create_debug_tab(self):
        # 选择单个TXT文件进行检查
        ttk.Label(self.debug_tab, text="TXT文件:").grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)
        self.debug_txt_var = tk.StringVar()
        ttk.Entry(self.debug_tab, textvariable=self.debug_txt_var, width=50).grid(column=1, row=0, padx=10, pady=10, sticky=tk.W)
        ttk.Button(self.debug_tab, text="浏览...", command=self.browse_debug_txt).grid(column=2, row=0, padx=10, pady=10)
        
        # TXT内容显示区域
        ttk.Label(self.debug_tab, text="TXT内容:").grid(column=0, row=1, padx=10, pady=5, sticky=tk.NW)
        self.txt_display = scrolledtext.ScrolledText(self.debug_tab, width=80, height=15)
        self.txt_display.grid(column=0, row=2, columnspan=3, padx=10, pady=5, sticky="nsew")
        
        # 调试信息
        self.debug_info = ttk.LabelFrame(self.debug_tab, text="解析结果")
        self.debug_info.grid(column=0, row=3, columnspan=3, padx=10, pady=10, sticky="we")
        
        ttk.Label(self.debug_info, text="检测到的类别ID:").grid(column=0, row=0, padx=10, pady=5, sticky=tk.W)
        self.detected_classes = tk.StringVar()
        ttk.Label(self.debug_info, textvariable=self.detected_classes).grid(column=1, row=0, padx=10, pady=5, sticky=tk.W)
        
        ttk.Label(self.debug_info, text="标注数量:").grid(column=0, row=1, padx=10, pady=5, sticky=tk.W)
        self.anno_count = tk.StringVar()
        ttk.Label(self.debug_info, textvariable=self.anno_count).grid(column=1, row=1, padx=10, pady=5, sticky=tk.W)
        
        # 检查按钮
        ttk.Button(self.debug_tab, text="解析TXT", command=self.parse_txt).grid(column=1, row=4, padx=10, pady=10)
    
    def create_train_tab(self):
        # 数据集配置文件
        ttk.Label(self.train_tab, text="数据集配置:").grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)
        self.data_yaml_var = tk.StringVar()
        self.data_yaml_var.set("datasets/strawberry_dataset/data.yaml")
        ttk.Entry(self.train_tab, textvariable=self.data_yaml_var, width=50).grid(column=1, row=0, padx=10, pady=10, sticky=tk.W)
        ttk.Button(self.train_tab, text="浏览...", command=self.browse_data_yaml).grid(column=2, row=0, padx=10, pady=10)
        
        # 模型配置文件
        ttk.Label(self.train_tab, text="模型配置:").grid(column=0, row=1, padx=10, pady=10, sticky=tk.W)
        self.model_yaml_var = tk.StringVar()
        self.model_yaml_var.set("ultralytics/cfg/models/v10/yolov10n.yaml")
        ttk.Entry(self.train_tab, textvariable=self.model_yaml_var, width=50).grid(column=1, row=1, padx=10, pady=10, sticky=tk.W)
        ttk.Button(self.train_tab, text="浏览...", command=self.browse_model_yaml).grid(column=2, row=1, padx=10, pady=10)
        
        # 预训练模型
        ttk.Label(self.train_tab, text="预训练模型:").grid(column=0, row=2, padx=10, pady=10, sticky=tk.W)
        self.pre_model_var = tk.StringVar()
        self.pre_model_var.set("yolov10n.pt")
        ttk.Entry(self.train_tab, textvariable=self.pre_model_var, width=50).grid(column=1, row=2, padx=10, pady=10, sticky=tk.W)
        ttk.Button(self.train_tab, text="浏览...", command=self.browse_pre_model).grid(column=2, row=2, padx=10, pady=10)
        
        # 训练参数框架
        param_frame = ttk.LabelFrame(self.train_tab, text="训练参数")
        param_frame.grid(column=0, row=3, columnspan=3, padx=10, pady=10, sticky="we")
        
        # 训练轮数
        ttk.Label(param_frame, text="训练轮数:").grid(column=0, row=0, padx=10, pady=5)
        self.epochs_var = tk.StringVar()
        self.epochs_var.set("150")
        ttk.Entry(param_frame, textvariable=self.epochs_var, width=10).grid(column=1, row=0, padx=10, pady=5)
        
        # 批次大小
        ttk.Label(param_frame, text="批次大小:").grid(column=2, row=0, padx=10, pady=5)
        self.batch_var = tk.StringVar()
        self.batch_var.set("2")
        ttk.Entry(param_frame, textvariable=self.batch_var, width=10).grid(column=3, row=0, padx=10, pady=5)
        
        # 图像尺寸
        ttk.Label(param_frame, text="图像尺寸:").grid(column=4, row=0, padx=10, pady=5)
        self.imgsz_var = tk.StringVar()
        self.imgsz_var.set("640")
        ttk.Entry(param_frame, textvariable=self.imgsz_var, width=10).grid(column=5, row=0, padx=10, pady=5)
        
        # 训练设备
        ttk.Label(param_frame, text="训练设备:").grid(column=0, row=1, padx=10, pady=5)
        self.device_var = tk.StringVar()
        self.device_var.set("cpu")
        ttk.Entry(param_frame, textvariable=self.device_var, width=10).grid(column=1, row=1, padx=10, pady=5)
        
        # 结果保存名称
        ttk.Label(param_frame, text="保存名称:").grid(column=2, row=1, padx=10, pady=5)
        self.name_var = tk.StringVar()
        self.name_var.set("strawberry_model")
        ttk.Entry(param_frame, textvariable=self.name_var, width=20).grid(column=3, row=1, padx=10, pady=5, columnspan=2)
        
        # 训练按钮
        train_btn = ttk.Button(self.train_tab, text="开始训练", command=self.train_model)
        train_btn.grid(column=1, row=4, padx=10, pady=20)
    
    def browse_images_dir(self):
        folder_path = filedialog.askdirectory(title="选择图像文件夹")
        if folder_path:
            self.images_dir_var.set(folder_path)
            if self.same_dir_var.get():
                self.txt_dir_var.set(folder_path)
    
    def browse_txt_dir(self):
        folder_path = filedialog.askdirectory(title="选择TXT标签文件夹")
        if folder_path:
            self.txt_dir_var.set(folder_path)
    
    def browse_output_dir(self):
        folder_path = filedialog.askdirectory(title="选择输出数据集文件夹")
        if folder_path:
            self.output_dir_var.set(folder_path)
    
    def browse_data_yaml(self):
        file_path = filedialog.askopenfilename(title="选择数据集配置文件", filetypes=[("YAML文件", "*.yaml")])
        if file_path:
            self.data_yaml_var.set(file_path)
    
    def browse_model_yaml(self):
        file_path = filedialog.askopenfilename(title="选择模型配置文件", filetypes=[("YAML文件", "*.yaml")])
        if file_path:
            self.model_yaml_var.set(file_path)
    
    def browse_pre_model(self):
        file_path = filedialog.askopenfilename(title="选择预训练模型", filetypes=[("PT文件", "*.pt")])
        if file_path:
            self.pre_model_var.set(file_path)
    
    def sync_dirs(self):
        if self.same_dir_var.get():
            self.txt_dir_var.set(self.images_dir_var.get())
    
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def set_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def run_process(self, command, on_complete=None):
        self.log(f"执行命令: {command}")
        
        def run():
            process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace',
                shell=True
            )
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self.log(output.strip())
            
            return_code = process.poll()
            if return_code == 0:
                self.set_status("命令执行成功")
                if on_complete:
                    on_complete()
            else:
                self.set_status(f"命令执行失败，返回码: {return_code}")
        
        threading.Thread(target=run).start()
    
    def prepare_dataset(self):
        images_dir = self.images_dir_var.get()
        txt_dir = self.txt_dir_var.get()
        output_dir = self.output_dir_var.get()
        train_ratio = float(self.train_ratio_var.get())
        val_ratio = float(self.val_ratio_var.get())
        test_ratio = float(self.test_ratio_var.get())
        class_names = self.class_names_var.get().split()
        
        if not images_dir or not txt_dir:
            messagebox.showerror("错误", "请选择图像和TXT标签文件夹")
            return
        
        # 验证分割比例
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            messagebox.showerror("错误", "划分比例之和必须等于1")
            return
        
        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 创建train、val、test子目录及其images和labels子目录
            for subset in ["train", "valid", "test"]:
                for subdir in ["images", "labels"]:
                    os.makedirs(os.path.join(output_dir, subset, subdir), exist_ok=True)
            
            # 获取所有图像文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(Path(images_dir).glob(f"*{ext}")))
                image_files.extend(list(Path(images_dir).glob(f"*{ext.upper()}")))
            
            if not image_files:
                messagebox.showerror("错误", f"在 {images_dir} 中没有找到图像文件")
                return
            
            # 打乱图像文件顺序
            import random
            random.shuffle(image_files)
            
            # 计算每个集合的图像数量
            total_images = len(image_files)
            train_size = int(total_images * train_ratio)
            val_size = int(total_images * val_ratio)
            
            train_files = image_files[:train_size]
            val_files = image_files[train_size:train_size+val_size]
            test_files = image_files[train_size+val_size:]
            
            # 复制文件到相应目录
            self.set_status("正在准备数据集...")
            self.log(f"找到 {total_images} 个图像文件")
            self.log(f"训练集: {len(train_files)} 张图像")
            self.log(f"验证集: {len(val_files)} 张图像")
            self.log(f"测试集: {len(test_files)} 张图像")
            
            # 处理每个子集
            self._process_subset(train_files, "train", images_dir, txt_dir, output_dir)
            self._process_subset(val_files, "valid", images_dir, txt_dir, output_dir)
            self._process_subset(test_files, "test", images_dir, txt_dir, output_dir)
            
            # 创建data.yaml文件
            self._create_data_yaml(output_dir, class_names)
            
            self.set_status("数据集准备完成")
            self.log(f"数据集已保存到 {output_dir}")
            self.data_yaml_var.set(os.path.join(output_dir, "data.yaml"))
            messagebox.showinfo("成功", f"数据集准备完成，配置文件保存在 {output_dir}/data.yaml")
            
        except Exception as e:
            self.log(f"错误: {str(e)}")
            messagebox.showerror("错误", f"准备数据集时出错: {str(e)}")
    
    def _process_subset(self, files, subset, images_dir, txt_dir, output_dir):
        """处理数据集子集，复制图像和标签文件"""
        for img_file in files:
            # 复制图像
            dest_img = os.path.join(output_dir, subset, "images", img_file.name)
            shutil.copy2(img_file, dest_img)
            
            # 对应的标签文件
            txt_file = os.path.join(txt_dir, img_file.stem + ".txt")
            if os.path.exists(txt_file):
                dest_txt = os.path.join(output_dir, subset, "labels", img_file.stem + ".txt")
                shutil.copy2(txt_file, dest_txt)
            else:
                self.log(f"警告: 未找到 {img_file.name} 对应的标签文件")
    
    def _create_data_yaml(self, output_dir, class_names):
        """创建data.yaml配置文件"""
        yaml_content = f"""# YOLOv5 dataset config
# Train/val/test sets

train: {os.path.abspath(os.path.join(output_dir, 'train'))}
val: {os.path.abspath(os.path.join(output_dir, 'valid'))}
test: {os.path.abspath(os.path.join(output_dir, 'test'))}

# Classes
nc: {len(class_names)}  # number of classes
names: {class_names}  # class names
"""
        
        with open(os.path.join(output_dir, "data.yaml"), 'w', encoding='utf-8') as f:
            f.write(yaml_content)
    
    def train_model(self):
        data_yaml = self.data_yaml_var.get()
        model_yaml = self.model_yaml_var.get()
        pre_model = self.pre_model_var.get()
        epochs = self.epochs_var.get()
        batch = self.batch_var.get()
        imgsz = self.imgsz_var.get()
        device = self.device_var.get()
        name = self.name_var.get()
        
        if not data_yaml:
            messagebox.showerror("错误", "请选择数据集配置文件")
            return
        
        command = f'python train_custom_dataset.py --data_yaml "{data_yaml}" --model_yaml "{model_yaml}" --pre_model "{pre_model}" --epochs {epochs} --batch {batch} --imgsz {imgsz} --device {device} --name {name}'
        
        self.set_status("正在训练模型...")
        self.run_process(command)
    
    def browse_debug_txt(self):
        file_path = filedialog.askopenfilename(title="选择TXT文件", filetypes=[("TXT文件", "*.txt")])
        if file_path:
            self.debug_txt_var.set(file_path)
            self.load_txt_content(file_path)
    
    def load_txt_content(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                txt_data = f.read()
                self.txt_display.delete(1.0, tk.END)
                self.txt_display.insert(tk.END, txt_data)
        except Exception as e:
            self.txt_display.delete(1.0, tk.END)
            self.txt_display.insert(tk.END, f"错误: 无法读取TXT文件\n{str(e)}")
    
    def parse_txt(self):
        txt_file = self.debug_txt_var.get()
        if not txt_file or not os.path.exists(txt_file):
            messagebox.showerror("错误", "请选择有效的TXT文件")
            return
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 分析YOLO格式TXT文件
            class_ids = set()
            valid_lines = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    self.log(f"警告: 行格式不符合YOLO标准 (应有5个值): {line}")
                    continue
                
                try:
                    class_id = int(parts[0])
                    class_ids.add(class_id)
                    valid_lines += 1
                except ValueError:
                    self.log(f"警告: 类别ID不是整数: {parts[0]}")
            
            # 更新界面
            self.detected_classes.set(", ".join(map(str, sorted(class_ids))))
            self.anno_count.set(str(valid_lines))
            
            if valid_lines > 0:
                messagebox.showinfo("解析结果", f"成功解析YOLO格式TXT文件，包含 {valid_lines} 个有效标注，检测到 {len(class_ids)} 个类别")
            else:
                messagebox.showwarning("解析结果", "未找到有效的YOLO格式标注")
            
        except Exception as e:
            messagebox.showerror("错误", f"解析TXT时出错: {str(e)}")
    
    def check_txt_format(self):
        txt_dir = self.txt_dir_var.get()
        if not txt_dir:
            messagebox.showerror("错误", "请先选择TXT标签文件夹")
            return
        
        try:
            txt_files = list(Path(txt_dir).glob("*.txt"))
            if not txt_files:
                messagebox.showerror("错误", f"在 {txt_dir} 中没有找到TXT文件")
                return
            
            # 加载第一个TXT文件进行分析
            first_txt = txt_files[0]
            self.debug_txt_var.set(str(first_txt))
            self.tab_control.select(2)  # 切换到debug标签页
            self.load_txt_content(first_txt)
            self.parse_txt()
            
        except Exception as e:
            messagebox.showerror("错误", f"检查TXT格式时出错: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = StrawberryDetectGUI(root)
    root.mainloop() 