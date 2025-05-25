import os
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import importlib.util

class StrawberryJiedianMain:
    def __init__(self, root):
        self.root = root
        self.root.title("草莓截点关联训练与检测系统")
        self.root.geometry("500x450")  # 增加一点高度以容纳新按钮
        self.root.resizable(True, True)
        
        # 检查是否存在所需模块
        self.modules_status = self.check_required_modules()
        
        # 创建主界面
        self.create_main_ui()
    
    def check_required_modules(self):
        """检查所需的Python模块是否已安装"""
        required_modules = {
            "torch": "PyTorch",
            "matplotlib": "Matplotlib",
            "ultralytics": "Ultralytics",
            "numpy": "NumPy",
            "opencv-python": "OpenCV",
            "PIL": "Pillow"
        }
        
        module_status = {}
        
        for module_name, display_name in required_modules.items():
            try:
                if module_name == "opencv-python":
                    # OpenCV的导入名称与包名不同
                    importlib.util.find_spec("cv2")
                    module_status[display_name] = True
                elif module_name == "PIL":
                    # Pillow的导入名称与包名不同
                    importlib.util.find_spec("PIL")
                    module_status[display_name] = True
                else:
                    importlib.util.find_spec(module_name)
                    module_status[display_name] = True
            except ImportError:
                module_status[display_name] = False
        
        return module_status
    
    def create_main_ui(self):
        """创建主用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # 标题标签
        title_label = ttk.Label(main_frame, text="草莓截点关联训练与检测系统", font=("SimHei", 16, "bold"))
        title_label.pack(pady=20)
        
        # 模块状态框
        status_frame = ttk.LabelFrame(main_frame, text="系统状态")
        status_frame.pack(fill="x", padx=10, pady=10)
        
        # 检查系统状态
        all_modules_ok = True
        row = 0
        
        for module_name, status in self.modules_status.items():
            status_text = "已安装" if status else "未安装"
            status_color = "green" if status else "red"
            
            ttk.Label(status_frame, text=f"{module_name}:").grid(column=0, row=row, sticky=tk.W, padx=10, pady=2)
            status_label = ttk.Label(status_frame, text=status_text)
            status_label.grid(column=1, row=row, sticky=tk.W, padx=10, pady=2)
            
            # 使用Label显示彩色状态
            color_indicator = tk.Label(status_frame, text="●", fg=status_color)
            color_indicator.grid(column=2, row=row, sticky=tk.W, padx=5, pady=2)
            
            if not status:
                all_modules_ok = False
            
            row += 1
        
        # 功能按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="both", expand=True, padx=10, pady=20)
        
        # 数据集准备按钮
        dataset_btn = ttk.Button(
            button_frame, 
            text="数据集准备", 
            command=self.launch_dataset_module,
            state="normal" if all_modules_ok else "disabled"
        )
        dataset_btn.pack(fill="x", pady=10)
        
        # 训练模块按钮
        train_btn = ttk.Button(
            button_frame, 
            text="模型训练", 
            command=self.launch_train_module,
            state="normal" if all_modules_ok else "disabled"
        )
        train_btn.pack(fill="x", pady=10)
        
        # 检测模块按钮
        detect_btn = ttk.Button(
            button_frame, 
            text="模型检测", 
            command=self.launch_detect_module,
            state="normal" if all_modules_ok else "disabled"
        )
        detect_btn.pack(fill="x", pady=10)
        
        # 安装依赖按钮
        if not all_modules_ok:
            install_btn = ttk.Button(
                button_frame, 
                text="安装缺失依赖", 
                command=self.install_dependencies
            )
            install_btn.pack(fill="x", pady=10)
        
        # 退出按钮
        exit_btn = ttk.Button(button_frame, text="退出系统", command=self.root.destroy)
        exit_btn.pack(fill="x", pady=10)
        
        # 版本和版权信息
        version_label = ttk.Label(main_frame, text="版本 1.0.0 | 草莓检测研究小组", font=("SimSun", 8))
        version_label.pack(side=tk.BOTTOM, pady=5)
    
    def launch_dataset_module(self):
        """启动数据集准备模块"""
        try:
            # 检查数据集准备模块文件是否存在
            if not os.path.exists("strawberry_jiedian_dataset.py"):
                messagebox.showerror("错误", "数据集准备模块文件不存在")
                return
            
            # 导入数据集准备模块
            import strawberry_jiedian_dataset
            
            # 创建数据集准备模块窗口
            dataset_window = tk.Toplevel(self.root)
            app = strawberry_jiedian_dataset.StrawberryJiedianDatasetGUI(dataset_window)
            
            # 设置窗口属性
            dataset_window.transient(self.root)  # 设置为主窗口的子窗口
            dataset_window.grab_set()  # 获取所有鼠标键盘事件
            
            # 等待窗口关闭
            self.root.wait_window(dataset_window)
            
        except Exception as e:
            messagebox.showerror("错误", f"启动数据集准备模块时出错: {str(e)}")
    
    def launch_train_module(self):
        """启动训练模块"""
        try:
            # 检查训练模块文件是否存在
            if not os.path.exists("strawberry_jiedian_train.py"):
                messagebox.showerror("错误", "训练模块文件不存在")
                return
            
            # 导入训练模块
            import strawberry_jiedian_train
            
            # 创建训练模块窗口
            train_window = tk.Toplevel(self.root)
            app = strawberry_jiedian_train.StrawberryJiedianTrainGUI(train_window)
            
            # 设置窗口属性
            train_window.transient(self.root)  # 设置为主窗口的子窗口
            train_window.grab_set()  # 获取所有鼠标键盘事件
            
            # 等待窗口关闭
            self.root.wait_window(train_window)
            
        except Exception as e:
            messagebox.showerror("错误", f"启动训练模块时出错: {str(e)}")
    
    def launch_detect_module(self):
        """启动检测模块"""
        try:
            # 检查检测模块文件是否存在
            if not os.path.exists("strawberry_jiedian_detect.py"):
                messagebox.showerror("错误", "检测模块文件不存在")
                return
            
            # 导入检测模块
            import strawberry_jiedian_detect
            
            # 创建检测模块窗口
            detect_window = tk.Toplevel(self.root)
            app = strawberry_jiedian_detect.StrawberryJiedianDetectGUI(detect_window)
            
            # 设置窗口属性
            detect_window.transient(self.root)  # 设置为主窗口的子窗口
            detect_window.grab_set()  # 获取所有鼠标键盘事件
            
            # 等待窗口关闭
            self.root.wait_window(detect_window)
            
        except Exception as e:
            messagebox.showerror("错误", f"启动检测模块时出错: {str(e)}")
    
    def install_dependencies(self):
        """安装缺失的依赖"""
        try:
            import subprocess
            
            # 创建安装进度窗口
            progress_window = tk.Toplevel(self.root)
            progress_window.title("安装依赖")
            progress_window.geometry("400x300")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            # 创建文本区域显示安装进度
            import tkinter.scrolledtext as scrolledtext
            text_area = scrolledtext.ScrolledText(progress_window, wrap=tk.WORD, width=40, height=15)
            text_area.pack(padx=10, pady=10, fill="both", expand=True)
            
            # 安装所需包
            missing_packages = []
            for module_name, status in self.modules_status.items():
                if not status:
                    if module_name == "Pillow":
                        missing_packages.append("pillow")
                    elif module_name == "OpenCV":
                        missing_packages.append("opencv-python")
                    else:
                        missing_packages.append(module_name.lower())
            
            if not missing_packages:
                text_area.insert(tk.END, "所有依赖已安装，无需重新安装。\n")
                return
            
            # 运行安装命令
            text_area.insert(tk.END, "开始安装缺失的依赖...\n")
            
            # 确保pip已安装
            text_area.insert(tk.END, "检查pip安装...\n")
            text_area.see(tk.END)
            progress_window.update()
            
            for package in missing_packages:
                text_area.insert(tk.END, f"正在安装 {package}...\n")
                text_area.see(tk.END)
                progress_window.update()
                
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", package],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    text_area.insert(tk.END, f"{package} 安装成功\n")
                    text_area.insert(tk.END, result.stdout + "\n")
                except subprocess.CalledProcessError as e:
                    text_area.insert(tk.END, f"{package} 安装失败: {e.stderr}\n")
                
                text_area.see(tk.END)
                progress_window.update()
            
            text_area.insert(tk.END, "依赖安装完成。请重新启动应用程序以应用更改。\n")
            
            # 添加关闭按钮
            ttk.Button(
                progress_window, 
                text="关闭", 
                command=progress_window.destroy
            ).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("错误", f"安装依赖时出错: {str(e)}")

if __name__ == "__main__":
    # 创建主窗口
    root = tk.Tk()
    app = StrawberryJiedianMain(root)
    root.mainloop() 