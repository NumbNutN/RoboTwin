import os
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_joint_angles(data_np, file_name, save_path=None):
    """
    绘制关节角度折线图，将14个关节分为左右两组
    """
    if len(data_np.shape) != 2 or data_np.shape[1] != 14:
        print(f"错误: 数据形状不正确，期望 (N, 14)，实际 {data_np.shape}")
        return
    
    # 定义关节名称
    left_joint_names = ["左臂关节1", "左臂关节2", "左臂关节3", "左臂关节4", "左臂关节5", "左臂关节6", "左夹爪"]
    right_joint_names = ["右臂关节1", "右臂关节2", "右臂关节3", "右臂关节4", "右臂关节5", "右臂关节6", "右夹爪"]
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(f'关节角度变化 - {file_name}', fontsize=16, fontweight='bold')
    
    # 时间轴
    time_steps = np.arange(data_np.shape[0])
    
    # 绘制左臂关节
    ax1.set_title('左臂关节角度', fontsize=14, fontweight='bold')
    for i in range(7):
        ax1.plot(time_steps, data_np[:, i], label=left_joint_names[i], linewidth=2, alpha=0.8)
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('关节角度 (弧度)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 绘制右臂关节
    ax2.set_title('右臂关节角度', fontsize=14, fontweight='bold')
    for i in range(7):
        ax2.plot(time_steps, data_np[:, i+7], label=right_joint_names[i], linewidth=2, alpha=0.8)
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('关节角度 (弧度)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()

def select_and_plot_file(processed_data_dir):
    """
    让用户选择一个文件并绘制关节角度图
    """
    # 获取所有任务目录
    task_dirs = []
    for item in os.listdir(processed_data_dir):
        item_path = os.path.join(processed_data_dir, item)
        if os.path.isdir(item_path):
            task_dirs.append(item_path)
    
    if not task_dirs:
        print("未找到任务目录")
        return
    
    # 显示可用的任务
    print("\n可用的任务:")
    for i, task_dir in enumerate(task_dirs):
        task_name = os.path.basename(task_dir)
        pt_files = [f for f in os.listdir(task_dir) if f.endswith('.pt')]
        print(f"{i+1}. {task_name} ({len(pt_files)} 个文件)")
    
    # 用户选择任务
    while True:
        try:
            task_choice = int(input(f"\n请选择任务 (1-{len(task_dirs)}): ")) - 1
            if 0 <= task_choice < len(task_dirs):
                selected_task_dir = task_dirs[task_choice]
                break
            else:
                print("无效选择，请重新输入")
        except ValueError:
            print("请输入数字")
    
    # 获取选中任务的所有 .pt 文件
    pt_files = []
    for file in os.listdir(selected_task_dir):
        if file.endswith('.pt'):
            pt_files.append(os.path.join(selected_task_dir, file))
    
    pt_files.sort()
    
    if not pt_files:
        print("该任务目录中没有 .pt 文件")
        return
    
    # 显示可用的文件
    print(f"\n{os.path.basename(selected_task_dir)} 中的文件:")
    for i, pt_file in enumerate(pt_files):
        print(f"{i+1}. {os.path.basename(pt_file)}")
    
    # 用户选择文件
    while True:
        try:
            file_choice = int(input(f"\n请选择文件 (1-{len(pt_files)}): ")) - 1
            if 0 <= file_choice < len(pt_files):
                selected_file = pt_files[file_choice]
                break
            else:
                print("无效选择，请重新输入")
        except ValueError:
            print("请输入数字")
    
    # 加载并绘制数据
    try:
        data = torch.load(selected_file, map_location='cpu')
        if isinstance(data, torch.Tensor):
            data_np = data.numpy()
            file_name = os.path.basename(selected_file)
            
            # 创建保存路径
            save_dir = os.path.join(processed_data_dir, "plots")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{file_name.replace('.pt', '_joint_angles.png')}")
            
            print(f"\n正在绘制 {file_name} 的关节角度图...")
            plot_joint_angles(data_np, file_name, save_path)
            
        else:
            print("选中的文件不是张量格式")
    except Exception as e:
        print(f"加载文件时出错: {e}")

def check_pt_file(file_path):
    """
    检查单个 .pt 文件的内容
    """
    try:
        # 加载 .pt 文件
        data = torch.load(file_path, map_location='cpu')
        
        print(f"\n=== 文件: {os.path.basename(file_path)} ===")
        print(f"文件路径: {file_path}")
        
        # 检查数据类型
        if isinstance(data, torch.Tensor):
            print(f"数据类型: PyTorch Tensor")
            print(f"张量形状: {data.shape}")
            print(f"张量类型: {data.dtype}")
            print(f"张量设备: {data.device}")
            
            # 转换为 numpy 数组以便分析
            data_np = data.numpy()
            
            # 基本统计信息
            print(f"最小值: {data_np.min():.6f}")
            print(f"最大值: {data_np.max():.6f}")
            print(f"平均值: {data_np.mean():.6f}")
            print(f"标准差: {data_np.std():.6f}")
            
            # 显示前几行数据
            print(f"\n前5行数据:")
            if len(data_np.shape) == 2:
                for i in range(min(5, data_np.shape[0])):
                    print(f"  第{i+1}行: {data_np[i]}")
            else:
                print(f"  数据: {data_np[:5]}")
                
            # 如果是动作数据，分析每个关节
            if len(data_np.shape) == 2 and data_np.shape[1] == 14:
                print(f"\n动作数据分析 (14维向量):")
                joint_names = [
                    "左臂关节1", "左臂关节2", "左臂关节3", "左臂关节4", "左臂关节5", "左臂关节6",
                    "左夹爪",
                    "右臂关节1", "右臂关节2", "右臂关节3", "右臂关节4", "右臂关节5", "右臂关节6",
                    "右夹爪"
                ]
                
                for i in range(14):
                    joint_data = data_np[:, i]
                    print(f"  {joint_names[i]}: 范围[{joint_data.min():.4f}, {joint_data.max():.4f}], 均值{joint_data.mean():.4f}")
            
            return {
                'shape': data.shape,
                'dtype': str(data.dtype),
                'min': float(data_np.min()),
                'max': float(data_np.max()),
                'mean': float(data_np.mean()),
                'std': float(data_np.std())
            }
            
        else:
            print(f"数据类型: {type(data)}")
            print(f"数据内容: {data}")
            return {'type': str(type(data))}
            
    except Exception as e:
        print(f"错误: 无法加载文件 {file_path}: {e}")
        return None

def check_task_directory(task_dir):
    """
    检查任务目录中的所有 .pt 文件
    """
    task_name = os.path.basename(task_dir)
    print(f"\n{'='*60}")
    print(f"检查任务: {task_name}")
    print(f"目录: {task_dir}")
    print(f"{'='*60}")
    
    if not os.path.isdir(task_dir):
        print(f"错误: 目录不存在 {task_dir}")
        return
    
    # 查找所有 .pt 文件
    pt_files = []
    for file in os.listdir(task_dir):
        if file.endswith('.pt'):
            pt_files.append(os.path.join(task_dir, file))
    
    if not pt_files:
        print(f"未找到 .pt 文件")
        return
    
    # 按文件名排序
    pt_files.sort()
    
    print(f"找到 {len(pt_files)} 个 .pt 文件:")
    for pt_file in pt_files:
        print(f"  - {os.path.basename(pt_file)}")
    
    # 检查每个文件
    file_info = {}
    for pt_file in pt_files:
        info = check_pt_file(pt_file)
        if info:
            file_info[os.path.basename(pt_file)] = info
    
    return file_info

def main():
    # 检查处理后的数据目录
    processed_data_dir = "/home/numbnut/repo/RobotTwin2/processed_data"
    
    if not os.path.isdir(processed_data_dir):
        print(f"错误: 处理后的数据目录不存在 {processed_data_dir}")
        return
    
    print(f"开始检查处理后的数据目录: {processed_data_dir}")
    
    # 询问用户想要执行的操作
    print("\n请选择操作:")
    print("1. 检查所有 .pt 文件")
    print("2. 选择文件并绘制关节角度图")
    
    while True:
        try:
            choice = int(input("请输入选择 (1 或 2): "))
            if choice in [1, 2]:
                break
            else:
                print("无效选择，请输入 1 或 2")
        except ValueError:
            print("请输入数字")
    
    if choice == 1:
        # 获取所有任务目录
        task_dirs = []
        for item in os.listdir(processed_data_dir):
            item_path = os.path.join(processed_data_dir, item)
            if os.path.isdir(item_path):
                task_dirs.append(item_path)
        
        if not task_dirs:
            print(f"未找到任务目录")
            return
        
        print(f"找到 {len(task_dirs)} 个任务目录")
        
        # 检查每个任务目录
        all_task_info = {}
        for task_dir in task_dirs:
            task_info = check_task_directory(task_dir)
            if task_info:
                all_task_info[os.path.basename(task_dir)] = task_info
        
        # 生成总结报告
        print(f"\n{'='*80}")
        print(f"总结报告")
        print(f"{'='*80}")
        
        for task_name, task_info in all_task_info.items():
            print(f"\n任务: {task_name}")
            print(f"  .pt 文件数量: {len(task_info)}")
            
            if task_info:
                # 检查所有文件是否具有相同的形状
                shapes = set()
                for file_name, info in task_info.items():
                    if 'shape' in info:
                        shapes.add(str(info['shape']))
                
                if len(shapes) == 1:
                    print(f"  所有文件形状一致: {list(shapes)[0]}")
                else:
                    print(f"  文件形状不一致: {shapes}")
                
                # 显示数据范围
                all_mins = [info.get('min', float('inf')) for info in task_info.values()]
                all_maxs = [info.get('max', float('-inf')) for info in task_info.values()]
                
                if all_mins and all_maxs:
                    print(f"  整体数据范围: [{min(all_mins):.6f}, {max(all_maxs):.6f}]")
    
    elif choice == 2:
        select_and_plot_file(processed_data_dir)

if __name__ == "__main__":
    main() 