import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_gripper_comparison(raw_actions, modified_actions, output_path):
    """
    可视化原始动作和修改后动作中夹爪关节（6和13）的对比。

    :param raw_actions: 包含原始动作的Numpy数组 (N, 14)。
    :param modified_actions: 包含修改后动作的Numpy数组 (N, 14)。
    :param output_path: 图片保存路径。
    """
    # 设置matplotlib以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 提取左右夹爪的数据
    raw_left_gripper = raw_actions[:, 6]
    mod_left_gripper = modified_actions[:, 6]
    raw_right_gripper = raw_actions[:, 13]
    mod_right_gripper = modified_actions[:, 13]

    timesteps = np.arange(raw_actions.shape[0])

    # 创建图表和子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle('夹爪动作修改前后对比', fontsize=18, fontweight='bold')

    # --- 左夹爪子图 ---
    ax1.plot(timesteps, raw_left_gripper, label='原始动作 (Raw)', color='cornflowerblue', linewidth=2, alpha=0.8)
    ax1.plot(timesteps, mod_left_gripper, label='修改后动作 (Modified)', color='crimson', linestyle='--', linewidth=2)
    ax1.set_title('左夹爪动作 (关节 6)', fontsize=14)
    ax1.set_ylabel('夹爪值')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 右夹爪子图 ---
    ax2.plot(timesteps, raw_right_gripper, label='原始动作 (Raw)', color='cornflowerblue', linewidth=2, alpha=0.8)
    ax2.plot(timesteps, mod_right_gripper, label='修改后动作 (Modified)', color='crimson', linestyle='--', linewidth=2)
    ax2.set_title('右夹爪动作 (关节 13)', fontsize=14)
    ax2.set_xlabel('时间步 (Frame)')
    ax2.set_ylabel('夹爪值')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图像
    plt.savefig(output_path, dpi=300)
    print(f"图表已保存到: {output_path}")

    # 显示图像
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="可视化对比原始动作和修改后动作中夹爪关节的变化。")
    parser.add_argument(
        "episode_dir",
        type=str,
        help="包含 raw_actions.npy 和 modified_actions.npy 文件的 episode 目录路径。"
    )
    args = parser.parse_args()

    # 构建文件路径
    raw_actions_path = os.path.join(args.episode_dir, "raw_actions.npy")
    modified_actions_path = os.path.join(args.episode_dir, "modified_actions.npy")
    output_plot_path = os.path.join(args.episode_dir, "gripper_actions_comparison.png")

    # 检查文件是否存在
    if not os.path.exists(raw_actions_path):
        print(f"错误: 未找到文件 {raw_actions_path}")
        return
    if not os.path.exists(modified_actions_path):
        print(f"错误: 未找到文件 {modified_actions_path}")
        return

    # 加载数据
    print("正在加载动作数据...")
    raw_actions = np.load(raw_actions_path)
    modified_actions = np.load(modified_actions_path)
    print("数据加载完毕。")

    # 调用绘图函数
    plot_gripper_comparison(raw_actions, modified_actions, output_plot_path)

if __name__ == "__main__":
    main() 