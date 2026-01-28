import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def visualize_trajectory(file_path, dt=0.02):
    """
    Visualize 14-DoF Robot Arm Joint Positions and Derived Velocities from HDF5.
    
    Args:
        file_path (str): Path to the .hdf5 file.
        dt (float): Timestep between frames for velocity calculation (default: 0.02s).
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Loading data from {file_path}...")
            
            # Check structure
            if 'joint_action' not in f:
                print("Error: 'joint_action' group not found in HDF5 file.")
                return
            
            ja = f['joint_action']
            if 'left_arm' not in ja or 'right_arm' not in ja:
                print("Error: 'left_arm' or 'right_arm' dataset not found in 'joint_action'.")
                return

            # Read Data
            qpos_l = ja['left_arm'][:]
            qpos_r = ja['right_arm'][:]
            
            n_steps = qpos_l.shape[0]
            dof_arm = qpos_l.shape[1] if len(qpos_l.shape) > 1 else 1
            print(f"Loaded {n_steps} steps. Arm DoF: {dof_arm}")

            # Try to load grippers
            gripper_l = None
            gripper_r = None
            if 'left_gripper' in ja:
                gripper_l = ja['left_gripper'][:]
                # Ensure shape (N, 1)
                if len(gripper_l.shape) == 1: gripper_l = gripper_l.reshape(-1, 1)
            
            if 'right_gripper' in ja:
                gripper_r = ja['right_gripper'][:]
                if len(gripper_r.shape) == 1: gripper_r = gripper_r.reshape(-1, 1)

            # Construct Full State per Arm
            # If gripper exists, append it.
            if gripper_l is not None:
                pose_l_full = np.hstack([qpos_l, gripper_l])
            else:
                pose_l_full = qpos_l

            if gripper_r is not None:
                pose_r_full = np.hstack([qpos_r, gripper_r])
            else:
                pose_r_full = qpos_r
            
            dof_full_per_side = pose_l_full.shape[1]
            qpos_all = np.hstack([pose_l_full, pose_r_full])
            total_dof = qpos_all.shape[1]
            
            print(f"Total DoF for Visualization: {total_dof} ({dof_full_per_side} per side)")

            # Calculate Velocity (v = dx/dt)
            qvel_all = np.diff(qpos_all, axis=0) / dt
            qvel_all = np.vstack([qvel_all, np.zeros((1, total_dof))]) 

            # Plotting
            fig, axes = plt.subplots(dof_full_per_side, 2, figsize=(15, 3 * dof_full_per_side), sharex=True)
            fig.suptitle(f'{total_dof}-DoF Joint Analysis: {os.path.basename(file_path)}', fontsize=16)
            
            time_steps = np.arange(n_steps) * dt

            # Function to label joints
            def get_label(idx, dof_arm):
                if idx < dof_arm:
                    return f'Joint {idx}'
                else:
                    return f'Gripper'

            # Left Arm
            for i in range(dof_full_per_side):
                ax = axes[i, 0] if dof_full_per_side > 1 else axes[0]
                # Position
                ax.plot(time_steps, pose_l_full[:, i], label=f'Pos', color='blue', linewidth=1.5)
                # Velocity (Twin Axis)
                ax2 = ax.twinx()
                ax2.plot(time_steps, qvel_all[:, i], label=f'Vel', color='orange', linestyle='--', linewidth=1.0)
                
                label_name = get_label(i, dof_arm)
                ax.set_ylabel(f'L_{label_name}')
                if i == 0: ax.set_title(f'Left Arm ({dof_full_per_side} DoF)')
                if i == dof_full_per_side - 1: ax.set_xlabel('Time (s)')
                
                # Combine legends (only for first plot to reduce clutter, or simple legend)
                # ax.legend(loc='upper left', fontsize='x-small')

            # Right Arm
            for i in range(dof_full_per_side):
                ax = axes[i, 1] if dof_full_per_side > 1 else axes[1]
                idx = i + dof_full_per_side
                # Position
                ax.plot(time_steps, pose_r_full[:, i], label=f'Pos', color='green', linewidth=1.5)
                # Velocity (Twin Axis)
                ax2 = ax.twinx()
                ax2.plot(time_steps, qvel_all[:, idx], label=f'Vel', color='red', linestyle='--', linewidth=1.0)
                
                label_name = get_label(i, dof_arm)
                ax.set_ylabel(f'R_{label_name}')
                if i == 0: ax.set_title(f'Right Arm ({dof_full_per_side} DoF)')
                if i == dof_full_per_side - 1: ax.set_xlabel('Time (s)')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            
            output_img = file_path.replace('.hdf5', '_traj_vis.png')
            plt.savefig(output_img)
            print(f"Visualization saved to: {output_img}")
            # plt.show() # Uncomment if running in valid display environment

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Robot Arm Joint Data from HDF5")
    parser.add_argument("file", help="Path to the .hdf5 trajectory file")
    parser.add_argument("--dt", type=float, default=0.02, help="Control timestep (default 0.02s / 50Hz)")
    
    args = parser.parse_args()
    visualize_trajectory(args.file, args.dt)
