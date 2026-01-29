
import os
import h5py
import numpy as np
import torch
import pickle
from tqdm import tqdm
import sys
import yaml

# Add workspace to path
sys.path.append(os.getcwd())

from envs.open_laptop import open_laptop
from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs._base_task import Base_Task
from envs.utils.save_file import save_pkl
from envs.utils import ArmTag, get_face_prod
from envs.utils.pkl2hdf5 import (
    load_pkl_file, 
    parse_dict_structure, 
    append_data_to_structure, 
    create_hdf5_from_dict, 
    images_to_video
)
from envs.utils.traj_inspector import analyze_trajectory

def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

# Define new class for Data Generation
class OpenLaptopDataGen(open_laptop):
    """
    Extended Open Laptop Env for Data Generation with Negative Sampling
    """
    def __init__(self):
        super().__init__()
        self.sample_type = 'anchor' # 'anchor', 'positive', 'negative'
        self.start_qpos = None
        self.branch_idx = 0
        self.neg_step_idx = 0
        self.pos_step_idx = 0
        
        # New: Counters and Configs for Phase 2
        self.pos_step_counter = 0
        self.current_phase = "default" # Tracks current high-level phase
        
        # Sampling Configuration per Phase
        # Allows granular control over when/how samples are generated
        self.sampling_config = {
            "grasp": {
                "neg": {"active": True, "interval": 100,  "duration": 100},
                "pos": {"active": False, "n_samples": 1, "duration": 100} 
            },
            "rotate": {
                "neg": {"active": True, "interval": 50,   "duration": 100},
                "pos": {"active": True,  "n_samples": 0, "duration": 100}
            },
            "default": {
                "neg": {"active": False, "interval": 999, "duration": 10},
                "pos": {"active": False, "n_samples": 0, "duration": 10}
            }
        }
        
        self.neg_duration = 50 # Default fallback
        self.pos_duration = 50 # Default fallback
    def check_collision(self):
        """
        Check if robot is in collision with anything other than target.
        For data generation, we want "safe" trajectories.
        This is a simple check using Sapien's get_contacts().
        """
        contacts = self.scene.get_contacts()
        for contact in contacts:
             # Basic logic: If robot links touch something that is NOT laptop
             # Assuming we have actor lists.
             # self.robot.left_entity -> list of links
             # self.laptop.actor
             
             # Filtering is environment dependent.
             # For now, return True if ANY contact with non-robot, non-ground (if any)
             # But grasping requires contact.
             # So we check impulsive force? Or specific forbidden links.
             
             # Placeholder: Assume safe if no large penetration
             pass
        return False # TODO: Implement real collision logic

    def set_sample_intervals(self, intervals):
        """
        Set sampling intervals for different phases
        """
        self.phase_intervals.update(intervals)

    def take_dense_action(self, control_seq, save_freq=-1):
        """
        Overridden to inject Negative Sampling logic during execution.
        """
        if self.sample_type == 'anchor':
             print(f"Executing take_dense_action {self.FRAME_IDX}")

        # Unpack control sequence
        left_arm, left_gripper, right_arm, right_gripper = (
            control_seq["left_arm"],
            control_seq["left_gripper"],
            control_seq["right_arm"],
            control_seq["right_gripper"],
        )

        save_freq = self.save_freq if save_freq == -1 else save_freq
        
        # Initial Save (Only in Replay Phase)
        if save_freq != None and not self.need_plan:
            self._take_picture()

        max_control_len = 0
        if left_arm is not None:
             max_control_len = max(max_control_len, left_arm["position"].shape[0])
        if left_gripper is not None:
             max_control_len = max(max_control_len, left_gripper["num_step"])
        if right_arm is not None:
             max_control_len = max(max_control_len, right_arm["position"].shape[0])
        if right_gripper is not None:
             max_control_len = max(max_control_len, right_gripper["num_step"])

        # Capture Start State for Positive Sampling (Look-back window start)
        # control_start_state = self.get_state() if (save_freq is not None and not self.need_plan) else None

        # Reset/Init segment history of states for this dense action sequence
        self.segment_states = []

        for control_idx in range(max_control_len):
             # Record state at the beginning of each step (before action applied)
             # This corresponds to state at index 'control_idx'
             if save_freq is not None and not self.need_plan and self.sample_type == 'anchor':
                  self.segment_states.append(self.get_state())

             # --- INJECTED NEGATIVE SAMPLING LOGIC START (Phase 2 Only) ---
             # Retrieve Config for Current Phase
             phase_cfg = self.sampling_config.get(self.current_phase, self.sampling_config["default"])
             neg_cfg = phase_cfg["neg"]

             # We only sample negatives if we are NOT planning (need_plan=False) 
             # and we are currently tracking a positive trajectory.
             # Also ensure we are in a data-saving mode (save_freq is not None)
             if (save_freq is not None and 
                 not self.need_plan and 
                 self.sample_type == 'anchor' and
                 neg_cfg["active"]):
                 
                 # Using internal counter to be consistent
                 if self.pos_step_counter % neg_cfg["interval"] == 0:
                     # 1. Save Current Good State
                     state_backup = self.get_state()
                     
                     # 2. Rollout Negative Sample (using helper)
                     # Using FRAME_IDX as branch_idx for traceability
                     # print(f"Sample Neg Traj at save index {self.FRAME_IDX} at control index {control_idx} for episode {self.ep_num}")
                     print(f"Generating Negative Sample for Episode {self.ep_num} at Frame {self.FRAME_IDX} (Control Step {control_idx})")
                     self.sample_neg_from(duration=neg_cfg["duration"], branch_idx=self.FRAME_IDX) 
                     
                     # 3. Restore State
                     self.set_state(state_backup)
                     
                 self.pos_step_counter += 1
             # --- INJECTED NEGATIVE SAMPLING LOGIC END ---

             if (left_arm is not None and control_idx < left_arm["position"].shape[0]): 
                 # control left arm
                 self.robot.set_arm_joints(
                     left_arm["position"][control_idx],
                     left_arm["velocity"][control_idx],
                     "left",
                 )

             if left_gripper is not None and control_idx < left_gripper["num_step"]:
                 self.robot.set_gripper(
                     left_gripper["result"][control_idx],
                     "left",
                     left_gripper["per_step"],
                 ) 

             if (right_arm is not None and control_idx < right_arm["position"].shape[0]): 
                 # control right arm
                 self.robot.set_arm_joints(
                     right_arm["position"][control_idx],
                     right_arm["velocity"][control_idx],
                     "right",
                 )

             if right_gripper is not None and control_idx < right_gripper["num_step"]:
                 self.robot.set_gripper(
                     right_gripper["result"][control_idx],
                     "right",
                     right_gripper["per_step"],
                 ) 

             self.scene.step()

             if self.render_freq and control_idx % self.render_freq == 0:
                 self._update_render()
                 if hasattr(self, 'viewer') and self.viewer:
                    self.viewer.render()

             # Capture Frame (Only in Replay Phase)
             if save_freq != None and control_idx % save_freq == 0 and not self.need_plan:
                 self._update_render()
                 self._take_picture()

        # Final Save (Only in Replay Phase)
        if save_freq != None and not self.need_plan:
            self._take_picture()
            
            # --- INJECTED POSITIVE SAMPLING LOGIC START ---
            # Retrieve Config for Current Phase
            phase_cfg = self.sampling_config.get(self.current_phase, self.sampling_config["default"])
            pos_cfg = phase_cfg["pos"]

            if (self.sample_type == 'anchor' and 
                pos_cfg["active"] and 
                hasattr(self, 'segment_states') and 
                len(self.segment_states) > 0):
                
                # We reuse the state at start of take_dense_action
                state_now = self.get_state() # Backup end state
                print(f"Generating Positive Sample for Episode {self.ep_num} at Frame {self.FRAME_IDX}")
                self.sample_pos_from(control_seq, 
                                     n_samples=pos_cfg["n_samples"], 
                                     duration=pos_cfg["duration"])
                self.set_state(state_now) # Restore end state to continue replay
            # ---------------------------------------------

        return True

    def play_once(self):
        """
        Rewritten play_once to support two-stage data generation structure.
        """
        # Determine active arm based on geometry (same as original)
        face_prod = get_face_prod(self.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
        arm_tag = ArmTag("left" if face_prod > 0 else "right")
        self.arm_tag = arm_tag

        # --- Phase 1: Grasp Laptop Details ---
        # If we are in Replay Phase (!need_plan), we set sampling frequency for this segment
        self.current_phase = "grasp"
        # self.sample_interval = self.phase_intervals.get("grasp", 10) # Deprecated by config
        
        # Execute Move: 
        # In Plan Phase, this plans and saves path.
        # In Replay Phase, this reads path, executes, and triggers take_dense_action (w/ sampling).
        self.move(self.grasp_actor(self.laptop, arm_tag=arm_tag, pre_grasp_dis=0.08, contact_point_id=0))

        # --- Phase 2: Rotate Lid Details ---
        # If we are in Replay Phase (!need_plan), adjust sampling frequency
        self.current_phase = "rotate"
        # self.sample_interval = self.phase_intervals.get("rotate", 5) # Deprecated by config
        
        for _ in range(15):
             # Safety Check for Replay Mode: Prevent moving beyond recorded trajectory
            if not self.need_plan:
                 if str(arm_tag) == 'left' and hasattr(self, 'left_joint_path') and self.left_cnt >= len(self.left_joint_path):
                     break
                 if str(arm_tag) == 'right' and hasattr(self, 'right_joint_path') and self.right_cnt >= len(self.right_joint_path):
                     break

            # Get target rotation pose
            self.move(
                self.grasp_actor(
                    self.laptop,
                    arm_tag=arm_tag,
                    pre_grasp_dis=0.0,
                    grasp_dis=0.0,
                    contact_point_id=1,
                ))
            
            # Phase 1 Planning Check: If planning failed, stop
            if self.need_plan and not self.plan_success:
                break
            
            # Success Check
            if self.check_success(target=0.5):
                break

        self.info["info"] = {
            "{A}": f"{self.model_name}/base{self.model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def _take_picture(self):
        """
        Overloaded to support saving anchor/positive/negative samples with start_qpos info.
        This writes pkl files with specific naming convention for Phase 2 data collection.
        Naming Convention:
        - Anchor (Main Trajectory): anchor_{FRAME_IDX}.pkl
        - Negative Branch: neg_branch{anchor_idx}_{step}.pkl
        - Positive Branch: pos_branch{anchor_idx}_{step}.pkl
        """
        if not self.save_data:
            return

        # Initialize cache folder on first anchor frame
        if self.FRAME_IDX == 0 and self.sample_type == 'anchor':
            self.folder_path = {"cache": f"{self.save_dir}/.cache/episode{self.ep_num}/"}
            if not os.path.exists(self.folder_path["cache"]):
                os.makedirs(self.folder_path["cache"])
            else:
                 # Clear previous data only if starting a new positive episode
                for file in os.listdir(self.folder_path["cache"]):
                    os.remove(os.path.join(self.folder_path["cache"], file))

        pkl_dic = self.get_obs()
        
        # Inject Phase 2 specific metadata
        pkl_dic['sample_type'] = self.sample_type
        # Only save start_qpos for branch samples to ensure Anchor consistency
        if self.sample_type == 'anchor':
             pkl_dic['start_qpos'] = []
        else:
             pkl_dic['start_qpos'] = self.start_qpos if self.start_qpos is not None else []
        
        # Determine filename
        filename = ""
        if self.sample_type == 'anchor':
            # Main Trajectory
            filename = f"anchor_{self.FRAME_IDX}.pkl"
            self.FRAME_IDX += 1
        elif self.sample_type == 'negative':
            # Negative samples designated by branch index and step index
            filename = f"neg_branch{self.branch_idx}_{self.neg_step_idx}.pkl"
            self.neg_step_idx += 1
        elif self.sample_type == 'positive':
             # Positive samples branching from anchor
            filename = f"pos_branch{self.branch_idx}_{self.pos_step_idx}.pkl"
            self.pos_step_idx += 1
            
        save_pkl(os.path.join(self.folder_path["cache"], filename), pkl_dic)

    def get_state(self):
        """
        Save current simulation state (robot qpos, object pose, etc.)
        Using SAPIEN's pack function if available or manual
        """
        # For full state restore, we ideally use packing. 
        # But here valid minimal state is: Robot Qpos, Object Pose
        
        robot_qpos = np.concatenate([self.robot.left_entity.get_qpos(), self.robot.right_entity.get_qpos()])
        laptop_pose = self.laptop.actor.get_pose()
        laptop_qpos = self.laptop.actor.get_qpos()
        
        return {
            "robot_qpos": robot_qpos,
            "laptop_pose": laptop_pose,
            "laptop_qpos": laptop_qpos
        }

    def set_state(self, state):
        """
        Restore state
        """
        # Robot
        # Since 'robot.set_qpos' usually needs splitting for left/right entities if separated in logic or combined
        # In robot.py, left_entity and right_entity are loaded.
        
        # We assume splitting logic or just set individually if we stored active joints
        # Let's use the stored structure:
        # Assuming robot structure is static (active joints count doesn't change)
        
        n_left = len(self.robot.left_entity.get_qpos())
        self.robot.left_entity.set_qpos(state["robot_qpos"][:n_left])
        self.robot.right_entity.set_qpos(state["robot_qpos"][n_left:])
        
        # Laptop
        self.laptop.actor.set_pose(state["laptop_pose"])
        self.laptop.actor.set_qpos(state["laptop_qpos"])
        
        # Step physics briefly? No, just set state.
        
    def _extract_arm_qpos(self, full_state_qpos):
        """
        Helper: Extract 7-DoF arm joint positions AND mapping indices from full robot state.
        Returns: start_l_arm, start_r_arm, (n_left_total, l_idxs, r_idxs)
        """
        n_left_total = len(self.robot.left_entity.get_qpos())
        start_qpos_l_full = full_state_qpos[:n_left_total]
        start_qpos_r_full = full_state_qpos[n_left_total:]
        
        # Identify Indices
        l_idxs = [self.robot.left_active_joints.index(j) for j in self.robot.left_arm_joints]
        r_idxs = [self.robot.right_active_joints.index(j) for j in self.robot.right_arm_joints]
        
        # Extract Values
        start_l = start_qpos_l_full[l_idxs] if len(l_idxs) > 0 else np.array([])
        start_r = start_qpos_r_full[r_idxs] if len(r_idxs) > 0 else np.array([])
        
        return start_l, start_r, (n_left_total, l_idxs, r_idxs)

    def _inject_arm_qpos(self, new_l_arm, new_r_arm, full_ref_qpos, mapping_info):
        """
        Helper: Reconstruct full robot state by injecting new arm configurations.
        """
        n_left_total, l_idxs, r_idxs = mapping_info
        
        # Copy original full states as baseline (to keep grippers/other joints unchanged)
        new_qpos_l_full = full_ref_qpos[:n_left_total].copy()
        new_qpos_r_full = full_ref_qpos[n_left_total:].copy()
        
        # Inject new arm positions
        if len(l_idxs) > 0 and len(new_l_arm) > 0: 
            new_qpos_l_full[l_idxs] = new_l_arm
        if len(r_idxs) > 0 and len(new_r_arm) > 0: 
            new_qpos_r_full[r_idxs] = new_r_arm
            
        return np.concatenate([new_qpos_l_full, new_qpos_r_full])

    def sample_pos_from(self, control_seq, n_samples=3, duration=None):
        """
        Generate Positive Samples by reconstructing trajectory backwards from the fixed END state.
        Handles mapping from full robot state (n-DoF) to controlled arm joints (7-DoF).
        """
        if not control_seq or "left_arm" not in control_seq:
            return

        # 1. Parse Control Sequence
        l_pos = control_seq["left_arm"]["position"] if control_seq["left_arm"] is not None else np.zeros((0, 0))
        l_vel = control_seq["left_arm"]["velocity"] if control_seq["left_arm"] is not None else np.zeros((0, 0))
        r_pos = control_seq["right_arm"]["position"] if control_seq["right_arm"] is not None else np.zeros((0, 0))
        r_vel = control_seq["right_arm"]["velocity"] if control_seq["right_arm"] is not None else np.zeros((0, 0))
        
        l_grip = control_seq["left_gripper"]["result"] if control_seq["left_gripper"] is not None else None
        r_grip = control_seq["right_gripper"]["result"] if control_seq["right_gripper"] is not None else None

        if len(l_pos) == 0 and len(r_pos) == 0:
             return
        
        full_seq_len = max(len(l_pos), len(r_pos))
        seq_len = full_seq_len
        
        # Determine actual Start State from history based on duration
        if duration is not None:
             seq_len = min(seq_len, duration)
             # If we only want the LAST 'duration' steps, we slice from the end
             if len(l_pos) > seq_len: l_pos = l_pos[-seq_len:]; l_vel = l_vel[-seq_len:]
             if len(r_pos) > seq_len: r_pos = r_pos[-seq_len:]; r_vel = r_vel[-seq_len:]
        
        # Check history availability
        if not hasattr(self, 'segment_states') or len(self.segment_states) == 0:
             print("Warning: No segment history found for positive sampling.")
             return

        # The 'start state' for this truncated sequence is found in history
        # History index:
        # If full_seq_len = 100, duration = 10. We use actions [90..99].
        # We need state at index 90.
        # index = full_seq_len - seq_len
        # Ensure index is valid
        start_idx = max(0, full_seq_len - seq_len)
        if start_idx >= len(self.segment_states):
             start_idx = len(self.segment_states) - 1 # Fallback
             
        start_state = self.segment_states[start_idx]

        self.branch_idx = self.FRAME_IDX 
        
        # 2. Extract Full Start State and Arm Configs using Helper
        full_qpos_ref = start_state["robot_qpos"]
        start_l_arm, start_r_arm, mapping_info = self._extract_arm_qpos(full_qpos_ref)
        _, l_arm_idxs, r_arm_idxs = mapping_info # Unpack for logic checks
        
        # Handle cases where one arm has no action (tile start pos to match length)
        if len(l_pos) == 0 and len(start_l_arm) > 0: 
            l_pos = np.tile(start_l_arm, (seq_len, 1))
            l_vel = np.zeros_like(l_pos)
        if len(r_pos) == 0 and len(start_r_arm) > 0: 
            r_pos = np.tile(start_r_arm, (seq_len, 1))
            r_vel = np.zeros_like(r_pos)
            
        # 3. Formulate Trajectory for Backwards Reconstruction: [Start, Step1...StepN]
        # qt shape: (N+1, 7) - Prepend Start State to form valid diffable trajectory
        qt_l = np.vstack([start_l_arm, l_pos]) if len(start_l_arm) > 0 else np.zeros((seq_len+1, 0))
        qt_r = np.vstack([start_r_arm, r_pos]) if len(start_r_arm) > 0 else np.zeros((seq_len+1, 0))
        
        # Calculate Deltas: D[t] = Q[t] - Q[t-1]
        deltas_l = qt_l[1:] - qt_l[:-1]
        deltas_r = qt_r[1:] - qt_r[:-1]
        
        # 4. Iterate Samples
        for i in range(n_samples):
            print(f"    Pos Sample {i+1}/{n_samples}: Backwards reconstruction...")
            # Generate Delta Noise
            noise_l = np.random.normal(0, 0.005, deltas_l.shape) if deltas_l.size > 0 else deltas_l
            noise_r = np.random.normal(0, 0.005, deltas_r.shape) if deltas_r.size > 0 else deltas_r
            
            # Reconstruct Trajectory Backwards from Fixed End State
            new_qt_l = np.zeros_like(qt_l)
            new_qt_r = np.zeros_like(qt_r)
            
            if qt_l.size > 0:
                new_qt_l[-1] = qt_l[-1] # Fix End
                for t in range(seq_len - 1, -1, -1):
                    new_qt_l[t] = new_qt_l[t+1] - (deltas_l[t] + noise_l[t])
            
            if qt_r.size > 0:
                new_qt_r[-1] = qt_r[-1] # Fix End
                for t in range(seq_len - 1, -1, -1):
                    new_qt_r[t] = new_qt_r[t+1] - (deltas_r[t] + noise_r[t])
            
            # Extract New Start State (Arm Only)
            new_start_l_arm = new_qt_l[0] if new_qt_l.size > 0 else np.array([])
            new_start_r_arm = new_qt_r[0] if new_qt_r.size > 0 else np.array([])
            
            # 5. Construct Modified Full Robot State for Simulation using Helper
            new_full_start_qpos = self._inject_arm_qpos(
                new_start_l_arm, new_start_r_arm, full_qpos_ref, mapping_info
            )
            
            # Set Simulation to New Start State
            modified_state = start_state.copy()
            modified_state["robot_qpos"] = new_full_start_qpos
            self.set_state(modified_state)
            
            # Update metadata
            self.sample_type = 'positive'
            self.start_qpos = new_full_start_qpos
            self.pos_step_idx = 0
            
            # 6. Execution Loop with Reconstructed Targets
            target_l_seq = new_qt_l[1:]
            target_r_seq = new_qt_r[1:]
            
            for t in range(seq_len):
                # Apply Arm Actions
                if len(l_arm_idxs) > 0:
                    self.robot.set_arm_joints(target_l_seq[t], l_vel[t], "left")
                if len(r_arm_idxs) > 0:
                    self.robot.set_arm_joints(target_r_seq[t], r_vel[t], "right")
                
                # Apply Gripper Actions (Preserve original)
                if l_grip is not None and t < len(l_grip):
                     self.robot.set_gripper(l_grip[t], "left", control_seq["left_gripper"]["per_step"])
                if r_grip is not None and t < len(r_grip):
                     self.robot.set_gripper(r_grip[t], "right", control_seq["right_gripper"]["per_step"])
                     
                self.scene.step()
                
                if self.check_collision():
                    break
                    
                self._update_render()
                self._take_picture()
            
            # Loop continues for next sample...
        
        # Reset to Anchor mode
        self.sample_type = 'anchor'
        self.start_qpos = None

    def sample_neg_from(self, duration=10, branch_idx=0):
        """
        Rollout a negative trajectory and save data via _take_picture.
        
        Args:
            duration: Number of steps to rollout.
            branch_idx: The index of the positive frame where this branch started.
        """
        # Set Phase 2 sampling flags
        self.sample_type = 'negative'
        self.branch_idx = branch_idx
        self.neg_step_idx = 0
        
        # Capture Start Qpos (Anchor State for Contrastive Learning)
        # Assuming robot qpos is sufficient for reference
        self.start_qpos = np.concatenate([
            self.robot.left_entity.get_qpos(),
            self.robot.right_entity.get_qpos()
        ])
        
        # Action Loop for Rollout
        for _ in range(duration):
            # 2. Get current targets/states
            # We use jointState to get n-dof position (excluding gripper)
            curr_qpos_l = self.robot.get_left_arm_jointState()[:-1] 
            curr_qpos_r = self.robot.get_right_arm_jointState()[:-1]

            # 1. Random perturbation (Exploration Noise) - Auto-detect DOF
            dof_l = len(curr_qpos_l)
            dof_r = len(curr_qpos_r)
            noise_l = np.random.normal(0, 0.05, dof_l) 
            noise_r = np.random.normal(0, 0.05, dof_r)
            
            # 3. Apply Noisy Action
            target_l = np.array(curr_qpos_l) + noise_l
            target_r = np.array(curr_qpos_r) + noise_r
            
            self.robot.set_arm_joints(target_l, np.zeros(dof_l), 'left')
            self.robot.set_arm_joints(target_r, np.zeros(dof_r), 'right')
            
            self.scene.step()
            
            # 4. Save Negative Sample Frame
            # Ensure we update the render before taking picture
            self._update_render()
            self._take_picture()
        
        # Reset flags (State restoration is caller's responsibility)
        self.sample_type = 'anchor'
        self.start_qpos = None

    def merge_pkl_to_hdf5_video(self):
        """
        Synthesize .pkl files (positive & negative) into a structured HDF5 and MP4 video.
        Structure:
          - / (Root): Positive Trajectory Data
            - observation/
            ...
          - /negative_trajs/branch_{idx}: Negative Trajectories
            - observation/
            - ...
            - attrs['start_qpos']: Anchor state
        """
        if not self.save_data:
            return

        cache_path = self.folder_path["cache"]
        target_file_path = f"{self.save_dir}/data/episode{self.ep_num}.hdf5"
        target_video_path = f"{self.save_dir}/video/episode{self.ep_num}.mp4"

        os.makedirs(f"{self.save_dir}/data", exist_ok=True)
        os.makedirs(f"{self.save_dir}/video", exist_ok=True)

        print(f"Processing cache to HDF5: {cache_path}")

        # 1. Classify files
        all_files = sorted(os.listdir(cache_path))
        anchor_files = [] # Main Trajectory
        pos_branch_files = {} # Positive Variations
        neg_files = {}  # Negative Variations

        for fname in all_files:
            if not fname.endswith(".pkl"):
                continue
            path = os.path.join(cache_path, fname)
            
            # 1. Anchor (Main)
            if fname.startswith("anchor_"):
                try:
                    idx = int(fname.split('_')[1].split('.')[0])
                    anchor_files.append((idx, path))
                except ValueError:
                    print(f"Skipping malformed anchor: {fname}")

            # 2. Positive Branches
            elif fname.startswith("pos_branch"):
                try:
                    # pos_branch{b_idx}_{step}.pkl
                    parts = fname.replace("pos_branch", "").replace(".pkl", "").split('_')
                    if len(parts) == 2:
                        b_idx = int(parts[0])
                        step_idx = int(parts[1])
                        if b_idx not in pos_branch_files:
                            pos_branch_files[b_idx] = []
                        pos_branch_files[b_idx].append((step_idx, path))
                except ValueError:
                    print(f"Skipping malformed pos branch: {fname}")

            # 3. Negative Branches
            elif fname.startswith("neg_branch"):
                # neg_branch{b_idx}_{step}.pkl
                try:
                    parts = fname.replace("neg_branch", "").replace(".pkl", "").split('_')
                    if len(parts) == 2:
                        b_idx = int(parts[0])
                        step_idx = int(parts[1])
                        if b_idx not in neg_files:
                            neg_files[b_idx] = []
                        neg_files[b_idx].append((step_idx, path))
                except ValueError:
                    print(f"Skipping malformed neg branch: {fname}")
            
            # Legacy fallback (if using generic 'pos_' for anchor)
            elif fname.startswith("pos_") and "branch" not in fname:
                 try:
                    idx = int(fname.split('_')[1].split('.')[0])
                    anchor_files.append((idx, path))
                 except ValueError:
                    pass

        # 2. Process Anchor Trajectory (Main Dataset)
        anchor_files.sort(key=lambda x: x[0])
        sorted_anchor_paths = [x[1] for x in anchor_files]
        
        if not sorted_anchor_paths:
            print("Warning: No anchor trajectory files found!")
            return

        # Initialize Main Data Structure using first frame
        full_data = parse_dict_structure(load_pkl_file(sorted_anchor_paths[0]))
        
        # Aggregate all anchor frames
        for pkl_path in sorted_anchor_paths:
            data = load_pkl_file(pkl_path)
            append_data_to_structure(full_data, data)

        # --- Debug Helper ---
        def debug_check_consistency(data, prefix=""):
            if isinstance(data, dict):
                for k, v in data.items():
                    debug_check_consistency(v, f"{prefix}/{k}")
            elif isinstance(data, list):
                if not data: return
                # Check shapes of elements in the list
                shapes = []
                types = []
                for item in data:
                    if hasattr(item, 'shape'):
                        shapes.append(item.shape)
                    elif isinstance(item, (list, tuple)):
                        shapes.append(len(item))
                    else:
                        shapes.append("scalar")
                    types.append(type(item))

                # Simply check if all shapes match first one
                first_shape = shapes[0]
                is_consistent = all(s == first_shape for s in shapes)
                
                if not is_consistent:
                    print(f"\n[DEBUG] Inconsistency found at Key: {prefix}")
                    print(f"  Total items: {len(data)}")
                    print(f"  First item shape: {first_shape}")
                    print(f"  Inconsistent indices:")
                    for i, s in enumerate(shapes):
                        if s != first_shape:
                            print(f"    - Index {i}: shape {s} (Type: {types[i]})")
                    
                    print("  -> Triggering PDB trace...")
                    import pdb; pdb.set_trace()
        # --------------------

        # 3. Create HDF5 File
        target_file_path = os.path.join(cache_path, "merged_data.h5") # Ensure filename
        print(f"Merging to {target_file_path}...")

        # Debug Anchor
        debug_check_consistency(full_data, "Anchor")

        with h5py.File(target_file_path, "w") as f:
            # Write Anchor Trajectory to Root
            create_hdf5_from_dict(f, full_data)
            
            # Helper to process branches
            def process_branches(branch_dict, group_name):
                if not branch_dict: return
                grp = f.create_group(group_name)
                for b_idx in sorted(branch_dict.keys()):
                    steps = sorted(branch_dict[b_idx], key=lambda x: x[0])
                    step_paths = [x[1] for x in steps]
                    
                    if not step_paths: continue
                    
                    # Initialize & Agg
                    first_frame = load_pkl_file(step_paths[0])
                    b_data = parse_dict_structure(first_frame)
                    for pkl_path in step_paths:
                        d = load_pkl_file(pkl_path)
                        append_data_to_structure(b_data, d)
                    
                    # Debug Branch
                    debug_check_consistency(b_data, f"{group_name}/branch_{b_idx}")

                    # Write Subgroup
                    subgrp = grp.create_group(f"branch_{b_idx}")
                    create_hdf5_from_dict(subgrp, b_data)
                    
                    # Metadata
                    if 'start_qpos' in first_frame:
                         qpos_val = first_frame['start_qpos']
                         if len(qpos_val) > 0:
                             subgrp.attrs['start_qpos'] = qpos_val
                    
                    # Generate Video (Optional, for debugging)
                    try:
                        if "observation" in b_data and "head_camera" in b_data["observation"]:
                            vid_rgb = np.array(b_data["observation"]["head_camera"]["rgb"])
                            prefix = "pos" if "positive" in group_name else "neg"
                            v_path = f"{self.save_dir}/video/episode{self.ep_num}_{prefix}_branch{b_idx}.mp4"
                            images_to_video(vid_rgb, out_path=v_path)
                    except Exception as e:
                        print(f"Error video for {group_name} branch {b_idx}: {e}")

            # Write Negative Trajectories
            process_branches(neg_files, "negative_trajs")
            
            # Write Positive Trajectories (Variations)
            process_branches(pos_branch_files, "positive_trajs")
            
        # 4. Generate Video (Anchor Only)

        # 4. Generate Video (Positive Trajectory Only)
        try:
            if "observation" in full_data and "head_camera" in full_data["observation"]:
                rgb_seq = np.array(full_data["observation"]["head_camera"]["rgb"])
                # images_to_video expects a numpy array of images
                images_to_video(rgb_seq, out_path=target_video_path)
                print(f"Video saved to {target_video_path}")
        except Exception as e:
            print(f"Error creating video: {e}")

class DataProcessor:
    def __init__(self, task_name, root_dir, output_dir):
        self.args = {}
        self.task_name = task_name
        self.task_config = "demo_clean" # Changed from aloha_default
        self.root_dir = root_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize Simulator for Negative Sampling
        self.init_sim()

    def init_sim(self):
        # Setup Simulation Environment (similar to visual_script_plan)
        # We need to load config to setup env correctly
        self.env = OpenLaptopDataGen() # Use our extended class
        
        # Load embodiment config
        config_path = f"./task_config/{self.task_config}.yml"
        with open(config_path, "r", encoding="utf-8") as f:
            self.args = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.args['task_name'] = self.task_name
        self.args['task_config'] = self.task_config
        
        # Enable GUI visualization for Sapien
        self.args['headless'] = False
        # If needed, set render frequency to ensure updates
        # self.args['render_freq'] = 1 

        # Setup embodiment
        embodiment_type = self.args.get("embodiment", ["aloha"]) # Default to aloha if not present
        embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
            
        def get_embodiment_file(etype):
            return _embodiment_types[etype]["file_path"]

        if len(embodiment_type) == 1:
            self.args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            self.args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
            self.args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            self.args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            self.args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
            self.args["embodiment_dis"] = embodiment_type[2]
            self.args["dual_arm_embodied"] = False

        self.args["left_embodiment_config"] = get_embodiment_config(self.args["left_robot_file"])
        self.args["right_embodiment_config"] = get_embodiment_config(self.args["right_robot_file"])
        
        # We need to minimally init the task to load robot
        # Since we are replaying, we need consistent embodiment
        # self.env.setup_demo(...) # This requires args.
        self.args['need_plan'] = False
        self.args['render_freq'] = 0
        self.args['save_data'] = True
        # TODO config save frequency here
        self.args['save_freq'] = 10 # Force save every frame to align FRAME_IDX with control steps
        
        # set new save dir here
        self.args["save_path"] = os.path.join(self.args["save_path"], str(self.args["task_name"]), self.args["task_config"])

        return

    def run_two_stage_collection(self):
        """
        Main Loop for Data Collection
        """

        # Phase 1: Ensure we have seeds/traj (Loaded from disk typically)
        # Assuming args['save_path'] contains _traj_data/episode_X.pkl
        
        clear_cache_freq = self.args["clear_cache_freq"]
        st_idx = 0

        # Phase 2: Replay and Collect
        
        seed_list_path = os.path.join(self.args["save_path"], "seed.txt")
        if not os.path.exists(seed_list_path):
            print(f"No seeds found in {seed_list_path}. Please run collect_data.py first to generate success seeds.")
            return

        with open(seed_list_path, "r") as file:
            seed_list = [int(i) for i in file.read().split()]
        
        # Verify seeds
        # TODO Debug Here
        print(f"Found {len(seed_list)} seeds.")
        
        # NOTE: H5 accumulation removed as it was redundant with merge_pkl_to_hdf5_video
        
        for epid, seed in enumerate(seed_list):
            # TODO Start process a traj
            print(f"Processing Episode {epid} (Seed {seed})")
            
            # 1. Setup Env for Replay
            # We need to force use_seed=True logic
            # args need to be compatible
            self.env.setup_demo(now_ep_num=epid, seed=seed, **self.args)
            
            # Check if cache already exists to skip simulation
            cache_path = f"{self.env.save_dir}/.cache/episode{epid}/"
            is_cached = os.path.exists(os.path.join(cache_path, "pos_0.pkl"))

            if is_cached:
                print(f"Skipping simulation for Episode {epid} (Found existing cache at {cache_path})")
                self.env.folder_path = {"cache": cache_path}
            else:
                # 2. Load Success Trajectory
                # This corresponds to 'Phase 2' in collect_data
                try:
                    traj_data = self.env.load_tran_data(epid)
                except FileNotFoundError:
                    print(f"Trajectory for ep {epid} not found. Skipping.")
                    continue
                
                # Using external tool for analysis (can be called in GDB)
                analyze_trajectory(traj_data, self.env, verbose=True)
                
                # Restore trajectory to env
                self.env.left_joint_path = traj_data['left_joint_path']
                self.env.right_joint_path = traj_data['right_joint_path']
                self.env.left_cnt = 0
                self.env.right_cnt = 0
                
                # Replay variables
                self.env.need_plan = False # IMPORTANT: Disable planner

                # TODO config render freq here
                # self.env.render_freq = 0 # Manual render
                
                # We can use a callback in the env
                self.env.on_step_callback = self.step_handler
                self.env.callback_data = {
                    # "writer": episode_group, # Not used
                }
            
                # Run Replay
                # This will trigger our callback at every step (or sparse steps)
                self.env.play_once()
            
            # merge pkl data to h5 and video
            print(f"Merge pkl to h5 file for episode {epid} and seed {seed}")
            self.env.close_env()
            self.env.merge_pkl_to_hdf5_video()
            assert self.env.check_success(), "Collect Error"

    def step_handler(self, env_instance):
        """
        Called inside simulation loop (e.g. periodically)
        """
        # 1. Record Positive Sample (Image + Current Pose)
        # obs = env_instance.get_obs() ...
        
        # 2. Negative Sampling (Sparse, e.g. every 10 steps)
        # if env_instance.FRAME_IDX % 10 == 0:
        #    state_backup = env_instance.get_state()
        #    neg_traj = env_instance.sample_neg_from(duration=10)
        #    env_instance.set_state(state_backup)
        #    # Save neg_traj
        pass

    def load_hdf5(self, file_path):
        return h5py.File(file_path, 'r')

if __name__ == "__main__":
    # Example Usage
    processor = DataProcessor("open_laptop", "data/open_laptop/data", "data/processed")
    processor.run_two_stage_collection()
    print("Data Processor initialized. Ready to run on HDF5 files.")

