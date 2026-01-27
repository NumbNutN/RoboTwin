
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
        self.sample_type = 'positive' # 'positive' or 'negative'
        self.start_qpos = None
        self.branch_idx = 0
        self.neg_step_idx = 0
        
        # New: Counters and Configs for Phase 2
        # self.sample_interval = 2 # Steps between negative samples
        self.pos_step_counter = 0
        self.phase_intervals = {
            "grasp": 100,
            "rotate": 50
        }

    def set_sample_intervals(self, intervals):
        """
        Set sampling intervals for different phases
        """
        self.phase_intervals.update(intervals)

    def take_dense_action(self, control_seq, save_freq=-1):
        """
        Overridden to inject Negative Sampling logic during execution.
        """
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

        for control_idx in range(max_control_len):

             # --- INJECTED NEGATIVE SAMPLING LOGIC START (Phase 2 Only) ---
             # We only sample negatives if we are NOT planning (need_plan=False) 
             # and we are currently tracking a positive trajectory.
             # Also ensure we are in a data-saving mode (save_freq is not None)
             if save_freq is not None and not self.need_plan and self.sample_type == 'positive':
                 # We simply use the current global frame index or specific counter
                 # Using internal counter to be consistent
                 if self.pos_step_counter % self.sample_interval == 0:
                     # 1. Save Current Good State
                     state_backup = self.get_state()
                     
                     # 2. Rollout Negative Sample (using helper)
                     # Using FRAME_IDX as branch_idx for traceability
                     print(f"Sample Neg Traj at save index {self.FRAME_IDX} at control index {control_idx} for episode {self.ep_num}")
                     self.sample_neg_from(duration=10, branch_idx=self.FRAME_IDX) 
                     
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
        if not self.need_plan:
             self.sample_interval = self.phase_intervals.get("grasp", 10) # High frequency for grasp approach
        
        # Execute Move: 
        # In Plan Phase, this plans and saves path.
        # In Replay Phase, this reads path, executes, and triggers take_dense_action (w/ sampling).
        self.move(self.grasp_actor(self.laptop, arm_tag=arm_tag, pre_grasp_dis=0.08, contact_point_id=0))

        # --- Phase 2: Rotate Lid Details ---
        # If we are in Replay Phase (!need_plan), adjust sampling frequency
        if not self.need_plan:
             self.sample_interval = self.phase_intervals.get("rotate", 5) # Very high frequency for rotation interactions
        
        for _ in range(15):
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
        Overloaded to support saving positive/negative samples with start_qpos info.
        This writes pkl files with specific naming convention for Phase 2 data collection.
        """
        if not self.save_data:
            return

        # Initialize cache folder on first positive frame
        if self.FRAME_IDX == 0 and self.sample_type == 'positive':
            self.folder_path = {"cache": f"{self.save_dir}/.cache/episode{self.ep_num}/"}
            if not os.path.exists(self.folder_path["cache"]):
                os.makedirs(self.folder_path["cache"])
            else:
                 # Clear previous data only if starting a new positive episode
                 # We assume negative samples are generated during the episode
                for file in os.listdir(self.folder_path["cache"]):
                    os.remove(os.path.join(self.folder_path["cache"], file))

        pkl_dic = self.get_obs()
        
        # Inject Phase 2 specific metadata
        pkl_dic['sample_type'] = self.sample_type
        pkl_dic['start_qpos'] = self.start_qpos if self.start_qpos is not None else []
        
        # Determine filename
        filename = ""
        if self.sample_type == 'positive':
            # Maintain compatibility with base generic sorters if possible, but distinct enough
            filename = f"pos_{self.FRAME_IDX}.pkl"
            self.FRAME_IDX += 1
        else:
            # Negative samples designated by branch index and step index
            filename = f"neg_branch{self.branch_idx}_{self.neg_step_idx}.pkl"
            self.neg_step_idx += 1
            
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
        self.sample_type = 'positive'
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
        pos_files = []
        neg_files = {}  # branch_idx -> list of (step_idx, path)

        for fname in all_files:
            if not fname.endswith(".pkl"):
                continue
            path = os.path.join(cache_path, fname)
            
            if fname.startswith("pos_"):
                # pos_{index}.pkl
                try:
                     # e.g., pos_0.pkl
                    idx = int(fname.split('_')[1].split('.')[0])
                    pos_files.append((idx, path))
                except ValueError:
                    print(f"Skipping malformed file: {fname}")

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
                    print(f"Skipping malformed file: {fname}")

        # 2. Process Positive Trajectory (Main Dataset)
        pos_files.sort(key=lambda x: x[0])
        sorted_pos_paths = [x[1] for x in pos_files]
        
        if not sorted_pos_paths:
            print("Warning: No positive trajectory files found!")
            return

        # Initialize Main Data Structure using first frame
        full_data = parse_dict_structure(load_pkl_file(sorted_pos_paths[0]))
        
        # Aggregate all positive frames
        for pkl_path in sorted_pos_paths:
            data = load_pkl_file(pkl_path)
            append_data_to_structure(full_data, data)

        # 3. Create HDF5 File
        with h5py.File(target_file_path, "w") as f:
            # Write Positive Trajectory to Root
            create_hdf5_from_dict(f, full_data)
            
            # Write Negative Trajectories to Subgroups
            if neg_files:
                neg_grp = f.create_group("negative_trajs")
                
                for b_idx in sorted(neg_files.keys()):
                    steps = sorted(neg_files[b_idx], key=lambda x: x[0])
                    step_paths = [x[1] for x in steps]
                    
                    if not step_paths: continue

                    # Initialize Branch Data Structure
                    first_neg_frame = load_pkl_file(step_paths[0])
                    branch_data = parse_dict_structure(first_neg_frame)
                    
                    # Aggregate Branch Frames
                    for pkl_path in step_paths:
                        data = load_pkl_file(pkl_path)
                        append_data_to_structure(branch_data, data)
                    
                    # Create Subgroup
                    branch_subgrp = neg_grp.create_group(f"branch_{b_idx}")
                    create_hdf5_from_dict(branch_subgrp, branch_data)
                    
                    # Save anchor metadata (from first frame of this branch)
                    if 'start_qpos' in first_neg_frame:
                         # Ensure it's stored as attribute or dataset
                         # start_qpos is same for all steps in branch, so taking first is fine
                         # But check if it's empty
                         qpos_val = first_neg_frame['start_qpos']
                         if len(qpos_val) > 0:
                             branch_subgrp.attrs['start_qpos'] = qpos_val

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
        
        # TODO Debug Here
        print(f"Found {len(seed_list)} seeds.")
        
        # Output HDF5 file
        out_h5_path = os.path.join(self.output_dir, "dataset.hdf5")
        
        with h5py.File(out_h5_path, "w") as out_f:
        
            for epid, seed in enumerate(seed_list):
                # TODO Start process a traj
                print(f"Processing Episode {epid} (Seed {seed})")
                
                # 1. Setup Env for Replay
                # We need to force use_seed=True logic
                # args need to be compatible
                self.env.setup_demo(now_ep_num=epid, seed=seed, **self.args)
                
                # 2. Load Success Trajectory
                # This corresponds to 'Phase 2' in collect_data
                try:
                    traj_data = self.env.load_tran_data(epid)
                except FileNotFoundError:
                    print(f"Trajectory for ep {epid} not found. Skipping.")
                    continue
                
                # Using external tool for analysis (can be called in GDB)
                analyze_trajectory(traj_data, self.env)
                # --- Analysis of Trajectory Data ---
                traj_len = sum(seg['position'].shape[0] for seg in traj_data['left_joint_path'])
                save_freq = self.env.save_freq if hasattr(self.env, 'save_freq') and self.env.save_freq else 1
                effective_len = traj_len // save_freq if save_freq > 0 else 0
                
                print(f"[Trajectory Info]")
                print(f"  Total Trajectory Length: {traj_len}")
                print(f"  Save Frequency: {save_freq}")
                print(f"  Effective Record Length: {effective_len}")
                print(f"  Negative Sample Intervals: {getattr(self.env, 'phase_intervals', 'Not Set')}")
                # -----------------------------------

                # --- Debug: Analyze traj_data structure ---
                print("\n[DEBUG] traj_data structure:")
                print(f"  Type: {type(traj_data)}")
                if isinstance(traj_data, dict):
                    print(f"  Keys: {list(traj_data.keys())}")
                    for key, val in traj_data.items():
                        if isinstance(val, list):
                            print(f"    Key '{key}': List (len={len(val)})")
                            if len(val) > 0:
                                elem = val[0]
                                print(f"      Element[0] type: {type(elem)}")
                                if isinstance(elem, dict):
                                     print(f"      Element[0] keys: {list(elem.keys())}")
                                     for sub_k, sub_v in elem.items():
                                         if hasattr(sub_v, 'shape'):
                                             print(f"        SubKey '{sub_k}': {type(sub_v)} shape={sub_v.shape}")
                                         else:
                                             print(f"        SubKey '{sub_k}': {type(sub_v)}")
                                elif hasattr(elem, 'shape'): 
                                    print(f"      Element[0] shape: {elem.shape}")
                        elif hasattr(val, 'shape'):
                            print(f"    Key '{key}': Array (shape={val.shape})")
                        else:
                            print(f"    Key '{key}': {type(val)}")
                print("------------------------------\n")
                # ------------------------------------------
                    
                # Restore trajectory to env
                # self.env.set_path_lst sets self.left_joint_path etc.
                # args need to hold the paths temporarily if using original API
                # But we can manually set it:
                self.env.left_joint_path = traj_data['left_joint_path']
                self.env.right_joint_path = traj_data['right_joint_path']
                self.env.left_cnt = 0
                self.env.right_cnt = 0
                
                # Replay variables
                self.env.need_plan = False # IMPORTANT: Disable planner

                # TODO config render freq here
                # self.env.render_freq = 0 # Manual render
                
                # Replay Loop
                # Instead of standard self.env.play_once(), we iterate step-by-step
                
                # The task structure in open_laptop.play_once is:
                # 1. grasp_actor (move to pre_grasp, move to grasp, close)
                # 2. move (open lid)
                
                # To hook into specific steps, we can overload 'take_dense_action' or break down play_once.
                # Or, simpler: Just execute play_once() but modify 'take_dense_action' to do sampling.
                # However, modifying 'take_dense_action' inside the loop is tricky.
                
                # Better approach:
                # We write a custom replay loop here that mimics 'play_once' logic 
                # OR we instrument 'move' / 'take_dense_action' via the class override.
                
                # Let's override 'take_dense_action' in OpenLaptopDataGen? 
                # Or adds a callback mechanism.
                
                # Sample Collection Storage
                episode_group = out_f.create_group(f"episode_{epid}")
                
                # We need to collect:
                # Images, JointPos
                
                # We can use a callback in the env
                self.env.on_step_callback = self.step_handler
                self.env.callback_data = {
                    "writer": episode_group,
                    "frame_idx": 0,
                    "negatives": []
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

    def segment_trajectory(self, qpos_seq, gripper_seq):
        """
        Segment trajectory based on gripper state
        Phase 1: Approach (Gripper Open -> Start Closing)
        Phase 2: Grasp (Closing -> Fully Closed)
        Phase 3: Manipulation (Closed -> End)
        """
        # Heuristic: Gripper width
        # Open ~ 1.0 (or > 0.8), Closed < 0.1
        
        # Find index where gripper first drops below 0.8 (Start Closing)
        close_start_idx = -1
        for i in range(len(gripper_seq)):
            if gripper_seq[i] < 0.9: # Threshold
                close_start_idx = i
                break
        
        if close_start_idx == -1:
            return [(0, len(qpos_seq), "approach")] # Never closed
            
        # Find index where gripper is fully closed/stable
        close_end_idx = -1
        for i in range(close_start_idx, len(gripper_seq)):
            if gripper_seq[i] < 0.1: # Fully closed
                close_end_idx = i
                break
        
        segments = []
        if close_start_idx > 0:
            segments.append((0, close_start_idx, "approach"))
        
        if close_end_idx > close_start_idx:
            segments.append((close_start_idx, close_end_idx, "grasp"))
            
        if close_end_idx != -1 and close_end_idx < len(qpos_seq):
            segments.append((close_end_idx, len(qpos_seq), "open_laptop"))
            
        return segments

    def generate_negative_sample(self, seed, current_step, current_qpos, phase):
        """
        Reset sim, replay to state, generate bad trajectory
        """
        # 1. Reset Env
        # Note: We need to pass the correct config args here. 
        # Assuming self.env is already configured or we re-configure it.
        # self.env.setup_demo(seed=seed, ...) 
        
        # Since implementation depends heavily on the specific robot/scene config loading 
        # and we don't have the full dataset to replay, we will write the logic 
        # as a function that would be called if the sim loop was active.
        
        # LOGIC:
        # env.setup_demo(seed=seed)
        # Replay actions 0...current_step-1
        # At current_step:
        #   Get Ground Truth Action (delta theta)
        #   Generate Negative Action:
        #      - Opposite direction
        #      - Random large noise
        #      - If phase == 'approach', move away from laptop center?
        #   Rollout Negative Action for N steps
        #   Return Negative Trajectory
        
        # Mock return for now
        simulated_neg_traj = current_qpos + np.random.randn(*current_qpos.shape) * 0.5 
        return simulated_neg_traj

    def process_episode(self, hdf5_path, episode_idx, seed):
         with self.load_hdf5(hdf5_path) as f:
            # Extract Data
            # Note: Structure depends on pkl2hdf5 output. 
            # Assuming standard: observation/qpos/vector or similar
            
            # For this example, let's assume we have:
            # f['observation']['head_camera']['rgb'] (N, H, W, 3)
            # f['joint_action']['vector'] (N, 14) -> [left_arm, left_grip, right_arm, right_grip]
            
            # Check keys
            if 'joint_action' not in f:
                return []
            
            joint_actions = f['joint_action']['vector'][:] # shape (T, 14)
            
            # Separate arm (assuming 7dof + 1 gripper)
            # Layout: Left(7), LG(1), Right(7), RG(1) -> Total 16? 
            # Or from robot.py: "vector" = left_jointstate + right_jointstate
            # left_jointstate = 7 joints + 1 gripper
            # So indices: 0-6 (Left Arm), 7 (Left Grip), 8-14 (Right Arm), 15 (Right Grip)
            
            left_qpos = joint_actions[:, 0:7]
            left_gripper = joint_actions[:, 7]
            
            # Segment
            segments = self.segment_trajectory(left_qpos, left_gripper)
            
            processed_samples = []
            
            for start, end, phase in segments:
                # We want to sample anchor points within this phase
                # For each anchor point t:
                #   Anchor: Image_t
                #   Positive: Future trajectory (deltas) p_t
                #   Negative: Generated bad trajectory bar_p_t
                
                # Sample a few frames from this segment
                sample_indices = range(start, end, 10) # Stride 10
                
                for t in sample_indices:
                    if t + 10 >= len(joint_actions): # Need future horizon
                        continue
                        
                    # 1. Image (Anchor)
                    # img = f['observation']['head_camera']['rgb'][t] # Don't load all to RAM yet, store path/idx
                    
                    # 2. Positive Trajectory (delta theta)
                    # p_t = {theta_t, delta_theta_t+1 ... t+n}
                    horizon = 10
                    future_qpos = left_qpos[t : t+horizon]
                    theta_t = left_qpos[t]
                    
                    # Deltas
                    deltas = future_qpos - theta_t # Relative to start
                    # Or relative steps: d_i = q_i - q_{i-1}
                    # User request: delta_theta_{t+1} - delta_theta_s_{t+n} 
                    # "delta_theta_{t+1}" usually means q_{t+1} - q_t.
                    # Let's use sequence of relative changes.
                    
                    pos_traj = deltas # Shape (Horizon, 7)
                    
                    # 3. Negative Sampling
                    # neg_traj = self.generate_negative_sample(seed, t, future_qpos, phase)
                    # For demo, generating random perturbation as negative
                    # In real sim, use logic described above
                    neg_traj = pos_traj + np.random.normal(0, 0.1, pos_traj.shape)
                    
                    # Store metadata
                    sample = {
                        "episode_path": hdf5_path,
                        "frame_idx": t,
                        "phase": phase,
                        "pos_traj": pos_traj,
                        "neg_traj": neg_traj,
                        # "theta_t": theta_t
                    }
                    processed_samples.append(sample)
            
            return processed_samples

    def run(self):
        all_samples = []
        # Walk through user data dir
        # Mock loop
        # for file in os.listdir(self.root_dir):
        #    if file.endswith('.hdf5'):
        #        samples = self.process_episode(os.path.join(self.root_dir, file), 0, 12345)
        #        all_samples.extend(samples)
        
        print(f"Processed {len(all_samples)} samples.")
        # Save to disk
        # with open(os.path.join(self.output_dir, 'train_data.pkl'), 'wb') as f:
        #    pickle.dump(all_samples, f)

if __name__ == "__main__":
    # Example Usage
    processor = DataProcessor("open_laptop", "data/open_laptop/data", "data/processed")
    processor.run_two_stage_collection()
    # processor.run()
    print("Data Processor initialized. Ready to run on HDF5 files.")

