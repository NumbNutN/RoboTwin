"""
Base class for data generation with positive/negative sampling support.

This module provides a common framework for generating training data with:
- Anchor trajectories (main demonstration)
- Positive samples (alternative successful approaches)
- Negative samples (perturbed trajectories for contrastive learning)

Usage:
    Subclass DataGenBase and implement task-specific methods:
    - define_sampling_points(): Define where in the task to trigger sampling
    - get_grasp_targets(): Return actors and contact points for grasp-based pos sampling
"""

import os
import numpy as np
import pickle
import h5py
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
import transforms3d as t3d

# Import base task utilities
import sys
sys.path.append(os.getcwd())

from envs.utils import ArmTag, get_face_prod
from envs.utils.save_file import save_pkl
from envs.utils.pkl2hdf5 import (
    load_pkl_file,
    parse_dict_structure,
    append_data_to_structure,
    create_hdf5_from_dict,
    images_to_video
)


class SampleType(Enum):
    """Enumeration for sample types."""
    ANCHOR = 'anchor'
    POSITIVE = 'positive'
    NEGATIVE = 'negative'


class SamplingTrigger(Enum):
    """When to trigger sampling."""
    ON_GRASP_COMPLETE = 'on_grasp_complete'  # After gripper closes on object
    ON_PHASE_START = 'on_phase_start'        # At the start of a phase
    ON_PHASE_END = 'on_phase_end'            # At the end of a phase
    ON_MOVE_COMPLETE = 'on_move_complete'    # After a move action completes
    MANUAL = 'manual'                         # Manually triggered


class DataGenBase:
    """
    Base class for data generation with positive/negative sampling.

    This class provides:
    1. State management (save/restore simulation state)
    2. Sample type tracking (anchor/positive/negative)
    3. Configurable sampling triggers
    4. Common data saving infrastructure
    5. Alternative grasp-based positive sampling
    6. Noise-based negative sampling

    Subclasses should:
    1. Call super().__init__() in their __init__
    2. Define sampling configuration via self.sampling_config
    3. Implement task-specific grasp target methods
    """

    def __init__(self):
        """Initialize data generation base class."""
        # Sample tracking
        self.sample_type = SampleType.ANCHOR.value
        self.start_qpos = None
        self.branch_idx = 0
        self.neg_step_idx = 0
        self.pos_step_idx = 0

        # Phase tracking
        self.current_phase = "default"
        self.pos_step_counter = 0

        # Sampling configuration (can be overridden by subclasses)
        self.sampling_config = {
            "default": {
                "neg": {"active": False, "interval": 999, "duration": 10},
                "pos": {"active": False, "n_samples": 0, "duration": 10}
            }
        }

        # Alternative grasp-based positive sampling config
        self.alt_grasp_pos_config = {
            "active": False,
            "n_samples": 3,
            "trigger": SamplingTrigger.ON_GRASP_COMPLETE,
        }

        # Segment state history for sampling
        self.segment_states = []

        # Hooks for sampling triggers
        self._sampling_hooks: Dict[SamplingTrigger, List[Callable]] = {
            trigger: [] for trigger in SamplingTrigger
        }

        # Track grasped actors for positive sampling
        self._grasped_actors = {}  # arm_tag -> actor

    # ==================== State Management ====================

    def get_state(self) -> Dict[str, Any]:
        """
        Save current simulation state.

        Returns:
            Dict containing robot qpos and relevant object states.
        """
        robot_qpos = np.concatenate([
            self.robot.left_entity.get_qpos(),
            self.robot.right_entity.get_qpos()
        ])

        state = {
            "robot_qpos": robot_qpos,
        }

        # Save task-specific object states (to be extended by subclasses)
        state.update(self._get_task_object_states())

        return state

    def set_state(self, state: Dict[str, Any]):
        """
        Restore simulation state.

        Args:
            state: Dict containing saved state from get_state()
        """
        # Restore robot state
        n_left = len(self.robot.left_entity.get_qpos())
        self.robot.left_entity.set_qpos(state["robot_qpos"][:n_left])
        self.robot.right_entity.set_qpos(state["robot_qpos"][n_left:])

        # Restore task-specific object states
        self._set_task_object_states(state)

    def _get_task_object_states(self) -> Dict[str, Any]:
        """
        Get task-specific object states. Override in subclasses.

        Returns:
            Dict containing object poses/states.
        """
        return {}

    def _set_task_object_states(self, state: Dict[str, Any]):
        """
        Set task-specific object states. Override in subclasses.

        Args:
            state: Dict containing saved object states.
        """
        pass

    # ==================== Sampling Triggers ====================

    def register_sampling_hook(self, trigger: SamplingTrigger, callback: Callable):
        """Register a callback for a sampling trigger."""
        self._sampling_hooks[trigger].append(callback)

    def _fire_sampling_trigger(self, trigger: SamplingTrigger, **kwargs):
        """Fire all callbacks for a trigger."""
        for callback in self._sampling_hooks[trigger]:
            callback(**kwargs)

    def on_grasp_complete(self, actor, arm_tag: ArmTag):
        """
        Called when a grasp action completes (gripper closes on object).

        This is a key sampling point for many manipulation tasks.
        """
        self._grasped_actors[str(arm_tag)] = actor
        self._fire_sampling_trigger(
            SamplingTrigger.ON_GRASP_COMPLETE,
            actor=actor,
            arm_tag=arm_tag
        )

        # Trigger positive sampling if configured
        if (self.alt_grasp_pos_config.get("active", False) and
            self.alt_grasp_pos_config.get("trigger") == SamplingTrigger.ON_GRASP_COMPLETE and
            self.sample_type == SampleType.ANCHOR.value):

            self._trigger_pos_sampling_at_grasp(actor, arm_tag)

    def on_phase_change(self, new_phase: str):
        """Called when transitioning to a new task phase."""
        old_phase = self.current_phase
        self.current_phase = new_phase
        self._fire_sampling_trigger(
            SamplingTrigger.ON_PHASE_START,
            old_phase=old_phase,
            new_phase=new_phase
        )

    # ==================== Positive Sampling ====================

    def _trigger_pos_sampling_at_grasp(self, actor, arm_tag: ArmTag):
        """
        Trigger positive sampling after a grasp completes.

        This saves state and schedules positive sample generation
        to happen after the anchor trajectory completes.
        """
        # Save the state right after grasp
        self._grasp_state_for_pos = {
            'state': self.get_state(),
            'actor': actor,
            'arm_tag': arm_tag,
            'frame_idx': self.FRAME_IDX,
        }
        print(f"[DataGen] Saved grasp state for positive sampling at frame {self.FRAME_IDX}")

    def collect_feasible_grasp_poses(self, actor, arm_tag, contact_point_id=0, pre_grasp_dis=0.08):
        """
        Collect all feasible grasp poses for an actor.

        Args:
            actor: The actor to grasp
            arm_tag: Which arm to use
            contact_point_id: Index of contact point on actor
            pre_grasp_dis: Pre-grasp distance

        Returns:
            List of feasible grasp info dicts
        """
        feasible_poses = []

        # Get base grasp pose from contact point
        contact_matrix = actor.get_contact_point(contact_point_id, "matrix")
        if contact_matrix is None:
            return feasible_poses

        # Transform contact point to grasp frame
        global_contact_pose_matrix = contact_matrix @ np.array([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])

        center_pose = actor.get_contact_point(contact_point_id, "list")

        # Generate base pose
        global_contact_pose_matrix_q = global_contact_pose_matrix[:3, :3]
        global_grasp_pose_p = (
            global_contact_pose_matrix[:3, 3] +
            global_contact_pose_matrix_q @ np.array([-0.12 - pre_grasp_dis, 0, 0]).T
        )
        global_grasp_pose_q = t3d.quaternions.mat2quat(global_contact_pose_matrix_q)
        base_pose = list(global_grasp_pose_p) + list(global_grasp_pose_q)

        # Get candidate poses from robot's rotation scheme
        candidate_poses = self.robot.create_target_pose_list(base_pose, center_pose, arm_tag)

        # Add Z-axis rotations for more variety
        from envs.utils import transforms as env_transforms
        z_rotations = [-np.pi/6, -np.pi/12, 0, np.pi/12, np.pi/6]
        extended_candidates = []
        for pose in candidate_poses:
            for z_rot in z_rotations:
                rotated_pose = env_transforms.rotate_along_axis(
                    pose, center_pose, [0, 0, 1], z_rot,
                    axis_type="target", towards=[0, 0, 1]
                )
                extended_candidates.append(rotated_pose)

        # Test each candidate for reachability
        plan_func = (self.robot.left_plan_path if arm_tag == "left"
                     else self.robot.right_plan_path)

        for pre_pose in extended_candidates:
            if pre_pose is None or pre_pose[0] == -1:
                continue

            # Compute actual grasp pose
            grasp_pose = deepcopy(pre_pose)
            grasp_pose = np.array(grasp_pose)
            direction_mat = t3d.quaternions.quat2mat(grasp_pose[-4:])
            grasp_pose[:3] += [pre_grasp_dis, 0, 0] @ np.linalg.inv(direction_mat)
            grasp_pose = grasp_pose.tolist()

            # Test reachability
            pre_result = plan_func(pre_pose)
            if pre_result["status"] != "Success":
                continue

            grasp_result = plan_func(grasp_pose)
            if grasp_result["status"] != "Success":
                continue

            feasible_poses.append({
                'pre_grasp_pose': pre_pose,
                'grasp_pose': grasp_pose,
                'pre_grasp_traj': pre_result,
                'grasp_traj': grasp_result,
            })

        return feasible_poses

    def sample_pos_from_grasp_state(self, n_samples=3):
        """
        Generate positive samples starting from the saved grasp state.

        This uses alternative grasp approaches and continues the task
        from where the grasp completed.
        """
        if not hasattr(self, '_grasp_state_for_pos'):
            print("[DataGen] No grasp state saved for positive sampling")
            return

        grasp_info = self._grasp_state_for_pos
        initial_state = grasp_info['state']
        actor = grasp_info['actor']
        arm_tag = grasp_info['arm_tag']

        print(f"\n=== Generating {n_samples} Positive Samples from Grasp State ===")

        # Collect feasible alternative grasps
        original_need_plan = self.need_plan
        self.need_plan = True

        # Reset to state before grasp to test alternative approaches
        # Actually we need the state before the grasp started
        # For now, use the post-grasp state and vary the continuation

        feasible_grasps = self.collect_feasible_grasp_poses(
            actor, arm_tag,
            contact_point_id=0,
            pre_grasp_dis=0.08
        )

        self.need_plan = original_need_plan

        if len(feasible_grasps) < 1:
            print(f"[DataGen] No feasible grasps found")
            return

        print(f"[DataGen] Found {len(feasible_grasps)} feasible grasp approaches")

        # Generate samples
        np.random.shuffle(feasible_grasps)
        selected = feasible_grasps[:n_samples]

        for idx, grasp in enumerate(selected):
            print(f"\n--- Positive Sample {idx+1}/{n_samples} ---")
            self._generate_pos_sample_with_grasp(
                grasp, actor, arm_tag, initial_state, idx
            )

        # Cleanup
        del self._grasp_state_for_pos

    def _generate_pos_sample_with_grasp(self, grasp_info, actor, arm_tag,
                                         initial_state, sample_idx):
        """
        Generate a single positive sample using the given grasp approach.
        Override in subclasses for task-specific continuation logic.
        """
        # Default implementation - subclasses should override
        print(f"[DataGen] Base implementation - override _generate_pos_sample_with_grasp in subclass")

    # ==================== Negative Sampling ====================

    def sample_neg_from(self, duration=10, branch_idx=0,
                        active_left=True, active_right=True):
        """
        Generate a negative sample by adding noise to robot actions.

        Args:
            duration: Number of steps to rollout
            branch_idx: Index of anchor frame where branch starts
            active_left: Whether to perturb left arm
            active_right: Whether to perturb right arm
        """
        # Set sampling mode
        self.sample_type = SampleType.NEGATIVE.value
        self.branch_idx = branch_idx
        self.neg_step_idx = 0

        # Save start state for contrastive learning
        self.start_qpos = np.concatenate([
            self.robot.left_entity.get_qpos(),
            self.robot.right_entity.get_qpos()
        ])

        # Rollout with noise
        for _ in range(duration):
            if active_left:
                curr_qpos_l = self.robot.get_left_arm_jointState()[:-1]
                dof_l = len(curr_qpos_l)
                noise_l = np.random.normal(0, 0.05, dof_l)
                target_l = np.array(curr_qpos_l) + noise_l
                self.robot.set_arm_joints(target_l, np.zeros(dof_l), 'left')

            if active_right:
                curr_qpos_r = self.robot.get_right_arm_jointState()[:-1]
                dof_r = len(curr_qpos_r)
                noise_r = np.random.normal(0, 0.05, dof_r)
                target_r = np.array(curr_qpos_r) + noise_r
                self.robot.set_arm_joints(target_r, np.zeros(dof_r), 'right')

            self.scene.step()

            self._update_render()
            if hasattr(self, 'viewer') and self.viewer:
                self.viewer.render()
            self._take_picture()

        # Reset to anchor mode
        self.sample_type = SampleType.ANCHOR.value
        self.start_qpos = None

    # ==================== Data Saving ====================

    def _take_picture(self):
        """
        Save current frame data with sample type metadata.
        Override of base class method.
        """
        if not self.save_data:
            return

        # Initialize cache on first anchor frame
        if self.FRAME_IDX == 0 and self.sample_type == SampleType.ANCHOR.value:
            self.folder_path = {"cache": f"{self.save_dir}/.cache/episode{self.ep_num}/"}
            if not os.path.exists(self.folder_path["cache"]):
                os.makedirs(self.folder_path["cache"])
            else:
                for file in os.listdir(self.folder_path["cache"]):
                    os.remove(os.path.join(self.folder_path["cache"], file))

        # Ensure folder exists for positive/negative samples
        if not hasattr(self, 'folder_path') or self.folder_path is None:
            self.folder_path = {"cache": f"{self.save_dir}/.cache/episode{self.ep_num}/"}
            os.makedirs(self.folder_path["cache"], exist_ok=True)

        pkl_dic = self.get_obs()

        # Add sampling metadata
        pkl_dic['sample_type'] = self.sample_type
        pkl_dic['start_qpos'] = (self.start_qpos.tolist()
                                  if self.start_qpos is not None else [])

        # Determine filename based on sample type
        if self.sample_type == SampleType.ANCHOR.value:
            filename = f"anchor_{self.FRAME_IDX}.pkl"
            self.FRAME_IDX += 1
        elif self.sample_type == SampleType.NEGATIVE.value:
            filename = f"neg_branch{self.branch_idx}_{self.neg_step_idx}.pkl"
            self.neg_step_idx += 1
        elif self.sample_type == SampleType.POSITIVE.value:
            filename = f"pos_branch{self.branch_idx}_{self.pos_step_idx}.pkl"
            self.pos_step_idx += 1
        else:
            filename = f"unknown_{self.FRAME_IDX}.pkl"
            self.FRAME_IDX += 1

        save_pkl(os.path.join(self.folder_path["cache"], filename), pkl_dic)

    def merge_pkl_to_hdf5_video(self):
        """
        Merge collected pkl files into HDF5 and video.
        """
        if not self.save_data:
            return

        cache_path = self.folder_path["cache"]
        target_file_path = f"{self.save_dir}/data/episode{self.ep_num}.hdf5"
        target_video_path = f"{self.save_dir}/video/episode{self.ep_num}.mp4"

        os.makedirs(f"{self.save_dir}/data", exist_ok=True)
        os.makedirs(f"{self.save_dir}/video", exist_ok=True)

        print(f"[DataGen] Processing cache to HDF5: {cache_path}")

        # Classify files
        all_files = sorted(os.listdir(cache_path))
        anchor_files = []
        pos_branch_files = {}
        neg_files = {}

        for fname in all_files:
            if not fname.endswith(".pkl"):
                continue
            path = os.path.join(cache_path, fname)

            if fname.startswith("anchor_"):
                try:
                    idx = int(fname.split('_')[1].split('.')[0])
                    anchor_files.append((idx, path))
                except ValueError:
                    pass
            elif fname.startswith("pos_branch"):
                try:
                    parts = fname.replace("pos_branch", "").replace(".pkl", "").split('_')
                    if len(parts) == 2:
                        b_idx, step_idx = int(parts[0]), int(parts[1])
                        if b_idx not in pos_branch_files:
                            pos_branch_files[b_idx] = []
                        pos_branch_files[b_idx].append((step_idx, path))
                except ValueError:
                    pass
            elif fname.startswith("neg_branch"):
                try:
                    parts = fname.replace("neg_branch", "").replace(".pkl", "").split('_')
                    if len(parts) == 2:
                        b_idx, step_idx = int(parts[0]), int(parts[1])
                        if b_idx not in neg_files:
                            neg_files[b_idx] = []
                        neg_files[b_idx].append((step_idx, path))
                except ValueError:
                    pass

        # Process anchor trajectory
        anchor_files.sort(key=lambda x: x[0])
        sorted_anchor_paths = [x[1] for x in anchor_files]

        if not sorted_anchor_paths:
            print("[DataGen] Warning: No anchor trajectory files found!")
            return

        # Build data structure
        full_data = parse_dict_structure(load_pkl_file(sorted_anchor_paths[0]))
        for pkl_path in sorted_anchor_paths:
            data = load_pkl_file(pkl_path)
            append_data_to_structure(full_data, data)

        # Create HDF5
        target_file_path = os.path.join(cache_path, "merged_data.h5")
        print(f"[DataGen] Merging to {target_file_path}...")

        with h5py.File(target_file_path, "w") as f:
            create_hdf5_from_dict(f, full_data)

            # Process branches
            def process_branches(branch_dict, group_name):
                if not branch_dict:
                    return
                grp = f.create_group(group_name)
                for b_idx in sorted(branch_dict.keys()):
                    steps = sorted(branch_dict[b_idx], key=lambda x: x[0])
                    step_paths = [x[1] for x in steps]
                    if not step_paths:
                        continue

                    first_frame = load_pkl_file(step_paths[0])
                    b_data = parse_dict_structure(first_frame)
                    for pkl_path in step_paths:
                        d = load_pkl_file(pkl_path)
                        append_data_to_structure(b_data, d)

                    subgrp = grp.create_group(f"branch_{b_idx}")
                    create_hdf5_from_dict(subgrp, b_data)

                    if 'start_qpos' in first_frame and len(first_frame['start_qpos']) > 0:
                        subgrp.attrs['start_qpos'] = first_frame['start_qpos']

            process_branches(neg_files, "negative_trajs")
            process_branches(pos_branch_files, "positive_trajs")

        # Generate videos
        try:
            # Anchor video
            if "observation" in full_data and "head_camera" in full_data["observation"]:
                rgb_seq = np.array(full_data["observation"]["head_camera"]["rgb"])
                images_to_video(rgb_seq, out_path=target_video_path)
                print(f"[DataGen] Anchor video saved to {target_video_path}")

            # Positive branch videos
            for b_idx in sorted(pos_branch_files.keys()):
                steps = sorted(pos_branch_files[b_idx], key=lambda x: x[0])
                step_paths = [x[1] for x in steps]
                if not step_paths:
                    continue
                first_frame = load_pkl_file(step_paths[0])
                b_data = parse_dict_structure(first_frame)
                for pkl_path in step_paths:
                    d = load_pkl_file(pkl_path)
                    append_data_to_structure(b_data, d)
                if "observation" in b_data and "head_camera" in b_data["observation"]:
                    pos_video_path = target_video_path.replace(
                        ".mp4", f"_pos_branch{b_idx}.mp4"
                    )
                    rgb_seq = np.array(b_data["observation"]["head_camera"]["rgb"])
                    images_to_video(rgb_seq, out_path=pos_video_path)
                    print(f"[DataGen] Positive branch {b_idx} video saved to {pos_video_path}")

            # Negative branch videos
            for b_idx in sorted(neg_files.keys()):
                steps = sorted(neg_files[b_idx], key=lambda x: x[0])
                step_paths = [x[1] for x in steps]
                if not step_paths:
                    continue
                first_frame = load_pkl_file(step_paths[0])
                b_data = parse_dict_structure(first_frame)
                for pkl_path in step_paths:
                    d = load_pkl_file(pkl_path)
                    append_data_to_structure(b_data, d)
                if "observation" in b_data and "head_camera" in b_data["observation"]:
                    neg_video_path = target_video_path.replace(
                        ".mp4", f"_neg_branch{b_idx}.mp4"
                    )
                    rgb_seq = np.array(b_data["observation"]["head_camera"]["rgb"])
                    images_to_video(rgb_seq, out_path=neg_video_path)
                    print(f"[DataGen] Negative branch {b_idx} video saved to {neg_video_path}")
        except Exception as e:
            print(f"[DataGen] Error creating video: {e}")

    # ==================== Helper Methods ====================

    def _extract_arm_qpos(self, full_state_qpos):
        """Extract arm joint positions from full robot state."""
        n_left_total = len(self.robot.left_entity.get_qpos())
        start_qpos_l_full = full_state_qpos[:n_left_total]
        start_qpos_r_full = full_state_qpos[n_left_total:]

        l_idxs = [self.robot.left_active_joints.index(j)
                  for j in self.robot.left_arm_joints]
        r_idxs = [self.robot.right_active_joints.index(j)
                  for j in self.robot.right_arm_joints]

        start_l = start_qpos_l_full[l_idxs] if l_idxs else np.array([])
        start_r = start_qpos_r_full[r_idxs] if r_idxs else np.array([])

        return start_l, start_r, (n_left_total, l_idxs, r_idxs)

    def _inject_arm_qpos(self, new_l_arm, new_r_arm, full_ref_qpos, mapping_info):
        """Inject arm positions into full robot state."""
        n_left_total, l_idxs, r_idxs = mapping_info

        new_qpos_l_full = full_ref_qpos[:n_left_total].copy()
        new_qpos_r_full = full_ref_qpos[n_left_total:].copy()

        if l_idxs and len(new_l_arm) > 0:
            new_qpos_l_full[l_idxs] = new_l_arm
        if r_idxs and len(new_r_arm) > 0:
            new_qpos_r_full[r_idxs] = new_r_arm

        return np.concatenate([new_qpos_l_full, new_qpos_r_full])

    def check_collision(self):
        """Check for unwanted collisions. Override for task-specific logic."""
        return False
