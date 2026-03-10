"""
Data generation for place_bread_basket task with positive/negative sampling.

This module provides:
- PlaceBreadBasketDataGen: Extended environment for data collection
- DataProcessor: Main data collection pipeline

Positive sampling strategy:
- Triggered AFTER the gripper grasps the bread
- Generates alternative trajectories for the "lift and place" phase
- Varies the placement approach direction

Negative sampling strategy:
- Injects noise during the placement phase
"""

import os
import h5py
import numpy as np
import pickle
from tqdm import tqdm
import sys
import yaml
from copy import deepcopy

sys.path.append(os.getcwd())

from envs.place_bread_basket import place_bread_basket
from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs.utils import ArmTag, get_face_prod
from envs.utils.save_file import save_pkl
from envs.utils.pkl2hdf5 import (
    load_pkl_file,
    parse_dict_structure,
    append_data_to_structure,
    create_hdf5_from_dict,
    images_to_video
)

from .data_gen_base import DataGenBase, SampleType, SamplingTrigger


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


class PlaceBreadBasketDataGen(place_bread_basket, DataGenBase):
    """
    Extended place_bread_basket environment for data generation with sampling.

    Key differences from base class:
    1. Tracks grasp completion for positive sampling triggers
    2. Overrides take_dense_action for negative sampling injection
    3. Implements alternative grasp-based positive sampling
    """

    def __init__(self):
        place_bread_basket.__init__(self)
        DataGenBase.__init__(self)

        # Task-specific sampling configuration
        self.sampling_config = {
            "grasp": {
                "neg": {"active": False, "interval": 100, "duration": 50},
                "pos": {"active": False, "n_samples": 0, "duration": 50}
            },
            "lift": {
                "neg": {"active": False, "interval": 200, "duration": 50},
                "pos": {"active": False, "n_samples": 0, "duration": 50}
            },
            "place": {
                "neg": {"active": True, "interval": 150, "duration": 50},
                "pos": {"active": False, "n_samples": 0, "duration": 50}
            },
            "default": {
                "neg": {"active": False, "interval": 999, "duration": 10},
                "pos": {"active": False, "n_samples": 0, "duration": 10}
            }
        }

        # Alternative grasp-based positive sampling
        # Triggered after grasp completes
        self.alt_grasp_pos_config = {
            "active": True,
            "n_samples": 2,
            "trigger": SamplingTrigger.ON_GRASP_COMPLETE,
        }

        # Track which bread is being handled
        self._current_bread_idx = None
        self._grasp_states_for_pos = []  # Can have multiple grasps (dual arm)

    def setup_demo(self, **kwargs):
        """Override to re-apply sampling config after parent init."""
        super().setup_demo(**kwargs)
        # _init_task_env_ calls super().__init__() which re-runs DataGenBase.__init__
        # and resets configs to defaults. Re-apply our task-specific config here.
        self.sampling_config = {
            "grasp": {
                "neg": {"active": False, "interval": 100, "duration": 50},
                "pos": {"active": False, "n_samples": 0, "duration": 50}
            },
            "lift": {
                "neg": {"active": False, "interval": 200, "duration": 50},
                "pos": {"active": False, "n_samples": 0, "duration": 50}
            },
            "place": {
                "neg": {"active": False, "interval": 150, "duration": 50},
                "pos": {"active": False, "n_samples": 0, "duration": 50}
            },
            "default": {
                "neg": {"active": False, "interval": 999, "duration": 10},
                "pos": {"active": False, "n_samples": 0, "duration": 10}
            }
        }
        self.alt_grasp_pos_config = {
            "active": True,
            "n_samples": 2,
            "trigger": SamplingTrigger.ON_GRASP_COMPLETE,
        }
        self._current_bread_idx = None
        self._grasp_states_for_pos = []

    def merge_pkl_to_hdf5_video(self):
        """Use DataGenBase's version which handles pos/neg branch videos."""
        DataGenBase.merge_pkl_to_hdf5_video(self)

    def _take_picture(self):
        return DataGenBase._take_picture(self)

    def _get_task_object_states(self):
        """Save bread and basket states."""
        states = {
            "breadbasket_pose": self.breadbasket.get_pose(),
            "bread_poses": [b.get_pose() for b in self.bread],
        }
        return states

    def _set_task_object_states(self, state):
        """Restore bread and basket states via underlying SAPIEN Entity."""
        if "breadbasket_pose" in state:
            self.breadbasket.actor.set_pose(state["breadbasket_pose"])
        if "bread_poses" in state:
            for i, pose in enumerate(state["bread_poses"]):
                if i < len(self.bread):
                    self.bread[i].actor.set_pose(pose)

    def take_dense_action(self, control_seq, save_freq=-1):
        """
        Override to inject sampling logic during execution.
        """
        if self.sample_type == SampleType.ANCHOR.value:
            if control_seq.get('left_arm') is not None and 'position' in control_seq['left_arm']:
                print(f"[PlaceBread] Executing at frame {self.FRAME_IDX}, "
                      f"control len {len(control_seq['left_arm']['position'])}")

        # Unpack control sequence
        left_arm = control_seq.get("left_arm")
        left_gripper = control_seq.get("left_gripper")
        right_arm = control_seq.get("right_arm")
        right_gripper = control_seq.get("right_gripper")

        save_freq = self.save_freq if save_freq == -1 else save_freq

        # Initial save
        if save_freq is not None and not self.need_plan:
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

        # Reset segment history
        self.segment_states = []

        for control_idx in range(max_control_len):
            # Record state for sampling
            if save_freq is not None and not self.need_plan and self.sample_type == SampleType.ANCHOR.value:
                self.segment_states.append(self.get_state())

            # Negative sampling injection
            phase_cfg = self.sampling_config.get(self.current_phase, self.sampling_config["default"])
            neg_cfg = phase_cfg["neg"]

            if (save_freq is not None and
                not self.need_plan and
                self.sample_type == SampleType.ANCHOR.value and
                neg_cfg["active"]):

                if self.pos_step_counter % neg_cfg["interval"] == 0 and self.pos_step_counter > 0:
                    state_backup = self.get_state()

                    print(f"[PlaceBread] Generating Negative Sample at Frame {self.FRAME_IDX}")
                    self.sample_neg_from(
                        duration=neg_cfg["duration"],
                        branch_idx=self.FRAME_IDX,
                        active_left=(left_arm is not None),
                        active_right=(right_arm is not None)
                    )

                    self.set_state(state_backup)
                    print(f"[PlaceBread] Restored to anchor at Frame {self.FRAME_IDX}")

                self.pos_step_counter += 1

            # Execute controls
            if left_arm is not None and control_idx < left_arm["position"].shape[0]:
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

            if right_arm is not None and control_idx < right_arm["position"].shape[0]:
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

            if save_freq is not None and control_idx % save_freq == 0 and not self.need_plan:
                self._update_render()
                self._take_picture()

        # Final save
        if save_freq is not None and not self.need_plan:
            self._take_picture()

        return True

    def play_once(self):
        """
        Override play_once to add grasp completion hooks.

        The task structure:
        1. Grasp bread(s) - TRIGGER POINT for positive sampling
        2. Lift
        3. Place into basket
        """

        def remove_bread_with_hook(bread_idx, num):
            """Modified remove_bread that triggers grasp hook."""
            arm_tag = ArmTag("right" if self.bread[bread_idx].get_pose().p[0] > 0 else "left")
            self._current_bread_idx = bread_idx

            # Phase: Grasp
            self.current_phase = "grasp"

            # Grasp the bread
            self.move(self.grasp_actor(self.bread[bread_idx], arm_tag=arm_tag, pre_grasp_dis=0.07))

            # *** TRIGGER: Grasp Complete ***
            # Save state for positive sampling AFTER grasp
            if (not self.need_plan and
                self.sample_type == SampleType.ANCHOR.value and
                self.alt_grasp_pos_config.get("active", False)):

                self._grasp_states_for_pos.append({
                    'state': self.get_state(),
                    'bread_idx': bread_idx,
                    'arm_tag': str(arm_tag),
                    'frame_idx': self.FRAME_IDX,
                    'remaining_actions': num,  # 0 = first bread, 1 = second
                })
                print(f"[PlaceBread] Saved grasp state for bread {bread_idx} at frame {self.FRAME_IDX}")

            # Phase: Lift
            self.current_phase = "lift"
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))

            # Phase: Place
            self.current_phase = "place"
            breadbasket_pose = self.breadbasket.get_functional_point(0)
            self.move(
                self.place_actor(
                    self.bread[bread_idx],
                    arm_tag=arm_tag,
                    target_pose=breadbasket_pose,
                    constrain="free",
                    pre_dis=0.12,
                ))

            if num == 0:
                self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.15, move_axis="arm"))
            else:
                self.move(self.open_gripper(arm_tag=arm_tag))

        def remove_dual_with_hook():
            """Modified dual-arm removal with grasp hooks."""
            id = 0 if self.bread[0].get_pose().p[0] < 0 else 1

            # Phase: Grasp (dual)
            self.current_phase = "grasp"

            self.move(
                self.grasp_actor(self.bread[id], arm_tag="left", pre_grasp_dis=0.05),
                self.grasp_actor(self.bread[id ^ 1], arm_tag="right", pre_grasp_dis=0.07),
            )

            # *** TRIGGER: Dual Grasp Complete ***
            if (not self.need_plan and
                self.sample_type == SampleType.ANCHOR.value and
                self.alt_grasp_pos_config.get("active", False)):

                self._grasp_states_for_pos.append({
                    'state': self.get_state(),
                    'bread_idx': id,  # Left arm bread
                    'arm_tag': 'left',
                    'frame_idx': self.FRAME_IDX,
                    'dual_grasp': True,
                    'other_bread_idx': id ^ 1,
                })
                print(f"[PlaceBread] Saved dual-grasp state at frame {self.FRAME_IDX}")

            # Phase: Lift
            self.current_phase = "lift"
            self.move(
                self.move_by_displacement(arm_tag="left", z=0.05, move_axis="arm"),
                self.move_by_displacement(arm_tag="right", z=0.05, move_axis="arm"),
            )

            # Phase: Place
            self.current_phase = "place"
            breadbasket_pose = self.breadbasket.get_functional_point(0)

            self.move(
                self.place_actor(
                    self.bread[id],
                    arm_tag="left",
                    target_pose=breadbasket_pose,
                    constrain="free",
                    pre_dis=0.13,
                ))

            self.move(self.move_by_displacement(arm_tag="left", z=0.1, move_axis="arm"))

            self.move(
                self.back_to_origin(arm_tag="left"),
                self.place_actor(
                    self.bread[id ^ 1],
                    arm_tag="right",
                    target_pose=breadbasket_pose,
                    constrain="free",
                    pre_dis=0.13,
                    dis=0.05,
                ),
            )

        # Clear previous grasp states
        self._grasp_states_for_pos = []

        arm_info = None
        if len(self.bread) <= 1 or (self.bread[0].get_pose().p[0] * self.bread[1].get_pose().p[0]) > 0:
            if len(self.bread) == 1:
                remove_bread_with_hook(0, 0)
                arm_info = "left" if self.bread[0].get_pose().p[0] < 0 else "right"
            else:
                id = 0 if self.bread[0].get_pose().p[1] < self.bread[1].get_pose().p[1] else 1
                arm_info = "left" if self.bread[0].get_pose().p[0] < 0 else "right"
                remove_bread_with_hook(id, 0)
                remove_bread_with_hook(id ^ 1, 1)
        else:
            remove_dual_with_hook()
            arm_info = "dual"

        self.info["info"] = {
            "{A}": f"076_breadbasket/base{self.basket_id}",
            "{B}": f"075_bread/base{self.bread_id[0]}",
            "{a}": arm_info,
        }
        if len(self.bread) == 2:
            self.info["info"]["{C}"] = f"075_bread/base{self.bread_id[1]}"

        return self.info

    def sample_pos_from_grasp_states(self, n_samples_per_grasp=2):
        """
        Generate positive samples from saved grasp states.

        For each saved grasp state, generates alternative trajectories
        for the lift-and-place phase.
        """
        if not self._grasp_states_for_pos:
            print("[PlaceBread] No grasp states saved for positive sampling")
            return

        print(f"\n=== Generating Positive Samples from {len(self._grasp_states_for_pos)} Grasp States ===")

        for grasp_idx, grasp_info in enumerate(self._grasp_states_for_pos):
            print(f"\n--- Processing Grasp State {grasp_idx + 1} ---")

            # Restore to grasp-complete state
            initial_state = grasp_info['state']
            arm_tag = ArmTag(grasp_info['arm_tag'])
            bread_idx = grasp_info['bread_idx']

            for sample_idx in range(n_samples_per_grasp):
                print(f"  Positive Sample {sample_idx + 1}/{n_samples_per_grasp}")

                # Reset to grasp state
                self.set_state(initial_state)

                # Reset counters
                self.left_joint_path = []
                self.right_joint_path = []
                self.left_cnt = 0
                self.right_cnt = 0

                # Set positive sample mode
                self.sample_type = SampleType.POSITIVE.value
                self.branch_idx = grasp_info['frame_idx'] * 100 + sample_idx
                self.pos_step_idx = 0
                self.start_qpos = np.concatenate([
                    self.robot.left_entity.get_qpos(),
                    self.robot.right_entity.get_qpos()
                ])

                # Execute alternative lift-and-place with variation
                success = self._execute_alternative_place(
                    bread_idx, arm_tag, sample_idx
                )

                if success:
                    print(f"    Sample {sample_idx + 1} completed successfully")
                else:
                    print(f"    Sample {sample_idx + 1} failed")

        # Reset to anchor mode
        self.sample_type = SampleType.ANCHOR.value
        self.start_qpos = None
        self._grasp_states_for_pos = []

    def _execute_alternative_place(self, bread_idx, arm_tag, variation_idx):
        """
        Execute lift-and-place with variation for positive sampling.

        Variations:
        - Different lift heights
        - Different approach angles to basket
        """
        # Variation parameters
        lift_heights = [0.08, 0.12, 0.15]
        pre_dis_values = [0.10, 0.14, 0.16]

        lift_z = lift_heights[variation_idx % len(lift_heights)]
        pre_dis = pre_dis_values[variation_idx % len(pre_dis_values)]

        # Phase: Lift (with variation)
        self.need_plan = True

        if str(arm_tag) == "left":
            origin_pose = np.array(self.robot.get_left_ee_pose(), dtype=np.float64)
        else:
            origin_pose = np.array(self.robot.get_right_ee_pose(), dtype=np.float64)

        # Add small random offset to lift direction
        lift_offset = np.random.uniform(-0.02, 0.02, 2)
        target_pose = origin_pose.copy()
        target_pose[0] += lift_offset[0]
        target_pose[1] += lift_offset[1]
        target_pose[2] += lift_z

        if str(arm_tag) == "left":
            lift_result = self.robot.left_plan_path(target_pose.tolist())
        else:
            lift_result = self.robot.right_plan_path(target_pose.tolist())

        if lift_result["status"] != "Success":
            return False

        # Execute lift
        self.need_plan = False
        control_seq = {
            "left_arm": lift_result if str(arm_tag) == "left" else None,
            "left_gripper": None,
            "right_arm": lift_result if str(arm_tag) == "right" else None,
            "right_gripper": None,
        }
        self._execute_control_seq_for_pos(control_seq)

        # Phase: Place (with variation)
        self.need_plan = True
        breadbasket_pose = self.breadbasket.get_functional_point(0)

        # Get place pose with variation
        place_pre_pose = self.get_place_pose(
            self.bread[bread_idx],
            arm_tag,
            breadbasket_pose,
            pre_dis=pre_dis,
            constrain="free",
        )

        place_pose = self.get_place_pose(
            self.bread[bread_idx],
            arm_tag,
            breadbasket_pose,
            pre_dis=0.02,
            constrain="free",
        )

        if str(arm_tag) == "left":
            pre_result = self.robot.left_plan_path(place_pre_pose)
            place_result = self.robot.left_plan_path(place_pose)
        else:
            pre_result = self.robot.right_plan_path(place_pre_pose)
            place_result = self.robot.right_plan_path(place_pose)

        if pre_result["status"] != "Success":
            return False

        # Execute pre-place
        self.need_plan = False
        control_seq = {
            "left_arm": pre_result if str(arm_tag) == "left" else None,
            "left_gripper": None,
            "right_arm": pre_result if str(arm_tag) == "right" else None,
            "right_gripper": None,
        }
        self._execute_control_seq_for_pos(control_seq)

        # Execute place
        if place_result["status"] == "Success":
            control_seq = {
                "left_arm": place_result if str(arm_tag) == "left" else None,
                "left_gripper": None,
                "right_arm": place_result if str(arm_tag) == "right" else None,
                "right_gripper": None,
            }
            self._execute_control_seq_for_pos(control_seq)

        # Open gripper
        if str(arm_tag) == "left":
            gripper_result = self.set_gripper(left_pos=1.0, set_tag="left")
            control_seq = {
                "left_arm": None,
                "left_gripper": gripper_result,
                "right_arm": None,
                "right_gripper": None,
            }
        else:
            gripper_result = self.set_gripper(right_pos=1.0, set_tag="right")
            control_seq = {
                "left_arm": None,
                "left_gripper": None,
                "right_arm": None,
                "right_gripper": gripper_result,
            }
        self._execute_control_seq_for_pos(control_seq)

        return True

    def _execute_control_seq_for_pos(self, control_seq, save_freq=-1):
        """Execute control sequence and save frames for positive samples."""
        left_arm = control_seq.get("left_arm")
        left_gripper = control_seq.get("left_gripper")
        right_arm = control_seq.get("right_arm")
        right_gripper = control_seq.get("right_gripper")

        save_freq = self.save_freq if save_freq == -1 else save_freq

        max_control_len = 0
        if left_arm is not None and "position" in left_arm:
            max_control_len = max(max_control_len, left_arm["position"].shape[0])
        if left_gripper is not None:
            max_control_len = max(max_control_len, left_gripper["num_step"])
        if right_arm is not None and "position" in right_arm:
            max_control_len = max(max_control_len, right_arm["position"].shape[0])
        if right_gripper is not None:
            max_control_len = max(max_control_len, right_gripper["num_step"])

        for control_idx in range(max_control_len):
            if left_arm is not None and control_idx < left_arm["position"].shape[0]:
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

            if right_arm is not None and control_idx < right_arm["position"].shape[0]:
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

            if save_freq is not None and control_idx % save_freq == 0:
                self._update_render()
                self._take_picture()

        if save_freq is not None:
            self._take_picture()


class DataProcessor:
    """Data collection pipeline for place_bread_basket task."""

    def __init__(self, task_name, root_dir, output_dir):
        self.args = {}
        self.task_name = task_name
        self.task_config = "demo_clean"
        self.root_dir = root_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.init_sim()

    def init_sim(self):
        self.env = PlaceBreadBasketDataGen()

        config_path = f"./task_config/{self.task_config}.yml"
        with open(config_path, "r", encoding="utf-8") as f:
            self.args = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.args['task_name'] = self.task_name
        self.args['task_config'] = self.task_config
        self.args['headless'] = False

        embodiment_type = self.args.get("embodiment", ["aloha"])
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

        self.args['need_plan'] = False
        self.args['render_freq'] = 10
        self.args['save_data'] = True
        self.args['save_freq'] = 10

        self.args["save_path"] = os.path.join(
            self.args["save_path"],
            str(self.args["task_name"]),
            self.args["task_config"]
        )

    def run_two_stage_collection(self):
        """Main data collection loop."""
        seed_list_path = os.path.join(self.args["save_path"], "seed.txt")
        if not os.path.exists(seed_list_path):
            print(f"No seeds found in {seed_list_path}. Run collect_data.py first.")
            return

        with open(seed_list_path, "r") as file:
            seed_list = [int(i) for i in file.read().split()]

        print(f"Found {len(seed_list)} seeds.")

        for epid, seed in enumerate(seed_list):
            print(f"\n=== Processing Episode {epid} (Seed {seed}) ===")

            self.env.setup_demo(now_ep_num=epid, seed=seed, **self.args)

            cache_path = f"{self.env.save_dir}/.cache/episode{epid}/"
            is_cached = os.path.exists(os.path.join(cache_path, "anchor_0.pkl"))

            if is_cached:
                print(f"Skipping simulation for Episode {epid} (Found cache)")
                self.env.folder_path = {"cache": cache_path}
            else:
                try:
                    traj_data = self.env.load_tran_data(epid)
                except FileNotFoundError:
                    print(f"Trajectory for ep {epid} not found. Skipping.")
                    continue

                self.env.left_joint_path = traj_data['left_joint_path']
                self.env.right_joint_path = traj_data['right_joint_path']
                self.env.left_cnt = 0
                self.env.right_cnt = 0
                self.env.need_plan = False

                # Run anchor trajectory
                self.env.play_once()

                # Generate positive samples from grasp states
                if (self.env.alt_grasp_pos_config.get("active", False) and
                    self.env.check_success()):
                    print(f"\n--- Generating Positive Samples ---")
                    n_samples = self.env.alt_grasp_pos_config.get("n_samples", 2)
                    self.env.sample_pos_from_grasp_states(n_samples_per_grasp=n_samples)
                    print(f"--- Positive Sampling Complete ---\n")

            # Merge and save
            print(f"Merging to HDF5 for episode {epid}")
            self.env.close_env()
            self.env.merge_pkl_to_hdf5_video()
            breakpoint()

            if not self.env.check_success():
                print(f"Warning: Episode {epid} did not succeed!")


if __name__ == "__main__":
    processor = DataProcessor("place_bread_basket", "data/place_bread_basket/data", "data/processed")
    processor.run_two_stage_collection()
    print("Data collection complete.")
