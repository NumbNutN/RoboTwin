"""
Data generation for place_bread_basket task with positive/negative sampling.

Positive sampling strategy:
- During anchor replay, maintains a rolling buffer of states during grasp phase
- After play_once, restores to N steps BEFORE grasp completes
- Re-plans and executes grasp -> lift -> place with need_plan=True
  (same API as envs/place_bread_basket.py)

Negative sampling strategy:
- Injects noise during the placement phase
"""

import os
import h5py
import numpy as np
import pickle
import sapien
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

    Positive sampling: captures rolling state buffer during grasp phase,
    then restores to N steps before grasp completes and re-plans via move() API.
    """

    PRE_GRASP_STEPS = 30  # Save state from N steps before grasp completes

    def __init__(self):
        place_bread_basket.__init__(self)
        DataGenBase.__init__(self)
        self._init_sampling_config()

    def _init_sampling_config(self):
        """Initialize/re-initialize all sampling configuration."""
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
        self._grasp_state_buffer = []  # Rolling buffer for pre-grasp states

    def setup_demo(self, **kwargs):
        """Override to re-apply sampling config after parent init resets."""
        super().setup_demo(**kwargs)
        self._init_sampling_config()

    # ==================== MRO Overrides ====================
    # Base_Task defines these too; explicitly delegate to DataGenBase.

    def merge_pkl_to_hdf5_video(self):
        DataGenBase.merge_pkl_to_hdf5_video(self)

    def _take_picture(self):
        return DataGenBase._take_picture(self)

    # ==================== State Management ====================

    def _get_task_object_states(self):
        return {
            "breadbasket_pose": self.breadbasket.get_pose(),
            "bread_poses": [b.get_pose() for b in self.bread],
        }

    def _set_task_object_states(self, state):
        if "breadbasket_pose" in state:
            self.breadbasket.actor.set_pose(state["breadbasket_pose"])
        if "bread_poses" in state:
            for i, pose in enumerate(state["bread_poses"]):
                if i < len(self.bread):
                    self.bread[i].actor.set_pose(pose)

    # ==================== Rolling Buffer ====================

    def _capture_grasp_state(self):
        """Append current state to rolling buffer (during grasp phase only)."""
        self._grasp_state_buffer.append(self.get_state())
        max_size = self.PRE_GRASP_STEPS + 10
        if len(self._grasp_state_buffer) > max_size:
            self._grasp_state_buffer = self._grasp_state_buffer[-self.PRE_GRASP_STEPS:]

    def _get_pre_grasp_state(self):
        """Return the state from N steps before the end of the buffer."""
        if not self._grasp_state_buffer:
            return self.get_state()
        N = min(self.PRE_GRASP_STEPS, len(self._grasp_state_buffer))
        return self._grasp_state_buffer[-N]

    # ==================== Execution Overrides ====================

    def _should_save_frames(self):
        """Whether to save frames in the current mode.
        - Anchor replay (need_plan=False, ANCHOR): save
        - Positive/Negative sampling (need_plan=True, non-ANCHOR): save
        - Initial planning (need_plan=True, ANCHOR): don't save
        """
        return not self.need_plan or self.sample_type != SampleType.ANCHOR.value

    def take_dense_action(self, control_seq, save_freq=-1):
        """Override with rolling buffer capture and pos-sample frame saving."""
        left_arm = control_seq.get("left_arm")
        left_gripper = control_seq.get("left_gripper")
        right_arm = control_seq.get("right_arm")
        right_gripper = control_seq.get("right_gripper")

        save_freq = self.save_freq if save_freq == -1 else save_freq
        should_save = save_freq is not None and self._should_save_frames()

        if should_save:
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
            # Rolling buffer: capture state during grasp phase of anchor replay
            if (self.current_phase == "grasp"
                    and self.sample_type == SampleType.ANCHOR.value
                    and not self.need_plan):
                self._capture_grasp_state()

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

            if should_save and control_idx % save_freq == 0:
                self._update_render()
                self._take_picture()

        if should_save:
            self._take_picture()
        return True

    def together_move_to_pose(
        self,
        left_target_pose,
        right_target_pose,
        left_constraint_pose=None,
        right_constraint_pose=None,
        use_point_cloud=False,
        use_attach=False,
        save_freq=-1,
    ):
        """Override to add rolling buffer capture during grasp phase."""
        if not self.plan_success:
            return
        if left_target_pose is None or right_target_pose is None:
            self.plan_success = False
            return
        if type(left_target_pose) == sapien.Pose:
            left_target_pose = left_target_pose.p.tolist() + left_target_pose.q.tolist()
        if type(right_target_pose) == sapien.Pose:
            right_target_pose = right_target_pose.p.tolist() + right_target_pose.q.tolist()

        save_freq = self.save_freq if save_freq == -1 else save_freq

        if self.need_plan:
            left_result = self.robot.left_plan_path(
                left_target_pose, constraint_pose=left_constraint_pose
            )
            right_result = self.robot.right_plan_path(
                right_target_pose, constraint_pose=right_constraint_pose
            )
            self.left_joint_path.append(deepcopy(left_result))
            self.right_joint_path.append(deepcopy(right_result))
        else:
            left_result = deepcopy(self.left_joint_path[self.left_cnt])
            right_result = deepcopy(self.right_joint_path[self.right_cnt])
            self.left_cnt += 1
            self.right_cnt += 1

        try:
            left_success = left_result["status"] == "Success"
            right_success = right_result["status"] == "Success"
            if not left_success or not right_success:
                self.plan_success = False
        except Exception:
            if left_result is None or right_result is None:
                self.plan_success = False
                return

        should_save = save_freq is not None and self._should_save_frames()

        if should_save:
            self._take_picture()

        now_left_id = 0
        now_right_id = 0
        i = 0
        left_n_step = left_result["position"].shape[0] if left_success else 0
        right_n_step = right_result["position"].shape[0] if right_success else 0

        while now_left_id < left_n_step or now_right_id < right_n_step:
            # Rolling buffer: capture state during grasp phase of anchor replay
            if (self.current_phase == "grasp"
                    and self.sample_type == SampleType.ANCHOR.value
                    and not self.need_plan):
                self._capture_grasp_state()

            if (left_success and now_left_id < left_n_step
                    and (not right_success or now_left_id / left_n_step <= now_right_id / right_n_step)):
                self.robot.set_arm_joints(
                    left_result["position"][now_left_id],
                    left_result["velocity"][now_left_id],
                    "left",
                )
                now_left_id += 1

            if (right_success and now_right_id < right_n_step
                    and (not left_success or now_right_id / right_n_step <= now_left_id / left_n_step)):
                self.robot.set_arm_joints(
                    right_result["position"][now_right_id],
                    right_result["velocity"][now_right_id],
                    "right",
                )
                now_right_id += 1

            self.scene.step()
            if self.render_freq and i % self.render_freq == 0:
                self._update_render()
                if hasattr(self, 'viewer') and self.viewer:
                    self.viewer.render()

            if should_save and i % save_freq == 0:
                self._update_render()
                self._take_picture()
            i += 1

        if should_save:
            self._take_picture()

    # ==================== play_once ====================

    def play_once(self):
        """Override with pre-grasp rolling-buffer state capture."""

        def remove_bread_with_hook(bread_idx, num):
            arm_tag = ArmTag("right" if self.bread[bread_idx].get_pose().p[0] > 0 else "left")
            self._current_bread_idx = bread_idx

            # Phase: Grasp - buffer capture starts
            self.current_phase = "grasp"
            self._grasp_state_buffer = []

            self.move(self.grasp_actor(self.bread[bread_idx], arm_tag=arm_tag, pre_grasp_dis=0.07))

            # Save pre-grasp state (N steps before grasp complete)
            if (not self.need_plan and
                    self.sample_type == SampleType.ANCHOR.value and
                    self.alt_grasp_pos_config.get("active", False)):
                self._grasp_states_for_pos.append({
                    'state': self._get_pre_grasp_state(),
                    'bread_idx': bread_idx,
                    'arm_tag': str(arm_tag),
                    'frame_idx': self.FRAME_IDX,
                    'remaining_actions': num,
                    'dual_grasp': False,
                })
                print(f"[PlaceBread] Saved pre-grasp state for bread {bread_idx} "
                      f"(buffer={len(self._grasp_state_buffer)}) at frame {self.FRAME_IDX}")
                self._grasp_state_buffer = []

            # Phase: Lift
            self.current_phase = "lift"
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))

            # Phase: Place
            self.current_phase = "place"
            breadbasket_pose = self.breadbasket.get_functional_point(0)
            self.move(self.place_actor(
                self.bread[bread_idx], arm_tag=arm_tag,
                target_pose=breadbasket_pose, constrain="free", pre_dis=0.12,
            ))

            if num == 0:
                self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.15, move_axis="arm"))
            else:
                self.move(self.open_gripper(arm_tag=arm_tag))

        def remove_dual_with_hook():
            id = 0 if self.bread[0].get_pose().p[0] < 0 else 1

            # Phase: Grasp - buffer capture starts
            self.current_phase = "grasp"
            self._grasp_state_buffer = []

            self.move(
                self.grasp_actor(self.bread[id], arm_tag="left", pre_grasp_dis=0.05),
                self.grasp_actor(self.bread[id ^ 1], arm_tag="right", pre_grasp_dis=0.07),
            )

            # Save pre-grasp state
            if (not self.need_plan and
                    self.sample_type == SampleType.ANCHOR.value and
                    self.alt_grasp_pos_config.get("active", False)):
                self._grasp_states_for_pos.append({
                    'state': self._get_pre_grasp_state(),
                    'bread_idx': id,
                    'arm_tag': 'left',
                    'frame_idx': self.FRAME_IDX,
                    'dual_grasp': True,
                    'other_bread_idx': id ^ 1,
                })
                print(f"[PlaceBread] Saved pre-grasp dual state "
                      f"(buffer={len(self._grasp_state_buffer)}) at frame {self.FRAME_IDX}")
                self._grasp_state_buffer = []

            # Phase: Lift
            self.current_phase = "lift"
            self.move(
                self.move_by_displacement(arm_tag="left", z=0.05, move_axis="arm"),
                self.move_by_displacement(arm_tag="right", z=0.05, move_axis="arm"),
            )

            # Phase: Place
            self.current_phase = "place"
            breadbasket_pose = self.breadbasket.get_functional_point(0)

            self.move(self.place_actor(
                self.bread[id], arm_tag="left",
                target_pose=breadbasket_pose, constrain="free", pre_dis=0.13,
            ))
            self.move(self.move_by_displacement(arm_tag="left", z=0.1, move_axis="arm"))
            self.move(
                self.back_to_origin(arm_tag="left"),
                self.place_actor(
                    self.bread[id ^ 1], arm_tag="right",
                    target_pose=breadbasket_pose, constrain="free", pre_dis=0.13, dis=0.05,
                ),
            )

        # ---- Main play_once logic ----
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

    # ==================== Positive Sampling ====================

    def sample_pos_from_grasp_states(self, n_samples_per_grasp=2):
        """
        Restore to pre-grasp state and re-plan grasp -> lift -> place
        using the same move() API as place_bread_basket.play_once.
        """
        if not self._grasp_states_for_pos:
            print("[PlaceBread] No grasp states saved for positive sampling")
            return

        print(f"\n=== Generating Positive Samples from "
              f"{len(self._grasp_states_for_pos)} Grasp States ===")

        # Save original replay state
        saved_ljp = self.left_joint_path
        saved_rjp = self.right_joint_path
        saved_lcnt = self.left_cnt
        saved_rcnt = self.right_cnt
        saved_need_plan = self.need_plan

        for grasp_idx, grasp_info in enumerate(self._grasp_states_for_pos):
            print(f"\n--- Processing Grasp State {grasp_idx + 1} ---")

            for sample_idx in range(n_samples_per_grasp):
                print(f"  Positive Sample {sample_idx + 1}/{n_samples_per_grasp}")

                # Restore to pre-grasp state
                self.set_state(grasp_info['state'])

                # Reset planning state for re-planning
                self.left_joint_path = []
                self.right_joint_path = []
                self.left_cnt = 0
                self.right_cnt = 0
                self.need_plan = True
                self.plan_success = True

                # Set positive sample mode
                self.sample_type = SampleType.POSITIVE.value
                self.branch_idx = grasp_info['frame_idx'] * 100 + sample_idx
                self.pos_step_idx = 0
                self.start_qpos = np.concatenate([
                    self.robot.left_entity.get_qpos(),
                    self.robot.right_entity.get_qpos()
                ])

                # Re-plan and execute: grasp -> lift -> place
                success = self._replay_grasp_and_place(grasp_info)

                if success:
                    print(f"    Sample {sample_idx + 1} completed successfully")
                else:
                    print(f"    Sample {sample_idx + 1} failed "
                          f"(plan_success={self.plan_success})")

        # Restore original state
        self.sample_type = SampleType.ANCHOR.value
        self.start_qpos = None
        self.need_plan = saved_need_plan
        self.left_joint_path = saved_ljp
        self.right_joint_path = saved_rjp
        self.left_cnt = saved_lcnt
        self.right_cnt = saved_rcnt
        self._grasp_states_for_pos = []

    def _replay_grasp_and_place(self, grasp_info):
        """Re-execute grasp -> lift -> place with need_plan=True."""
        if grasp_info.get('dual_grasp', False):
            return self._replay_dual(grasp_info)
        else:
            return self._replay_single(grasp_info)

    def _replay_single(self, grasp_info):
        """Single-arm: grasp -> lift -> place (mirrors remove_bread in play_once)."""
        arm_tag = ArmTag(grasp_info['arm_tag'])
        bread_idx = grasp_info['bread_idx']
        num = grasp_info.get('remaining_actions', 0)

        # Grasp
        self.move(self.grasp_actor(
            self.bread[bread_idx], arm_tag=arm_tag, pre_grasp_dis=0.07
        ))
        if not self.plan_success:
            return False

        # Lift
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))
        if not self.plan_success:
            return False

        # Place
        breadbasket_pose = self.breadbasket.get_functional_point(0)
        self.move(self.place_actor(
            self.bread[bread_idx], arm_tag=arm_tag,
            target_pose=breadbasket_pose, constrain="free", pre_dis=0.12,
        ))
        if not self.plan_success:
            return False

        if num == 0:
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.15, move_axis="arm"))
        else:
            self.move(self.open_gripper(arm_tag=arm_tag))

        return self.plan_success

    def _replay_dual(self, grasp_info):
        """Dual-arm: grasp -> lift -> place (mirrors remove_dual in play_once)."""
        bread_idx = grasp_info['bread_idx']
        other_bread_idx = grasp_info['other_bread_idx']

        # Dual grasp
        self.move(
            self.grasp_actor(self.bread[bread_idx], arm_tag="left", pre_grasp_dis=0.05),
            self.grasp_actor(self.bread[other_bread_idx], arm_tag="right", pre_grasp_dis=0.07),
        )
        if not self.plan_success:
            return False

        # Lift both
        self.move(
            self.move_by_displacement(arm_tag="left", z=0.05, move_axis="arm"),
            self.move_by_displacement(arm_tag="right", z=0.05, move_axis="arm"),
        )
        if not self.plan_success:
            return False

        # Place left bread
        breadbasket_pose = self.breadbasket.get_functional_point(0)
        self.move(self.place_actor(
            self.bread[bread_idx], arm_tag="left",
            target_pose=breadbasket_pose, constrain="free", pre_dis=0.13,
        ))
        if not self.plan_success:
            return False

        # Lift left
        self.move(self.move_by_displacement(arm_tag="left", z=0.1, move_axis="arm"))
        if not self.plan_success:
            return False

        # Retract left, place right
        self.move(
            self.back_to_origin(arm_tag="left"),
            self.place_actor(
                self.bread[other_bread_idx], arm_tag="right",
                target_pose=breadbasket_pose, constrain="free", pre_dis=0.13, dis=0.05,
            ),
        )
        return self.plan_success


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

            #! WARN: cache file would skip play_once
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

                # Generate positive samples from pre-grasp states
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

            if not self.env.check_success():
                print(f"Warning: Episode {epid} did not succeed!")


if __name__ == "__main__":
    processor = DataProcessor("place_bread_basket", "data/place_bread_basket/data", "data/processed")
    processor.run_two_stage_collection()
    print("Data collection complete.")
