"""
Data generation for place_bread_basket task with positive/negative sampling.

Positive sampling strategy (per-phase):
- Saves state at the START of each phase (grasp / lift / place)
- After play_once, restores to phase-start state
- Re-plans and executes ONLY that phase with need_plan=True
- Controlled via sampling_config[phase]["pos"]["active"]

Negative sampling strategy:
- Injects noise during the placement phase
"""

import os
import numpy as np
import sapien
import sys
import yaml
from copy import deepcopy

sys.path.append(os.getcwd())

from envs.place_bread_basket import place_bread_basket
from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs.utils import ArmTag

from .data_gen_base import DataGenBase, SampleType, SamplingTrigger


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


class PlaceBreadBasketDataGen(place_bread_basket, DataGenBase):
    """
    Extended place_bread_basket environment with per-phase positive sampling.

    Each phase (grasp, lift, place) can independently generate positive samples
    by restoring to the phase-start state and re-planning via move() API.
    """

    def __init__(self):
        place_bread_basket.__init__(self)
        DataGenBase.__init__(self)
        self._init_sampling_config()

    def _init_sampling_config(self):
        """Initialize/re-initialize all sampling configuration."""
        # Per-phase sampling: enable pos/neg independently for each phase
        self.sampling_config = {
            "grasp": {
                "neg": {"active": False, "interval": 100, "duration": 50},
                "pos": {"active": True, "n_samples": 2}
            },
            "lift": {
                "neg": {"active": False, "interval": 200, "duration": 50},
                "pos": {"active": False, "n_samples": 0}
            },
            "place": {
                "neg": {"active": False, "interval": 150, "duration": 50},
                "pos": {"active": False, "n_samples": 0}
            },
            "default": {
                "neg": {"active": False, "interval": 999, "duration": 10},
                "pos": {"active": False, "n_samples": 0}
            }
        }
        # Phase-start states for positive sampling
        # Each entry: {'phase', 'state', 'frame_idx', 'bread_idx', 'arm_tag',
        #              'is_dual', 'other_bread_idx', ...}
        self._phase_states_for_pos = []

    def setup_demo(self, **kwargs):
        """Override to re-apply sampling config after parent init resets."""
        super().setup_demo(**kwargs)
        self._init_sampling_config()

    # ==================== MRO Overrides ====================

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

    def _save_phase_state(self, phase, bread_idx, arm_tag, is_dual=False,
                          other_bread_idx=None, **extra):
        """Save state at the START of a phase for later positive sampling."""
        pos_cfg = self.sampling_config.get(phase, {}).get("pos", {})
        if (not pos_cfg.get("active", False)
                or self.need_plan
                or self.sample_type != SampleType.ANCHOR.value):
            return

        info = {
            'phase': phase,
            'state': self.get_state(),
            'frame_idx': self.FRAME_IDX,
            'bread_idx': bread_idx,
            'arm_tag': str(arm_tag),
            'is_dual': is_dual,
        }
        if is_dual and other_bread_idx is not None:
            info['other_bread_idx'] = other_bread_idx
        info.update(extra)

        self._phase_states_for_pos.append(info)
        print(f"[PlaceBread] Saved phase-start state: "
              f"phase={phase}, bread={bread_idx}, frame={self.FRAME_IDX}")

    # ==================== Execution Overrides ====================

    def _should_save_frames(self):
        """Save frames during anchor replay OR positive/negative generation."""
        return not self.need_plan or self.sample_type != SampleType.ANCHOR.value

    def take_dense_action(self, control_seq, save_freq=-1):
        """Override with pos-sample frame saving support."""
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
        """Override to support frame saving during positive sampling."""
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
                left_target_pose, constraint_pose=left_constraint_pose)
            right_result = self.robot.right_plan_path(
                right_target_pose, constraint_pose=right_constraint_pose)
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
            if (left_success and now_left_id < left_n_step
                    and (not right_success
                         or now_left_id / left_n_step <= now_right_id / right_n_step)):
                self.robot.set_arm_joints(
                    left_result["position"][now_left_id],
                    left_result["velocity"][now_left_id],
                    "left",
                )
                now_left_id += 1
            if (right_success and now_right_id < right_n_step
                    and (not left_success
                         or now_right_id / right_n_step <= now_left_id / left_n_step)):
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
        """Override with per-phase state saving at phase START."""

        def remove_bread_with_hook(bread_idx, num):
            arm_tag = ArmTag(
                "right" if self.bread[bread_idx].get_pose().p[0] > 0 else "left")
            ctx = dict(bread_idx=bread_idx, arm_tag=arm_tag,
                       is_dual=False, remaining_actions=num)

            # ---- Grasp ----
            self.current_phase = "grasp"
            self._save_phase_state("grasp", **ctx)
            self.move(self.grasp_actor(
                self.bread[bread_idx], arm_tag=arm_tag, pre_grasp_dis=0.07))

            # ---- Lift ----
            self.current_phase = "lift"
            self._save_phase_state("lift", **ctx)
            self.move(self.move_by_displacement(
                arm_tag=arm_tag, z=0.1, move_axis="arm"))

            # ---- Place ----
            self.current_phase = "place"
            self._save_phase_state("place", **ctx)
            breadbasket_pose = self.breadbasket.get_functional_point(0)
            self.move(self.place_actor(
                self.bread[bread_idx], arm_tag=arm_tag,
                target_pose=breadbasket_pose, constrain="free", pre_dis=0.12,
            ))
            if num == 0:
                self.move(self.move_by_displacement(
                    arm_tag=arm_tag, z=0.15, move_axis="arm"))
            else:
                self.move(self.open_gripper(arm_tag=arm_tag))

        def remove_dual_with_hook():
            id = 0 if self.bread[0].get_pose().p[0] < 0 else 1
            ctx = dict(bread_idx=id, arm_tag='left',
                       is_dual=True, other_bread_idx=id ^ 1)

            # ---- Grasp (dual) ----
            self.current_phase = "grasp"
            self._save_phase_state("grasp", **ctx)
            self.move(
                self.grasp_actor(self.bread[id], arm_tag="left",
                                 pre_grasp_dis=0.05),
                self.grasp_actor(self.bread[id ^ 1], arm_tag="right",
                                 pre_grasp_dis=0.07),
            )

            # ---- Lift (dual) ----
            self.current_phase = "lift"
            self._save_phase_state("lift", **ctx)
            self.move(
                self.move_by_displacement(arm_tag="left", z=0.05,
                                          move_axis="arm"),
                self.move_by_displacement(arm_tag="right", z=0.05,
                                          move_axis="arm"),
            )

            # ---- Place (dual: left then right) ----
            self.current_phase = "place"
            self._save_phase_state("place", **ctx)
            breadbasket_pose = self.breadbasket.get_functional_point(0)
            self.move(self.place_actor(
                self.bread[id], arm_tag="left",
                target_pose=breadbasket_pose, constrain="free", pre_dis=0.13,
            ))
            self.move(self.move_by_displacement(
                arm_tag="left", z=0.1, move_axis="arm"))
            self.move(
                self.back_to_origin(arm_tag="left"),
                self.place_actor(
                    self.bread[id ^ 1], arm_tag="right",
                    target_pose=breadbasket_pose, constrain="free",
                    pre_dis=0.13, dis=0.05,
                ),
            )

        # ---- Main ----
        self._phase_states_for_pos = []

        arm_info = None
        if (len(self.bread) <= 1
                or (self.bread[0].get_pose().p[0]
                    * self.bread[1].get_pose().p[0]) > 0):
            if len(self.bread) == 1:
                remove_bread_with_hook(0, 0)
                arm_info = ("left" if self.bread[0].get_pose().p[0] < 0
                            else "right")
            else:
                id = (0 if self.bread[0].get_pose().p[1]
                      < self.bread[1].get_pose().p[1] else 1)
                arm_info = ("left" if self.bread[0].get_pose().p[0] < 0
                            else "right")
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

    def sample_pos_from_phase_states(self):
        """
        Generate positive samples for each saved phase.
        Only phases with sampling_config[phase]["pos"]["active"] == True
        are processed. Each positive sample replays ONLY that phase.
        """
        if not self._phase_states_for_pos:
            print("[PlaceBread] No phase states saved for positive sampling")
            return

        # Save original replay state
        saved_ljp = self.left_joint_path
        saved_rjp = self.right_joint_path
        saved_lcnt = self.left_cnt
        saved_rcnt = self.right_cnt
        saved_need_plan = self.need_plan

        for phase_info in self._phase_states_for_pos:
            phase = phase_info['phase']
            pos_cfg = self.sampling_config.get(phase, {}).get("pos", {})
            if not pos_cfg.get("active", False):
                continue
            n_samples = pos_cfg.get("n_samples", 2)

            print(f"\n--- Phase '{phase}': generating {n_samples} positive samples ---")

            for sample_idx in range(n_samples):
                print(f"  Positive sample {sample_idx + 1}/{n_samples} "
                      f"(phase={phase})")

                # Restore to phase-start state
                self.set_state(phase_info['state'])
                # Step once to sync physics/renderer after state restore
                self.scene.step()
                self._update_render()

                # Reset planning state
                self.left_joint_path = []
                self.right_joint_path = []
                self.left_cnt = 0
                self.right_cnt = 0
                self.need_plan = True
                self.plan_success = True

                # Set positive sample mode
                self.sample_type = SampleType.POSITIVE.value
                self.branch_idx = phase_info['frame_idx'] * 100 + sample_idx
                self.pos_step_idx = 0
                self.start_qpos = np.concatenate([
                    self.robot.left_entity.get_qpos(),
                    self.robot.right_entity.get_qpos()
                ])

                # Replay ONLY this phase
                success = self._replay_phase(phase_info)

                status = "OK" if success else f"FAIL (plan={self.plan_success})"
                print(f"    -> {status}")

        # Restore
        self.sample_type = SampleType.ANCHOR.value
        self.start_qpos = None
        self.need_plan = saved_need_plan
        self.left_joint_path = saved_ljp
        self.right_joint_path = saved_rjp
        self.left_cnt = saved_lcnt
        self.right_cnt = saved_rcnt
        self._phase_states_for_pos = []

    # Keep old name as alias for compatibility with DataProcessor
    def sample_pos_from_grasp_states(self, n_samples_per_grasp=None):
        self.sample_pos_from_phase_states()

    # ==================== Phase Replay ====================

    def _replay_phase(self, phase_info):
        """Replay a single phase with need_plan=True. Returns success bool."""
        phase = phase_info['phase']
        is_dual = phase_info.get('is_dual', False)

        if phase == 'grasp':
            return self._replay_grasp(phase_info, is_dual)
        elif phase == 'lift':
            return self._replay_lift(phase_info, is_dual)
        elif phase == 'place':
            return self._replay_place(phase_info, is_dual)
        return False

    def _replay_grasp(self, info, is_dual):
        bread_idx = info['bread_idx']
        if is_dual:
            other = info['other_bread_idx']
            self.move(
                self.grasp_actor(self.bread[bread_idx], arm_tag="left",
                                 pre_grasp_dis=0.05),
                self.grasp_actor(self.bread[other], arm_tag="right",
                                 pre_grasp_dis=0.07),
            )
        else:
            arm_tag = ArmTag(info['arm_tag'])
            self.move(self.grasp_actor(
                self.bread[bread_idx], arm_tag=arm_tag, pre_grasp_dis=0.07))
        return self.plan_success

    def _replay_lift(self, info, is_dual):
        if is_dual:
            self.move(
                self.move_by_displacement(arm_tag="left", z=0.05,
                                          move_axis="arm"),
                self.move_by_displacement(arm_tag="right", z=0.05,
                                          move_axis="arm"),
            )
        else:
            arm_tag = ArmTag(info['arm_tag'])
            self.move(self.move_by_displacement(
                arm_tag=arm_tag, z=0.1, move_axis="arm"))
        return self.plan_success

    def _replay_place(self, info, is_dual):
        bread_idx = info['bread_idx']
        breadbasket_pose = self.breadbasket.get_functional_point(0)

        if is_dual:
            other = info['other_bread_idx']
            self.move(self.place_actor(
                self.bread[bread_idx], arm_tag="left",
                target_pose=breadbasket_pose, constrain="free", pre_dis=0.13,
            ))
            if not self.plan_success:
                return False
            self.move(self.move_by_displacement(
                arm_tag="left", z=0.1, move_axis="arm"))
            if not self.plan_success:
                return False
            self.move(
                self.back_to_origin(arm_tag="left"),
                self.place_actor(
                    self.bread[other], arm_tag="right",
                    target_pose=breadbasket_pose, constrain="free",
                    pre_dis=0.13, dis=0.05,
                ),
            )
        else:
            arm_tag = ArmTag(info['arm_tag'])
            self.move(self.place_actor(
                self.bread[bread_idx], arm_tag=arm_tag,
                target_pose=breadbasket_pose, constrain="free", pre_dis=0.12,
            ))
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
        embodiment_config_path = os.path.join(CONFIGS_PATH,
                                              "_embodiment_config.yml")
        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        def get_embodiment_file(etype):
            return _embodiment_types[etype]["file_path"]

        if len(embodiment_type) == 1:
            self.args["left_robot_file"] = get_embodiment_file(
                embodiment_type[0])
            self.args["right_robot_file"] = get_embodiment_file(
                embodiment_type[0])
            self.args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            self.args["left_robot_file"] = get_embodiment_file(
                embodiment_type[0])
            self.args["right_robot_file"] = get_embodiment_file(
                embodiment_type[1])
            self.args["embodiment_dis"] = embodiment_type[2]
            self.args["dual_arm_embodied"] = False

        self.args["left_embodiment_config"] = get_embodiment_config(
            self.args["left_robot_file"])
        self.args["right_embodiment_config"] = get_embodiment_config(
            self.args["right_robot_file"])

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
            print(f"No seeds found in {seed_list_path}. "
                  "Run collect_data.py first.")
            return

        with open(seed_list_path, "r") as file:
            seed_list = [int(i) for i in file.read().split()]

        print(f"Found {len(seed_list)} seeds.")

        for epid, seed in enumerate(seed_list):
            print(f"\n=== Processing Episode {epid} (Seed {seed}) ===")

            self.env.setup_demo(now_ep_num=epid, seed=seed, **self.args)

            cache_path = f"{self.env.save_dir}/.cache/episode{epid}/"
            is_cached = os.path.exists(
                os.path.join(cache_path, "anchor_0.pkl"))

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

                # Generate per-phase positive samples
                if self.env._phase_states_for_pos and self.env.check_success():
                    print(f"\n--- Generating Positive Samples ---")
                    self.env.sample_pos_from_phase_states()
                    print(f"--- Positive Sampling Complete ---\n")

            # Merge and save
            print(f"Merging to HDF5 for episode {epid}")
            self.env.close_env()
            self.env.merge_pkl_to_hdf5_video()

            if not self.env.check_success():
                print(f"Warning: Episode {epid} did not succeed!")


if __name__ == "__main__":
    processor = DataProcessor(
        "place_bread_basket",
        "data/place_bread_basket/data",
        "data/processed"
    )
    processor.run_two_stage_collection()
    print("Data collection complete.")
