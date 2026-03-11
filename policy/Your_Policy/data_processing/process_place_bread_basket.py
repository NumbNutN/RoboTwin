"""
Data generation for place_bread_basket task with per-phase pos/neg sampling.

Positive sampling (per-phase):
- Saves state at the START of each phase
- Restores -> re-plans ONLY that phase with need_plan=True

Negative sampling (per-phase):
- Maintains rolling buffer during execution
- Saves state at N steps BEFORE phase end
- Restores -> re-plans with PERTURBED target -> generates failure trajectory
"""

import os
import numpy as np
import sapien
import sys
import yaml
import transforms3d as t3d
from copy import deepcopy

sys.path.append(os.getcwd())

from envs.place_bread_basket import place_bread_basket
from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs.utils import ArmTag

from .data_gen_base import DataGenBase, SampleType


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


class PlaceBreadBasketDataGen(place_bread_basket, DataGenBase):
    """
    Extended place_bread_basket with per-phase positive & negative sampling.

    Each phase (grasp/lift/place) independently saves:
      - start_state   -> for positive samples (replay whole phase)
      - pre_end_state -> for negative samples (replay with perturbed target)
    """

    PRE_END_STEPS = 50  # N steps before phase end for negative sampling

    def __init__(self):
        place_bread_basket.__init__(self)
        DataGenBase.__init__(self)
        self._init_sampling_config()

    def _init_sampling_config(self):
        self.sampling_config = {
            "grasp": {
                "pos": {"active": True, "n_samples": 2},
                "neg": {"active": True, "n_samples": 2,
                        "pre_end_steps": 50},
            },
            "lift": {
                "pos": {"active": False, "n_samples": 0},
                "neg": {"active": False, "n_samples": 0,
                        "pre_end_steps": 30},
            },
            "place": {
                "pos": {"active": False, "n_samples": 0},
                "neg": {"active": False, "n_samples": 0,
                        "pre_end_steps": 30},
            },
            "default": {
                "pos": {"active": False, "n_samples": 0},
                "neg": {"active": False, "n_samples": 0,
                        "pre_end_steps": 30},
            },
        }
        self._phase_records = []       # unified per-phase records
        self._phase_state_buffer = []  # rolling buffer for current phase

    def setup_demo(self, **kwargs):
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

    # ==================== Phase Record Helpers ====================

    def _save_phase_start(self, phase, bread_idx, arm_tag,
                          is_dual=False, other_bread_idx=None, **extra):
        """Called at the START of a phase. Creates record with start_state."""
        if self.need_plan or self.sample_type != SampleType.ANCHOR.value:
            return

        pos_cfg = self.sampling_config.get(phase, {}).get("pos", {})
        neg_cfg = self.sampling_config.get(phase, {}).get("neg", {})
        if not pos_cfg.get("active") and not neg_cfg.get("active"):
            return

        record = {
            'phase': phase,
            'start_state': self.get_state(),
            'start_frame_idx': self.FRAME_IDX,
            'pre_end_state': None,           # filled by _finalize_phase
            'pre_end_frame_idx': None,
            'bread_idx': bread_idx,
            'arm_tag': str(arm_tag),
            'is_dual': is_dual,
        }
        if is_dual and other_bread_idx is not None:
            record['other_bread_idx'] = other_bread_idx
        record.update(extra)

        self._phase_records.append(record)
        self._phase_state_buffer = []  # reset rolling buffer
        print(f"[PlaceBread] Phase '{phase}' started at frame {self.FRAME_IDX}")

    def _finalize_phase(self):
        """Called at the END of a phase. Saves pre_end_state from buffer."""
        if not self._phase_records:
            return
        record = self._phase_records[-1]
        if record['pre_end_state'] is not None:
            return  # already finalized

        neg_cfg = self.sampling_config.get(
            record['phase'], {}).get("neg", {})
        if neg_cfg.get("active") and self._phase_state_buffer:
            N = neg_cfg.get("pre_end_steps", self.PRE_END_STEPS)
            idx = max(0, len(self._phase_state_buffer) - N)
            record['pre_end_state'] = self._phase_state_buffer[idx]
            record['pre_end_frame_idx'] = self.FRAME_IDX
            print(f"[PlaceBread] Phase '{record['phase']}' ended: "
                  f"buffer={len(self._phase_state_buffer)}, "
                  f"saved pre_end at -{min(N, len(self._phase_state_buffer))} steps")

        self._phase_state_buffer = []

    def _capture_buffer_state(self):
        """Capture current state into rolling buffer (called each sim step)."""
        self._phase_state_buffer.append(self.get_state())
        max_size = self.PRE_END_STEPS + 20
        if len(self._phase_state_buffer) > max_size:
            self._phase_state_buffer = self._phase_state_buffer[-self.PRE_END_STEPS:]

    # ==================== Execution Overrides ====================

    def _should_save_frames(self):
        return not self.need_plan or self.sample_type != SampleType.ANCHOR.value

    def _should_capture_buffer(self):
        """Whether to capture states in rolling buffer."""
        if self.need_plan or self.sample_type != SampleType.ANCHOR.value:
            return False
        neg_cfg = self.sampling_config.get(
            self.current_phase, {}).get("neg", {})
        return neg_cfg.get("active", False)

    def take_dense_action(self, control_seq, save_freq=-1):
        left_arm = control_seq.get("left_arm")
        left_gripper = control_seq.get("left_gripper")
        right_arm = control_seq.get("right_arm")
        right_gripper = control_seq.get("right_gripper")

        save_freq = self.save_freq if save_freq == -1 else save_freq
        should_save = save_freq is not None and self._should_save_frames()
        should_capture = self._should_capture_buffer()

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
            if should_capture:
                self._capture_buffer_state()

            if left_arm is not None and control_idx < left_arm["position"].shape[0]:
                self.robot.set_arm_joints(
                    left_arm["position"][control_idx],
                    left_arm["velocity"][control_idx], "left")
            if left_gripper is not None and control_idx < left_gripper["num_step"]:
                self.robot.set_gripper(
                    left_gripper["result"][control_idx],
                    "left", left_gripper["per_step"])
            if right_arm is not None and control_idx < right_arm["position"].shape[0]:
                self.robot.set_arm_joints(
                    right_arm["position"][control_idx],
                    right_arm["velocity"][control_idx], "right")
            if right_gripper is not None and control_idx < right_gripper["num_step"]:
                self.robot.set_gripper(
                    right_gripper["result"][control_idx],
                    "right", right_gripper["per_step"])

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
        self, left_target_pose, right_target_pose,
        left_constraint_pose=None, right_constraint_pose=None,
        use_point_cloud=False, use_attach=False, save_freq=-1,
    ):
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
        should_capture = self._should_capture_buffer()

        if should_save:
            self._take_picture()

        now_left_id = 0
        now_right_id = 0
        i = 0
        left_n_step = left_result["position"].shape[0] if left_success else 0
        right_n_step = right_result["position"].shape[0] if right_success else 0

        while now_left_id < left_n_step or now_right_id < right_n_step:
            if should_capture:
                self._capture_buffer_state()

            if (left_success and now_left_id < left_n_step
                    and (not right_success
                         or now_left_id / left_n_step <= now_right_id / right_n_step)):
                self.robot.set_arm_joints(
                    left_result["position"][now_left_id],
                    left_result["velocity"][now_left_id], "left")
                now_left_id += 1
            if (right_success and now_right_id < right_n_step
                    and (not left_success
                         or now_right_id / right_n_step <= now_left_id / left_n_step)):
                self.robot.set_arm_joints(
                    right_result["position"][now_right_id],
                    right_result["velocity"][now_right_id], "right")
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

        def remove_bread_with_hook(bread_idx, num):
            arm_tag = ArmTag(
                "right" if self.bread[bread_idx].get_pose().p[0] > 0 else "left")
            ctx = dict(bread_idx=bread_idx, arm_tag=arm_tag, is_dual=False)

            self.current_phase = "grasp"
            self._save_phase_start("grasp", **ctx)
            self.move(self.grasp_actor(
                self.bread[bread_idx], arm_tag=arm_tag, pre_grasp_dis=0.07))
            self._finalize_phase()

            self.current_phase = "lift"
            self._save_phase_start("lift", **ctx)
            self.move(self.move_by_displacement(
                arm_tag=arm_tag, z=0.1, move_axis="arm"))
            self._finalize_phase()

            self.current_phase = "place"
            self._save_phase_start("place", **ctx)
            breadbasket_pose = self.breadbasket.get_functional_point(0)
            self.move(self.place_actor(
                self.bread[bread_idx], arm_tag=arm_tag,
                target_pose=breadbasket_pose, constrain="free", pre_dis=0.12))
            if num == 0:
                self.move(self.move_by_displacement(
                    arm_tag=arm_tag, z=0.15, move_axis="arm"))
            else:
                self.move(self.open_gripper(arm_tag=arm_tag))
            self._finalize_phase()

        def remove_dual_with_hook():
            id = 0 if self.bread[0].get_pose().p[0] < 0 else 1
            ctx = dict(bread_idx=id, arm_tag='left',
                       is_dual=True, other_bread_idx=id ^ 1)

            self.current_phase = "grasp"
            self._save_phase_start("grasp", **ctx)
            self.move(
                self.grasp_actor(self.bread[id], arm_tag="left",
                                 pre_grasp_dis=0.05),
                self.grasp_actor(self.bread[id ^ 1], arm_tag="right",
                                 pre_grasp_dis=0.07))
            self._finalize_phase()

            self.current_phase = "lift"
            self._save_phase_start("lift", **ctx)
            self.move(
                self.move_by_displacement(arm_tag="left", z=0.05,
                                          move_axis="arm"),
                self.move_by_displacement(arm_tag="right", z=0.05,
                                          move_axis="arm"))
            self._finalize_phase()

            self.current_phase = "place"
            self._save_phase_start("place", **ctx)
            bp = self.breadbasket.get_functional_point(0)
            self.move(self.place_actor(
                self.bread[id], arm_tag="left",
                target_pose=bp, constrain="free", pre_dis=0.13))
            self.move(self.move_by_displacement(
                arm_tag="left", z=0.1, move_axis="arm"))
            self.move(
                self.back_to_origin(arm_tag="left"),
                self.place_actor(
                    self.bread[id ^ 1], arm_tag="right",
                    target_pose=bp, constrain="free",
                    pre_dis=0.13, dis=0.05))
            self._finalize_phase()

        self._phase_records = []

        arm_info = None
        if (len(self.bread) <= 1
                or self.bread[0].get_pose().p[0]
                * self.bread[1].get_pose().p[0] > 0):
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

    # ==================== Sampling Entry Points ====================

    def _enter_sampling_mode(self, sample_type, state, frame_idx, sample_idx):
        """Common setup before replaying a pos/neg sample."""
        self.set_state(state)
        self.scene.step()
        self._update_render()

        self.left_joint_path = []
        self.right_joint_path = []
        self.left_cnt = 0
        self.right_cnt = 0
        self.need_plan = True
        self.plan_success = True

        self.sample_type = sample_type
        self.branch_idx = frame_idx * 100 + sample_idx
        if sample_type == SampleType.POSITIVE.value:
            self.pos_step_idx = 0
        else:
            self.neg_step_idx = 0
        self.start_qpos = np.concatenate([
            self.robot.left_entity.get_qpos(),
            self.robot.right_entity.get_qpos()
        ])

    def _exit_sampling_mode(self, saved):
        """Restore state after all sampling is done."""
        self.sample_type = SampleType.ANCHOR.value
        self.start_qpos = None
        self.need_plan = saved['need_plan']
        self.left_joint_path = saved['ljp']
        self.right_joint_path = saved['rjp']
        self.left_cnt = saved['lcnt']
        self.right_cnt = saved['rcnt']
        self._phase_records = []

    def _save_replay_state(self):
        return {
            'ljp': self.left_joint_path,
            'rjp': self.right_joint_path,
            'lcnt': self.left_cnt,
            'rcnt': self.right_cnt,
            'need_plan': self.need_plan,
        }

    # ---------- Positive ----------

    def sample_pos_from_phase_states(self):
        """Generate positive samples: restore to phase START, replay phase."""
        saved = self._save_replay_state()

        for record in self._phase_records:
            phase = record['phase']
            cfg = self.sampling_config.get(phase, {}).get("pos", {})
            if not cfg.get("active"):
                continue
            n = cfg.get("n_samples", 2)
            print(f"\n--- Pos samples for '{phase}' (x{n}) ---")

            for i in range(n):
                self._enter_sampling_mode(
                    SampleType.POSITIVE.value,
                    record['start_state'],
                    record['start_frame_idx'], i)

                ok = self._replay_phase(record)
                print(f"  pos[{i}] phase={phase}: "
                      f"{'OK' if ok else 'FAIL'}")

        self._exit_sampling_mode(saved)

    # Keep alias
    def sample_pos_from_grasp_states(self):
        self.sample_pos_from_phase_states()

    # ---------- Negative ----------

    def sample_neg_from_phase_states(self):
        """Generate negative samples: restore to pre-end, replay with perturbation."""
        saved = self._save_replay_state()

        for record in self._phase_records:
            phase = record['phase']
            cfg = self.sampling_config.get(phase, {}).get("neg", {})
            if not cfg.get("active") or record['pre_end_state'] is None:
                continue
            n = cfg.get("n_samples", 2)
            print(f"\n--- Neg samples for '{phase}' (x{n}) ---")

            for i in range(n):
                self._enter_sampling_mode(
                    SampleType.NEGATIVE.value,
                    record['pre_end_state'],
                    record.get('pre_end_frame_idx',
                               record['start_frame_idx']), i)

                ok = self._replay_neg_phase(record, perturbation_idx=i)
                print(f"  neg[{i}] phase={phase}: "
                      f"{'OK' if ok else 'FAIL'}")

        self._exit_sampling_mode(saved)

    # ==================== Phase Replay (Positive) ====================

    def _replay_phase(self, record):
        phase = record['phase']
        is_dual = record.get('is_dual', False)
        if phase == 'grasp':
            return self._replay_grasp(record, is_dual)
        elif phase == 'lift':
            return self._replay_lift(record, is_dual)
        elif phase == 'place':
            return self._replay_place(record, is_dual)
        return False

    def _replay_grasp(self, r, is_dual):
        if is_dual:
            self.move(
                self.grasp_actor(self.bread[r['bread_idx']],
                                 arm_tag="left", pre_grasp_dis=0.05),
                self.grasp_actor(self.bread[r['other_bread_idx']],
                                 arm_tag="right", pre_grasp_dis=0.07))
        else:
            self.move(self.grasp_actor(
                self.bread[r['bread_idx']],
                arm_tag=ArmTag(r['arm_tag']), pre_grasp_dis=0.07))
        return self.plan_success

    def _replay_lift(self, r, is_dual):
        if is_dual:
            self.move(
                self.move_by_displacement(arm_tag="left", z=0.05,
                                          move_axis="arm"),
                self.move_by_displacement(arm_tag="right", z=0.05,
                                          move_axis="arm"))
        else:
            self.move(self.move_by_displacement(
                arm_tag=ArmTag(r['arm_tag']), z=0.1, move_axis="arm"))
        return self.plan_success

    def _replay_place(self, r, is_dual):
        bp = self.breadbasket.get_functional_point(0)
        if is_dual:
            self.move(self.place_actor(
                self.bread[r['bread_idx']], arm_tag="left",
                target_pose=bp, constrain="free", pre_dis=0.13))
            if not self.plan_success:
                return False
            self.move(self.move_by_displacement(
                arm_tag="left", z=0.1, move_axis="arm"))
            if not self.plan_success:
                return False
            self.move(
                self.back_to_origin(arm_tag="left"),
                self.place_actor(
                    self.bread[r['other_bread_idx']], arm_tag="right",
                    target_pose=bp, constrain="free",
                    pre_dis=0.13, dis=0.05))
        else:
            self.move(self.place_actor(
                self.bread[r['bread_idx']],
                arm_tag=ArmTag(r['arm_tag']),
                target_pose=bp, constrain="free", pre_dis=0.12))
        return self.plan_success

    # ==================== Phase Replay (Negative) ====================

    def _replay_neg_phase(self, record, perturbation_idx):
        phase = record['phase']
        is_dual = record.get('is_dual', False)
        if phase == 'grasp':
            return self._replay_neg_grasp(record, is_dual, perturbation_idx)
        elif phase == 'lift':
            return self._replay_neg_lift(record, is_dual, perturbation_idx)
        elif phase == 'place':
            return self._replay_neg_place(record, is_dual, perturbation_idx)
        return False

    def _replay_neg_grasp(self, r, is_dual, pidx):
        """Negative grasp: approach a PERTURBED grasp pose, then close gripper."""
        bread_idx = r['bread_idx']

        def _neg_grasp_one_arm(bread_i, arm_tag, pre_dis=0):
            # Compute original grasp target from contact point
            grasp_target = self.get_grasp_pose(
                self.bread[bread_i], arm_tag,
                contact_point_id=0, pre_dis=0)
            if grasp_target is None:
                return None
            # Perturb the target
            perturbed = self._perturb_for_neg(grasp_target, pidx, "grasp")
            return arm_tag, perturbed

        if is_dual:
            other = r['other_bread_idx']
            left_info = _neg_grasp_one_arm(bread_idx, "left", 0.05)
            right_info = _neg_grasp_one_arm(other, "right", 0.07)
            if left_info is None or right_info is None:
                return False
            # Plan to perturbed targets
            self.move(
                self.move_to_pose("left", left_info[1]),
                self.move_to_pose("right", right_info[1]))
            if not self.plan_success:
                return False
            # Close both grippers
            self.move(self.close_gripper(arm_tag="left"))
            self.move(self.close_gripper(arm_tag="right"))
        else:
            arm_tag = ArmTag(r['arm_tag'])
            info = _neg_grasp_one_arm(bread_idx, arm_tag, 0.07)
            if info is None:
                return False
            self.move(self.move_to_pose(arm_tag, info[1]))
            if not self.plan_success:
                return False
            self.move(self.close_gripper(arm_tag=arm_tag))

        return True

    def _replay_neg_lift(self, r, is_dual, pidx):
        """Negative lift: wrong displacement (too low, wrong direction, etc.)."""
        params = self._get_neg_lift_params(pidx)

        if is_dual:
            self.move(
                self.move_by_displacement(arm_tag="left",
                                          move_axis="arm", **params),
                self.move_by_displacement(arm_tag="right",
                                          move_axis="arm", **params))
        else:
            arm_tag = ArmTag(r['arm_tag'])
            self.move(self.move_by_displacement(
                arm_tag=arm_tag, move_axis="arm", **params))
        return self.plan_success

    def _replay_neg_place(self, r, is_dual, pidx):
        """Negative place: perturbed basket target -> miss the basket."""
        bp = self.breadbasket.get_functional_point(0)
        perturbed_bp = self._perturb_for_neg(bp, pidx, "place")

        if is_dual:
            self.move(self.place_actor(
                self.bread[r['bread_idx']], arm_tag="left",
                target_pose=perturbed_bp, constrain="free", pre_dis=0.13))
        else:
            arm_tag = ArmTag(r['arm_tag'])
            self.move(self.place_actor(
                self.bread[r['bread_idx']], arm_tag=arm_tag,
                target_pose=perturbed_bp, constrain="free", pre_dis=0.12))
        return self.plan_success

    # ==================== Perturbation Strategies ====================

    def _perturb_for_neg(self, pose, strategy_idx, phase="grasp"):
        """
        Apply a heuristic perturbation to a target pose.

        Args:
            pose: [x, y, z, qw, qx, qy, qz]
            strategy_idx: selects which perturbation to apply
            phase: affects perturbation magnitude/type
        Returns:
            Perturbed pose as list.
        """
        pose = np.array(pose, dtype=np.float64)
        strategy = strategy_idx % 4

        if phase == "grasp":
            if strategy == 0:  # Lateral miss
                axis = np.random.choice([0, 1])
                pose[axis] += np.random.uniform(0.04, 0.08) \
                    * np.random.choice([-1, 1])
            elif strategy == 1:  # Depth overshoot / undershoot
                direction = t3d.quaternions.quat2mat(pose[3:7])[:, 0]
                offset = np.random.uniform(0.04, 0.08) \
                    * np.random.choice([-1, 1])
                pose[:3] += offset * direction
            elif strategy == 2:  # Wrong approach angle
                angle = np.random.uniform(np.pi / 4, np.pi / 2) \
                    * np.random.choice([-1, 1])
                rot_q = t3d.euler.euler2quat(0, 0, angle)
                pose[3:7] = t3d.quaternions.qmult(pose[3:7], rot_q)
            elif strategy == 3:  # Height miss + small lateral
                pose[2] += np.random.uniform(0.03, 0.07) \
                    * np.random.choice([-1, 1])
                pose[np.random.choice([0, 1])] += \
                    np.random.uniform(-0.02, 0.02)

        elif phase == "place":
            if strategy == 0:  # Miss basket laterally
                axis = np.random.choice([0, 1])
                pose[axis] += np.random.uniform(0.06, 0.12) \
                    * np.random.choice([-1, 1])
            elif strategy == 1:  # Too high / too low
                pose[2] += np.random.uniform(0.05, 0.10) \
                    * np.random.choice([-1, 1])
            elif strategy == 2:  # Diagonal miss
                pose[0] += np.random.uniform(0.04, 0.08) \
                    * np.random.choice([-1, 1])
                pose[1] += np.random.uniform(0.04, 0.08) \
                    * np.random.choice([-1, 1])
            elif strategy == 3:  # Wrong orientation
                angle = np.random.uniform(np.pi / 4, np.pi / 2) \
                    * np.random.choice([-1, 1])
                rot_q = t3d.euler.euler2quat(angle, 0, 0)
                pose[3:7] = t3d.quaternions.qmult(pose[3:7], rot_q)

        return pose.tolist()

    @staticmethod
    def _get_neg_lift_params(strategy_idx):
        """Return perturbed displacement kwargs for negative lift."""
        strategies = [
            {'z': 0.02},                         # Insufficient lift
            {'z': -0.03},                         # Move DOWN
            {'x': 0.06, 'z': 0.02},              # Lateral drift + low lift
            {'y': 0.06 * np.random.choice([-1, 1]),
             'z': 0.01},                          # Sideways drift
        ]
        return strategies[strategy_idx % len(strategies)]


# ==================== DataProcessor ====================

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
        embodiment_config_path = os.path.join(
            CONFIGS_PATH, "_embodiment_config.yml")
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
            self.args["task_config"])

    def run_two_stage_collection(self):
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

                self.env.play_once()

                if self.env._phase_records and self.env.check_success():
                    print(f"\n--- Generating Positive Samples ---")
                    self.env.sample_pos_from_phase_states()
                    print(f"--- Positive Sampling Complete ---")

                    print(f"\n--- Generating Negative Samples ---")
                    self.env.sample_neg_from_phase_states()
                    print(f"--- Negative Sampling Complete ---\n")

            print(f"Merging to HDF5 for episode {epid}")
            self.env.close_env()
            self.env.merge_pkl_to_hdf5_video()

            if not self.env.check_success():
                print(f"Warning: Episode {epid} did not succeed!")


if __name__ == "__main__":
    processor = DataProcessor(
        "place_bread_basket",
        "data/place_bread_basket/data",
        "data/processed")
    processor.run_two_stage_collection()
    print("Data collection complete.")
