import os
import time
from collections import OrderedDict
from collections import deque

import gym
from gym import spaces
import numpy as np
import pybullet
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R

from lifelike.sim_envs.pybullet_envs.legged_robot.legged_robot import LeggedRobot
from lifelike.sim_envs.pybullet_envs.primitive_level_env.motion_lib import MotionLib
from lifelike.utils.obstacle import get_obstacle_pose
from lifelike.sim_envs.pybullet_envs.legged_robot import get_urdf_path


def quat2axisangle(quat):
    rotvec = R.from_quat(quat).as_rotvec()
    angle = np.sqrt(np.sum(rotvec ** 2))
    axis = rotvec / (angle + 1e-8)
    return axis, angle


class PrimitiveLevelEnv(gym.Env):
    def __init__(self,
                 enable_render=False,
                 control_freq=50.0,
                 sim_freq=500.0,
                 kp=50.0,
                 kd=0.5,
                 foot_lateral_friction=0.5,
                 max_tau=18,
                 enable_gui=True,
                 video_path=None,
                 data_path="",
                 prop_type=None,
                 stack_frame_num=3,
                 prioritized_sample_factor=0.0,
                 set_obstacle=False,
                 obstacle_height=0.2,
                 reward_weights=None):

        self._enable_render = enable_render
        # Outer loop controller time step
        self._policy_step = 1.0 / control_freq
        # Inner loop PD controller time step
        self._time_step = 1.0 / sim_freq

        # Number of time of inner loops within one out loop
        self.num_env_steps = int(self._policy_step / self._time_step)
        if video_path is not None:
            assert isinstance(video_path, str) and video_path.endswith('.mp4')
            self._enable_render = True
        self._video_path = video_path

        if self._enable_render:
            self._bullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            self._bullet_client.configureDebugVisualizer(self._bullet_client.COV_ENABLE_RENDERING, 0)
        else:
            self._bullet_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

        self._legged_robot = LeggedRobot(
            bullet_client=self._bullet_client,
            time_step=self._time_step,
            mode="dynamic",
            init_pos=[0.0, 0.0, 0.0],
            init_orn=[0.0, 0.0, 0.0, 1.0],
            kp=kp,
            kd=kd,
            foot_lateral_friction=foot_lateral_friction,
            max_tau=max_tau)
        self._legged_robot_kin = LeggedRobot(
            bullet_client=self._bullet_client,
            time_step=0,
            mode="kinematic",
            init_pos=[0.0, 0.0, 0.5],
            init_orn=[0.0, 0.0, 0.0, 1.0],
        )
        plane_path = os.path.join(get_urdf_path(), "plane.urdf")
        self._terrain_id = self._bullet_client.loadURDF(plane_path)

        self._max_tau = max_tau

        if self._enable_render:
            robot_pos, _ = self._bullet_client.getBasePositionAndOrientation(self._legged_robot.robot_id)
            self._bullet_client.resetDebugVisualizerCamera(1.5, 0, -10, robot_pos)  # near view
            # self._bullet_client.resetDebugVisualizerCamera(5, 0, -45, robot_pos)  # far view
            self._bullet_client.configureDebugVisualizer(self._bullet_client.COV_ENABLE_RENDERING, 1)
            if enable_gui:
                self._bullet_client.configureDebugVisualizer(self._bullet_client.COV_ENABLE_GUI, 0)
            if self._video_path is not None:
                self.video_id = self._bullet_client.startStateLogging(
                    self._bullet_client.STATE_LOGGING_VIDEO_MP4, video_path)

        self._prop_type = prop_type
        self._stack_frame_num = stack_frame_num
        self._reward_weights = reward_weights

        if isinstance(prop_type, list):
            full_prop_size = {
                'joint_pos': 12,
                'joint_vel': 12,
                'root_lin_vel_loc': 3,
                'root_ang_vel_loc': 3,
                'e_g': 3,
            }
            prop_size = 0
            for e in prop_type:
                prop_size += full_prop_size[e]
        else:
            raise TypeError("Expected 'prop_type' to be a list.")
        prop_size *= self._stack_frame_num
        prop_a_size = 12 * self._stack_frame_num

        dict_obs_space = OrderedDict({
            'prop': spaces.Box(0, 0, shape=(prop_size,)),
            'prop_a': spaces.Box(0, 0, shape=(prop_a_size,)),
            'future': spaces.Box(0, 0, shape=(72,)),
        })

        self.observation_space = spaces.Dict(dict_obs_space)
        self.action_space = spaces.Box(0, 0, shape=(12,))

        self._obstacle_id = None
        self._obstacle = None

        self._motion_generator = MotionLib(data_path, self._policy_step)

        self._max_steps = self._motion_generator.max_steps
        self._prioritized_sample_probability = self._motion_generator.prioritized_sample_probability
        self._avg_reward_sum = np.zeros_like(self._prioritized_sample_probability)
        self.avg_episode_len = np.zeros_like(self._prioritized_sample_probability)
        self.sampled_data_idx = None
        self._prioritized_sample_factor = prioritized_sample_factor
        self._episode_steps = 0
        self.reward_sum = 0.0
        self.time = 0

        self._set_obstacle = set_obstacle
        self._obstacle_height = obstacle_height

        # stack states
        self.history_props = deque(maxlen=self._stack_frame_num)
        # store previous and current rl action
        self.history_actions = deque(maxlen=max(2, self._stack_frame_num))
        self.last_action = None

    def reset(self):
        self.last_action = None
        # reset tau_max
        self._legged_robot.max_taus = [self._max_tau if not isinstance(self._max_tau, list) else np.random.uniform(
            *self._max_tau)] * len(self._legged_robot.actuated_joint_indices)
        self.reward_sum = 0.0

        kin_states_info = self._reset_kin_robot_state()

        if self._set_obstacle:
            self._create_obstacle()
        self._episode_steps = 0
        self._legged_robot_kin.set_states_info(kin_states_info)
        self._legged_robot.set_states_info(kin_states_info)

        states_info = self._legged_robot.get_states_info()
        states_info_target_future = self._motion_generator.get_states_info_future()

        self.history_props.clear()
        self.history_actions.clear()
        obs = self._prepare_obs(states_info, states_info_target_future, np.zeros(12, ))
        return obs

    def _create_obstacle(self):
        if self._obstacle_id is not None:
            self._bullet_client.removeBody(self._obstacle_id)

        self._obstacle = self._motion_generator.obstacle
        if self._obstacle is not None:
            self.ob_id = 0
            # self._obstacle_id = self._bullet_client.loadURDF(os.path.join(get_urdf_path(), "cube.urdf"),
            #                                                basePosition=[10, 10, 0], globalScaling=1)
            self.obstacle_shape = self._bullet_client.createCollisionShape(
                shapeType=self._bullet_client.GEOM_BOX,
                halfExtents=[0.025, 0.5, self._obstacle_height],
            )
            self._obstacle_id = self._bullet_client.createMultiBody(0, self.obstacle_shape,
                                                                  basePosition=[10, 10, 10],
                                                                  baseOrientation=[0, 0, 0, 1])
            self._bullet_client.changeVisualShape(self._obstacle_id, -1, rgbaColor=[1, 1, 1, 1])

            pos, orn = get_obstacle_pose(self._obstacle['pos'][self.ob_id, :],
                                         self._obstacle['orn_otho'][self.ob_id, :])
            self._bullet_client.resetBasePositionAndOrientation(self._obstacle_id, pos, orn)

    def step(self, rl_action):
        start_time = time.time()
        self._episode_steps += 1
        action = np.array(rl_action)
        joint_pos, _ = self._legged_robot.get_joint_states_info()
        tgt_joint_pos = np.array(joint_pos) + action
        # apply inner loop action
        for _ in range(self.num_env_steps):
            self._bullet_client.setTimeStep(self._time_step)
            self._legged_robot.apply_action(tgt_joint_pos)

            self._bullet_client.stepSimulation()

            self._motion_generator.step(self.time)

            self.time += self._time_step

        if self._enable_render:
            self._legged_robot.camera_on()

        self.last_action = action.copy()
        # update kinematic states
        state_info = self._motion_generator.get_states_info()
        self._legged_robot_kin.set_states_info(state_info)

        # get state
        states_info = self._legged_robot.get_states_info()
        states_info_target_future = self._motion_generator.get_states_info_future()

        if self._set_obstacle:
            self._update_obstacle()

        obs = self._prepare_obs(states_info, states_info_target_future, rl_action)

        # compute reward
        reward = self._compute_reward()
        self.reward_sum += reward
        done = self._check_terminate(states_info)

        info = {}
        if done:
            self._avg_reward_sum[self.sampled_data_idx] = self.reward_sum / self._max_steps[self.sampled_data_idx]
            self.avg_episode_len[self.sampled_data_idx] = self._episode_steps / (
                    self._max_steps[self.sampled_data_idx] + 1)
            self._prioritized_sample_probability[:] = (1 - self._avg_reward_sum) ** self._prioritized_sample_factor
            self._prioritized_sample_probability[:] /= np.sum(self._prioritized_sample_probability, axis=0)
        elapsed_time = time.time() - start_time
        sleep_time = self._policy_step - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        return obs, reward, done, info

    def _prepare_full_prop(self, states_info):
        return OrderedDict({
            'joint_pos': states_info["joint_pos"],
            'joint_vel': states_info["joint_vel"],
            'root_lin_vel_loc': list(R.from_quat(states_info["base_orn"]).inv().apply(states_info["base_lin_vel"])),
            'root_ang_vel_loc': list(R.from_quat(states_info["base_orn"]).inv().apply(states_info["base_ang_vel"])),
            'e_g': R.from_quat(states_info['base_orn']).as_matrix()[2, :],
        })

    def _cfg_prop(self, full_prop: dict, keys: list):
        prop = []
        for e in keys:
            prop.append(np.array(full_prop[e]))
        return np.concatenate(prop, axis=-1)

    def _update_obstacle(self):
        if self._obstacle is not None:
            while self.ob_id < len(self._obstacle['time']) - 1 and self.time > self._obstacle['time'][self.ob_id] + 0.5:
                self.ob_id += 1
            pos, orn = get_obstacle_pose(self._obstacle['pos'][self.ob_id, :],
                                         self._obstacle['orn_otho'][self.ob_id, :])
            self._bullet_client.resetBasePositionAndOrientation(self._obstacle_id, pos, orn)

    def _reset_kin_robot_state(self):
        self.time, kin_states_info = self._motion_generator.reset()
        self._legged_robot_kin.set_states_info(kin_states_info)
        self.sampled_data_idx = self._motion_generator.sampled_data_idx
        return kin_states_info

    def _prepare_obs(self, states_info, states_info_target_future, action):
        future = self.calculate_future(list(states_info['base_pos']),
                                       list(states_info['base_orn']),
                                       states_info_target_future)
        prop = self._cfg_prop(self._prepare_full_prop(states_info), self._prop_type)

        while len(self.history_props) < self._stack_frame_num:
            self.history_props.append(prop)
        self.history_props.append(prop)
        props = np.concatenate(self.history_props, axis=-1)

        while len(self.history_actions) < self._stack_frame_num:
            self.history_actions.append(action)
        self.history_actions.append(action)
        actions = np.concatenate(self.history_actions)

        obs = OrderedDict({
            'prop': props,
            'prop_a': actions,
            'future': future,
        })
        return obs

    @staticmethod
    def calculate_future(base_pos, base_orn, states_info_future):
        goals_local = []
        r_b = R.from_quat(base_orn)
        for i in range(len(states_info_future)):
            pos_i = states_info_future[i]['base_pos']
            quat_i = states_info_future[i]['base_orn']
            joint_pos_i = states_info_future[i]['joint_pos']
            rotation_diff_i = (r_b.inv() * R.from_quat(quat_i)).as_quat()
            axis, angle = quat2axisangle(rotation_diff_i)

            pos_diff_world = [a-b for a, b in zip(pos_i, base_pos)]
            pos_diff_base = r_b.inv().apply(pos_diff_world)
            goals_local += list(pos_diff_base)
            goals_local += [axis[0] * angle, axis[1] * angle, axis[2] * angle]

            goals_local += joint_pos_i

        return np.array(goals_local)

    def _check_dyn_kin_difference(self, state_info_dyn):
        max_pose_err = 1.0
        max_pos_err = 1.0

        terminates = False
        state_info_kin = self._legged_robot_kin.get_states_info()
        base_pos1, base_quat1 = state_info_dyn['base_pos'], state_info_dyn['base_orn']
        base_pos2, base_quat2 = state_info_kin['base_pos'], state_info_kin['base_orn']
        quat_diff = (R.from_quat(base_quat2) * R.from_quat(base_quat1).inv()).as_quat()
        _, angle = quat2axisangle(quat_diff)
        if np.abs(angle) > max_pose_err:
            terminates = True

        """ terminate when position error is too large """
        if np.sum(np.square(np.array(base_pos1) - np.array(base_pos2))) > max_pos_err:
            terminates = True
        return terminates

    def _check_terminate(self, states_info):
        done_dyn = self._legged_robot.check_terminate(states_info)
        done_kin = self._motion_generator.is_ended()
        done_dyn_kin_diff = self._check_dyn_kin_difference(states_info)
        done_collision = False
        if self._obstacle is not None:
            contacts = self._bullet_client.getContactPoints(bodyA=self._legged_robot.robot_id)
            for contact in contacts:
                if contact[2] == self._obstacle_id:
                    done_collision = True
        done = done_dyn or done_kin or done_dyn_kin_diff or done_collision
        return done

    def _compute_reward(self):
        """ weights of rewards """
        if self._reward_weights is not None:
            weight_joint_pos = self._reward_weights['joint_pos']
            weight_joint_vel = self._reward_weights['joint_vel']
            weight_end_effector = self._reward_weights['end_effector']
            weight_root_pose = self._reward_weights['root_pose']
            weight_root_vel = self._reward_weights['root_vel']
        else:
            weight_joint_pos = 0.6  # 0.5
            weight_joint_vel = 0.05
            weight_end_effector = 0.1  # 0.2
            weight_root_pose = 0.15
            weight_root_vel = 0.1

        sum_weight = weight_joint_pos + weight_joint_vel + weight_end_effector + weight_root_pose + weight_root_vel
        weight_joint_pos /= sum_weight
        weight_joint_vel /= sum_weight
        weight_end_effector /= sum_weight
        weight_root_pose /= sum_weight
        weight_root_vel /= sum_weight

        """ scaling factors of exponent """
        scale_joint_pos = -1.0  # -5.0
        scale_joint_vel = -0.1
        scale_end_effector = -40.0
        scale_root_pose = [-20.0, -10.0]
        scale_root_vel = [-2, -0.2]

        """ calculate reward for joint positions """

        states_info_dyn = self._legged_robot.get_states_info()
        states_info_kin = self._legged_robot_kin.get_states_info()

        jp1 = np.array(states_info_dyn['joint_pos'])
        jp2 = np.array(states_info_kin['joint_pos'])
        reward_joint_pos = np.exp(scale_joint_pos * np.sum(np.square(jp1 - jp2)))

        """ calculate reward for joint velocities """
        jv1 = np.array(states_info_dyn['joint_vel'])
        jv2 = np.array(states_info_kin['joint_vel'])
        reward_joint_vel = np.exp(scale_joint_vel * np.sum(np.square(jv1 - jv2)))

        """ calculate reward for end effector positions """
        base_pos1, base_quat1 = states_info_dyn['base_pos'], states_info_dyn['base_orn']
        base_pos2, base_quat2 = states_info_kin['base_pos'], states_info_kin['base_orn']

        self._legged_robot.compute_end_effector_info()
        self._legged_robot_kin.compute_end_effector_info()
        pd1 = self._legged_robot.end_effector_position
        pd2 = self._legged_robot_kin.end_effector_position

        reward_end_effector = np.exp(scale_end_effector * np.sum(np.square(pd1 - pd2)))

        """ calculate reward for root pose and velocity """
        base_trans_vel1, base_euler_vel1 = states_info_dyn['base_lin_vel'], states_info_dyn['base_ang_vel']
        base_trans_vel2, base_euler_vel2 = states_info_kin['base_lin_vel'], states_info_kin['base_ang_vel']

        pos1, pos2 = np.array(base_pos1), np.array(base_pos2)

        quat_diff = (R.from_quat(base_quat2) * R.from_quat(base_quat1).inv()).as_quat()
        _, angle = quat2axisangle(quat_diff)

        reward_root_pose = np.exp(scale_root_pose[0] * np.sum(np.square(pos1 - pos2)) +
                                  scale_root_pose[1] * np.square(angle))

        tvel1, tvel2 = np.array(base_trans_vel1), np.array(base_trans_vel2)
        evel1, evel2 = np.array(base_euler_vel1), np.array(base_euler_vel2)
        reward_root_vel = np.exp(scale_root_vel[0] * np.sum(np.square(tvel1 - tvel2)) +
                                 scale_root_vel[1] * np.sum(np.square(evel1 - evel2)))

        reward = (weight_joint_pos * reward_joint_pos +
                  weight_joint_vel * reward_joint_vel +
                  weight_end_effector * reward_end_effector +
                  weight_root_pose * reward_root_pose +
                  weight_root_vel * reward_root_vel)
        return reward

    def close(self):
        if self._video_path is not None:
            self._bullet_client.stopStateLogging(self.video_id)
        self._bullet_client.disconnect()

    def __del__(self):
        if self._bullet_client.isConnected() > 0:
            self.close()
