from collections import OrderedDict
from collections import deque
from copy import deepcopy

import os
import gym
import numpy as np
import pybullet
from gym import spaces
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R
from lifelike.sim_envs.pybullet_envs.legged_robot.legged_robot import LeggedRobot
from lifelike.sim_envs.pybullet_envs.max_game.game_manager import GameManager
from lifelike.sim_envs.pybullet_envs.max_game_elements.playground_env import apply_zaxis_rotation
from lifelike.sim_envs.pybullet_envs.max_game_elements.playground_env import RayCast
from lifelike.utils.constants import compute_terrain_rectangle
from lifelike.sim_envs.pybullet_envs.max_game import get_urdf_path
from lifelike.sim_envs.pybullet_envs.randomizer.push_randomizer import PushRandomizer


class ChaseTagGameEnv(gym.Env):
    def __init__(self, enable_render=False,
                 control_freq=25.0,
                 kp=50.0, kd=1.0,
                 max_tau=16,
                 terrain_perception=None,
                 prop_type=None,
                 stack_frame_num=3,
                 n_max=2,
                 max_steps=1000,
                 visible_angle=np.pi,
                 obs_randomization=None,
                 env_randomize_config=None,
                 element_config=None):

        self._obs_randomization = obs_randomization
        self._episodic_noise = None
        if element_config is None:
            element_config = {}

        # bullet env part
        self._enable_render = enable_render
        if self._enable_render:
            self.bullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.bullet_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        self.bullet_client.configureDebugVisualizer(self.bullet_client.COV_ENABLE_RENDERING, 1)
        self._game_mgr = GameManager(self.bullet_client, version='v4', element_config=element_config)
        self.env_randomize_config = env_randomize_config

        # Inner loop PD controller time step
        self.time_step = 1.0 / 500.0
        # Outer loop controller time step
        self.policy_step = 1.0 / control_freq
        # Number of time of inner loops within one out loop
        self.num_env_steps = int(self.policy_step / self.time_step)

        colors = ['g', 'b']
        init_poss = [[1.0, 1.0, 0.5], [-1.0, -1.0, 0.5]]
        self.max_tau = max_tau
        self.foot_lateral_friction = np.random.uniform(*env_randomize_config['friction_range'])
        self.legged_robots = [LeggedRobot(
            bullet_client=self.bullet_client,
            time_step=self.time_step,
            mode="dynamic",
            init_pos=init_poss[_],
            init_orn=[0.0, 0.0, 0.0, 1.0],
            kp=kp,
            kd=kd,
            foot_lateral_friction=self.foot_lateral_friction,
            max_tau=max_tau,
            color=colors[_]) for _ in range(2)]
        self.flag_id = self._create_flag()
        self.rays_casts = [RayCast(num_rays=128, len_ray=20.0, bullet_client=self.bullet_client, color=color)
                           for color in colors]
        # agent part
        self.with_flag = [True, False]
        self.prop_type = prop_type
        self.terrain_perception = terrain_perception
        self.stack_frame_num = stack_frame_num
        self.n_max = n_max
        self.max_steps = max_steps
        self._visible_angle = visible_angle
        self.oppo_visible = [True, True]
        self.flag_visible = [True, True]
        self.last_two_rob_pos_diff_len = 0.0
        self.last_esc_flag_pos_diff_len = 0.0
        self.switch_flag_at_this_frame = False
        # obs
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
        prop_size *= self.stack_frame_num
        prop_a_size = 12 * self.stack_frame_num
        percept_2d_size = (25, 13)
        percept_1d_size = 128
        percept_vec_size = 5
        oppo_info_size = 15
        flag_info_size = 7
        with_flag_size = 2
        dict_obs_space = OrderedDict({
            'prop': spaces.Box(0, 0, shape=(prop_size,)),
            'prop_a': spaces.Box(0, 0, shape=(prop_a_size,)),
            'percept_2d': spaces.Box(0, 0, shape=percept_2d_size),
            'percept_1d': spaces.Box(0, 0, shape=(percept_1d_size,)),
            'percept_front': spaces.Box(0, 0, shape=percept_2d_size),
            'percept_vec': spaces.Box(0, 0, shape=(percept_vec_size,)),
            'oppo_info': spaces.Box(0, 0, shape=(oppo_info_size,)),
            'oppo_info_cheat': spaces.Box(0, 0, shape=(oppo_info_size,)),
            'flag_info': spaces.Box(0, 0, shape=(flag_info_size,)),
            'flag_info_cheat': spaces.Box(0, 0, shape=(flag_info_size,)),
            'with_flag': spaces.Box(0, 0, shape=(with_flag_size,)),
            'control_spd': spaces.Box(0, 0, shape=(1,)),
        })
        self.observation_space = spaces.Tuple([spaces.Dict(dict_obs_space)] * self.n_max)

        # act
        joystick_dim = 1
        action_dim = 12
        self.action_space = spaces.Tuple(
            [spaces.Dict(OrderedDict({
                'A_HLC': spaces.Box(0, 0, shape=(joystick_dim,)),
                'A_Z': spaces.Discrete(256),
                'A_LLC': spaces.Box(0, 0, shape=(action_dim,))
            }))] * self.n_max)

        self.time = 0
        self.counter = 0
        self.history_props = [deque(maxlen=self.stack_frame_num), deque(maxlen=self.stack_frame_num)]
        self.history_actions = [deque(maxlen=self.stack_frame_num), deque(maxlen=self.stack_frame_num)]
        self.last_action = None
        self.total_spds = [0.0, 0.0]
        self.max_spds = [0.0, 0.0]

        # for region selection and importance sampling
        self._region_id = -1
        self._num_region = 10
        self._region_hist_ret = [deque(maxlen=10) for _ in range(self._num_region)]
        self._region_ave_ret = np.zeros(shape=(self._num_region,))
        self._region_prob = np.ones(shape=(self._num_region,)) / float(self._num_region)
        self._dynamic_sam_pow = 3

        # adaptation
        if 'disturb_force_config' in env_randomize_config:
            self.force_randomizer = PushRandomizer(env=self, **env_randomize_config['disturb_force_config'])
        else:
            self.force_randomizer = None

        # arrow
        if self._enable_render:
            self._load_robot_render_flag()

    def _create_flag(self):
        length = 0.1
        width = 0.1
        height = 0.5
        visual_shape = self.bullet_client.createVisualShape(
            shapeType=self.bullet_client.GEOM_BOX,
            halfExtents=[
                length / 2,
                width / 2,
                height / 2,
            ],
            rgbaColor=[1, 0, 0, 1]
        )
        collision_shape = self.bullet_client.createCollisionShape(
            shapeType=self.bullet_client.GEOM_BOX,
            halfExtents=[
                length / 2,
                width / 2,
                height / 2,
            ],
        )
        position = [0.0, 0.0, height / 2]
        flag_id = self.bullet_client.createMultiBody(
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
        )
        return flag_id

    def _load_robot_render_flag(self):
        self.robot_render_flag_id = self.bullet_client.loadURDF(os.path.join(get_urdf_path(), "flag.urdf"))

    def _set_robot_render_flag(self, pos):
        self.bullet_client.resetBasePositionAndOrientation(
            self.robot_render_flag_id, pos, [0, 0, 0, 1], physicsClientId=self.bullet_client._client)

    def sample_episodic_noise(self):
        if self._obs_randomization:
            self._episodic_noise = {
                k: np.random.uniform(*self._obs_randomization[k]) for k in self._obs_randomization}

    def randomize_init_states(self):
        init_poss = [[np.random.uniform(-2.0, 2.0), np.random.uniform(-2.0, 2.0), 0.5],
                     [np.random.uniform(-2.0, 2.0), np.random.uniform(-2.0, 2.0), 0.5]]
        self.last_two_rob_pos_diff_len = np.linalg.norm((np.array(init_poss[1]) - np.array(init_poss[0]))[:2])
        # randomize pos and orn
        for _ in range(self.n_max):
            zaxis_angle = 360 * np.random.rand()
            init_states_info = self.legged_robots[_].get_init_states_info()
            init_states_info['base_orn'] = list(
                apply_zaxis_rotation(zaxis_angle, np.array(init_states_info['base_orn'])))
            init_states_info['base_pos'] = init_poss[_]
            self.legged_robots[_].set_states_info(init_states_info)
        self._randomize_flag_pos()

    def _randomize_flag_pos(self):
        # randomize flag pos
        flag_pos = [np.random.uniform(-2.0, 2.0), np.random.uniform(-2.0, 2.0), 0.25]  # half height of flag
        self.bullet_client.resetBasePositionAndOrientation(
            self.flag_id, flag_pos, [0.0, 0.0, 0.0, 1.0])
        self.target_pos = flag_pos

        if self.with_flag[0]:
            self.last_esc_flag_pos_diff_len = np.linalg.norm(
                (np.array(flag_pos) - np.array(self.legged_robots[1].get_states_info()['base_pos']))[:2])
        else:
            self.last_esc_flag_pos_diff_len = np.linalg.norm(
                (np.array(flag_pos) - np.array(self.legged_robots[0].get_states_info()['base_pos']))[:2])

    def _update_colors(self):
        self.default_colors = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
        # MAX colors
        if self.with_flag[0]:
            if self.oppo_visible[0]:
                for j in range(-1, self.legged_robots[1].num_joints):
                    self.bullet_client.changeVisualShape(
                        self.legged_robots[1].robot_id, j, rgbaColor=self.default_colors[1] + [1])
            else:
                for j in range(-1, self.legged_robots[1].num_joints):
                    self.bullet_client.changeVisualShape(
                        self.legged_robots[1].robot_id, j, rgbaColor=self.default_colors[1] + [0.1])
        else:
            if self.oppo_visible[1]:
                for j in range(-1, self.legged_robots[0].num_joints):
                    self.bullet_client.changeVisualShape(
                        self.legged_robots[0].robot_id, j, rgbaColor=self.default_colors[0] + [1])
            else:
                for j in range(-1, self.legged_robots[0].num_joints):
                    self.bullet_client.changeVisualShape(
                        self.legged_robots[0].robot_id, j, rgbaColor=self.default_colors[0] + [0.1])

    def _update_robot_render_flag(self):
        for _ in range(2):
            if self.with_flag[_]:
                states_info = self.legged_robots[_].get_states_info()
                pos = states_info['base_pos']
                self._set_robot_render_flag([pos[0], pos[1], pos[2] + 0.3])

    def reset(self, **kwargs):
        self.episodic_fix_spd = np.random.uniform(0.5, 3.0)
        self.total_spds = [0.0, 0.0]
        self.max_spds = [0.0, 0.0]
        self._game_mgr.static_objs.reset()
        with_flag = np.random.randint(0, 2)
        self.with_flag = [bool(with_flag), bool(1 - with_flag)]
        self.switch_flag_at_this_frame = False

        if self._enable_render:
            self.last_oppo_visible = [True, True]
            self._update_colors()

        self.counter = 0
        self.last_action = None
        # randomize friction
        foot_friction = np.random.uniform(*self.env_randomize_config['friction_range'])
        for rob in self.legged_robots:
            rob.set_foot_lateral_friction(foot_lateral_friction=foot_friction)
        print('Current episodic friction: {}'.format(foot_friction))
        # randomize force
        if self.force_randomizer is not None:
            self.force_randomizer.reset()
        # reset tau_max
        for robot in self.legged_robots:
            robot.max_taus = [self.max_tau if not isinstance(self.max_tau, list)
                              else np.random.uniform(*self.max_tau)] * robot.num_joints
        # reset state
        self.time = 0

        self.sample_episodic_noise()
        # randomize init states for both robots
        self.randomize_init_states()

        states = []
        for i in range(self.n_max):
            states.append(self.legged_robots[i].get_states_info())
            self.history_props[i].clear()
            self.history_actions[i].clear()
        states_infos = [robot.get_states_info() for robot in self.legged_robots]
        drill = self._prepare_drill(states_infos)
        # masks = self._prepare_action_masks(curr_pos)
        obs = self._prepare_obs(states, drill, [np.zeros(12, ), np.zeros(12, )])
        return obs

    def _get_e_g(self, states_info):
        base_quat = states_info['base_orn']
        rotation = R.from_quat(base_quat)
        base_rot_mat = rotation.as_matrix()
        e_g = base_rot_mat[2, :]
        return e_g

    def _prepare_full_prop(self, states_info):
        return OrderedDict({
            'joint_pos': states_info["joint_pos"],
            'joint_vel': states_info["joint_vel"],
            'root_lin_vel_loc': list(R.from_quat(states_info["base_orn"]).inv().apply(states_info["base_lin_vel"])),
            'root_ang_vel_loc': list(R.from_quat(states_info["base_orn"]).inv().apply(states_info["base_ang_vel"])),
            'e_g': self._get_e_g(states_info),
        })

    def _cfg_prop(self, full_prop: dict, keys: list):
        prop = []
        for e in keys:
            prop.append(np.array(full_prop[e]))
        return np.concatenate(prop, axis=-1)

    def _prepare_obs(self, states, drill, actions):
        props_list = []
        actions_list = []
        for i in range(self.n_max):
            if isinstance(self.prop_type, list):
                prop = self._cfg_prop(self._prepare_full_prop(states[i]), self.prop_type)
            else:
                raise NotImplementedError

            while len(self.history_props[i]) < self.stack_frame_num:
                self.history_props[i].append(prop)
            self.history_props[i].append(prop)
            props = np.concatenate(self.history_props[i], axis=-1)
            props_list.append(props)

            while len(self.history_actions[i]) < self.stack_frame_num:
                self.history_actions[i].append(actions[i])
            self.history_actions[i].append(actions[i])
            acts = np.concatenate(self.history_actions[i])
            actions_list.append(acts)

        obs = [OrderedDict({
            'prop': p,
            'prop_a': a,
            'percept_2d': p_2d,
            'percept_1d': p_1d,
            'percept_front': p_f,
            'percept_vec': p_vec,
            'oppo_info': np.array(o),
            'oppo_info_cheat': np.array(oc),
            'flag_info': np.array(f),
            'flag_info_cheat': np.array(fc),
            'with_flag': np.array(wf),
            'control_spd': np.array([self.env_randomize_config.get('control_spd', self.episodic_fix_spd)]),
        }) for p, a, p_2d, p_1d, p_f, p_vec, o, oc, f, fc, wf in zip(
            props_list, actions_list, *drill.values())]
        return obs

    def bullet_update(self):
        if self.force_randomizer is not None:
            self.force_randomizer.step()
        self.bullet_client.stepSimulation()

    def stat_spd(self, states_infos):
        current_vel = [np.linalg.norm(np.array(list(state_info['base_lin_vel']))[:2]) for state_info in states_infos]
        for _ in range(len(current_vel)):
            self.total_spds[_] += current_vel[_]
            if current_vel[_] > self.max_spds[_]:
                self.max_spds[_] = current_vel[_]

    def step(self, rl_actions: list):
        rl_actions = [np.array(a['A_LLC']) for a in rl_actions]
        tgt_joint_pos = [robot.get_joint_states_info()[0] + act for act, robot in zip(rl_actions, self.legged_robots)]

        # apply inner loop action
        for _ in range(self.num_env_steps):
            self.bullet_client.setTimeStep(self.time_step)
            for i in range(self.n_max):
                self.legged_robots[i].apply_action(tgt_joint_pos[i])
            self.bullet_update()
            self.time += self.time_step
        if self._enable_render:
            # self.env.legged_robots[0].camera_on()
            # joystick arrow
            for _ in range(2):
                self._update_robot_render_flag()

        # get state
        states_infos = [robot.get_states_info() for robot in self.legged_robots]
        drill = self._prepare_drill(states_infos)

        obs = self._prepare_obs(states_infos, drill, rl_actions)
        self.stat_spd(states_infos)

        # info
        self.counter += 1
        done, done_dict = self._check_terminate(states_infos)
        rewards = self._compute_all_rewards(states_infos)
        info = {
            'avg_spd0': self.total_spds[0] / self.counter,
            'avg_spd1': self.total_spds[1] / self.counter,
            'max_spd0': self.max_spds[0],
            'max_spd1': self.max_spds[1],
        }
        if done:
            if done_dict['done_contact']:
                if self.with_flag[0]:
                    rewards[0] += 1.0
                    rewards[1] -= 1.0
                else:
                    rewards[0] -= 1.0
                    rewards[1] += 1.0

        assert len(obs) == len(rewards) == 2
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        return obs, rewards, done, info

    def _detect_body_contact(self, robot):
        body_indices = robot.leg_indices + robot.wheel_indices
        contacts = self.bullet_client.getContactPoints()
        for c in contacts:
            if c[1] == robot.robot_id and c[2] == robot.robot_id:
                continue
            elif c[1] == robot.robot_id:
                robot_contact_index = c[3]
                if robot_contact_index in body_indices:
                    return c[2]
            elif c[2] == robot.robot_id:
                robot_contact_index = c[4]
                if robot_contact_index in body_indices:
                    return c[1]
        return None

    def _check_contact_status(self, robot):
        robot_contact_who = self._detect_body_contact(robot)
        if robot_contact_who is None:
            return False
        robot_ids = [self.legged_robots[0].robot_id, self.legged_robots[1].robot_id]
        if robot_contact_who in robot_ids and robot_contact_who != robot.robot_id:
            return True  # two robots contact
        else:
            return False  # trivial contact

    def _check_contact_flag(self, robot):
        robot_contact_who = self._detect_body_contact(robot)
        if robot_contact_who == self.flag_id:
            return True
        return False

    def _check_terminate(self, states_infos):
        done_robot = np.array([robot.check_terminate(info) for robot, info in zip(self.legged_robots, states_infos)])
        done_time = self.counter >= self.max_steps

        done_contact = self._check_contact_status(self.legged_robots[0])

        done = bool(done_robot[0] or done_time or done_contact)
        done_dict = {
            "done_robot": done_robot,
            "done_time": done_time,
            "done_contact": done_contact,
        }
        return done, done_dict

    def _ray_test_visible(self, current_position, global_pos_diff, yaws):
        convex_points_set = [rob.get_convex_points_position() for rob in self.legged_robots]
        head_points_set = [rob.get_head_points_position() for rob in self.legged_robots]
        root_ray = self.bullet_client.rayTest(
            np.array(current_position[0]), np.array(current_position[1]), collisionFilterMask=6)
        test_visible = [root_ray[0][0] < 0] * 2
        for i in range(len(test_visible)):
            if not test_visible[i]:
                for p1 in head_points_set[i]:
                    for p2 in convex_points_set[1 - i]:
                        point_ray = self.bullet_client.rayTest(
                            np.array(list(p1)), np.array(list(p2)), collisionFilterMask=6)
                        test_visible[i] |= (point_ray[0][0] < 0)
                        if test_visible[i]:
                            break
                    if test_visible[i]:
                        break
        test_visible = np.array(test_visible)
        cos_visible_thetas = [(np.cos(yaw) * pos_diff[0] + np.sin(yaw) * pos_diff[1]) / np.linalg.norm(pos_diff[0:2])
                              for pos_diff, yaw in zip(global_pos_diff, yaws)]
        oppo_visible = np.logical_and(cos_visible_thetas >= np.cos(self._visible_angle), test_visible)
        return oppo_visible

    def _prepare_drill(self, states_infos):
        self_percept_dict = OrderedDict()

        current_position = [list(info['base_pos']) for info in states_infos]
        rotations = [R.from_quat(info["base_orn"]) for info in states_infos]
        r_bs = [r.as_matrix() for r in rotations]
        yaws = [np.arctan2(r_b[1, 0], r_b[0, 0]) for r_b in r_bs]
        if self._episodic_noise:
            if 'pos_x_bias' in self._episodic_noise:
                current_position = [[pos[0] + self._episodic_noise['pos_x_bias'],
                                     pos[1] + self._episodic_noise['pos_y_bias'],
                                     pos[2]]
                                    for pos in current_position]
            if 'yaw_bias' in self._episodic_noise:
                yaws = [yaw + self._episodic_noise['yaw_bias'] for yaw in yaws]
        self_percept_dict['position'] = current_position
        self_percept_dict['yaw'] = [[np.cos(yaw), np.sin(yaw)] for yaw in yaws]
        # local 2d
        self_percept_front_2d = []
        for p, r in zip(current_position, rotations):
            front = self._get_perception_front(p, r)
            self_percept_front_2d.append(front)
        self_percept_2d = [self._get_perception_height_v2(p, r) for p, r in zip(current_position, rotations)]
        # local 1d ray
        hit_distances = []
        for i in range(len(self.rays_casts)):
            base_pos = current_position[i]
            yaw = yaws[i]
            rays_from, rays_to = self.rays_casts[i].compute_rays(*base_pos, yaw)
            rays_hit_correct = self.rays_casts[i].draw_rays(rays_from, rays_to)
            hit_distance = np.sum((np.array(rays_hit_correct) - np.array(rays_from)) ** 2, axis=-1) ** 0.5
            hit_distances.append(list(hit_distance))
        self_percept_1d = np.array(hit_distances)
        self_percept_vec = [np.concatenate(values) for values in zip(*self_percept_dict.values())]

        # opponent info
        opponent_state_dict = OrderedDict()
        # check if opponent is visible
        global_pos_diff = [np.array(pos2) - np.array(pos1)
                           for pos1, pos2 in zip(current_position, reversed(current_position))]
        # oppo and visible
        self.oppo_visible = self._ray_test_visible(current_position, global_pos_diff, yaws)
        if self._enable_render:
            if (self.with_flag[0] and self.oppo_visible[0] != self.last_oppo_visible[0]) or (
                self.with_flag[1] and self.oppo_visible[1] != self.last_oppo_visible[1]):
                self._update_colors()
                self.last_oppo_visible = deepcopy(self.oppo_visible)
        opponent_state_dict['oppo_visible'] = [[visible] for visible in self.oppo_visible.astype(float)]
        opponent_state_dict['oppo_pos'] = [p for p in reversed(current_position)]
        opponent_state_dict['oppo_pos_diff'] = [list(r.inv().apply(pos_diff))
                                                for pos_diff, r in zip(global_pos_diff, rotations)]
        opponent_state_dict['yaw_diff'] = [[np.cos(yaw2 - yaw1), np.sin(yaw2 - yaw1)]
                                           for yaw1, yaw2 in zip(yaws, reversed(yaws))]
        opponent_state_dict['local_base_lin_vel'] = [list(r.inv().apply(oppo_state_info['base_lin_vel']))
                                                     for oppo_state_info, r in zip(reversed(states_infos), rotations)]
        opponent_state_dict['local_base_ang_vel'] = [list(r.inv().apply(oppo_state_info['base_ang_vel']))
                                                     for oppo_state_info, r in zip(reversed(states_infos), rotations)]
        opponent_state = [np.concatenate(values) for values in zip(*opponent_state_dict.values())]
        oppo_info = [list(state) if visible else list(np.zeros_like(state))
                     for state, visible in zip(opponent_state, self.oppo_visible)]
        oppo_info_cheat = opponent_state

        # flag and visible
        flag_pos = self.target_pos

        self.flag_visible = np.array([True, True])

        flag_state_dict = OrderedDict()
        flag_state_dict['flag_visible'] = [[visible] for visible in self.flag_visible.astype(float)]
        flag_pos_diff = [np.array(flag_pos) - np.array(p) for p in current_position]
        flag_state_dict['flag_pos'] = [flag_pos] * 2
        flag_state_dict['flag_pos_diff'] = [list(r.inv().apply(pos_diff))
                                            for pos_diff, r in zip(flag_pos_diff, rotations)]
        flag_state = [np.concatenate(values) for values in zip(*flag_state_dict.values())]
        flag_info = [list(state) if visible else list(np.zeros_like(state))
                     for state, visible in zip(flag_state, self.flag_visible)]
        flag_info_cheat = flag_state
        # switch flag
        if (self.with_flag[0] and self._check_contact_flag(self.legged_robots[1])) or (
            self.with_flag[1] and self._check_contact_flag(self.legged_robots[0])):
            self.with_flag = self.with_flag[::-1]
            self.switch_flag_at_this_frame = True
            self._randomize_flag_pos()
            if self._enable_render:
                self._update_colors()
        else:
            self.switch_flag_at_this_frame = False

        with_flag_state = np.array([self.with_flag, self.with_flag[::-1]], dtype=np.float)

        drill_dict = OrderedDict({
            'self_percept_2d': self_percept_2d,
            'self_percept_1d': self_percept_1d,
            'self_percept_front_2d': self_percept_front_2d,
            'self_percept_vec': self_percept_vec,
            'oppo_info': oppo_info,
            'oppo_info_cheat': oppo_info_cheat,
            'flag_info': flag_info,
            'flag_info_cheat': flag_info_cheat,
            'with_flag': with_flag_state,
        })
        return drill_dict

    def _get_perception_front(self, base_pos, base_rotation):
        max_y = 0.25
        max_z = 0.3
        sample_rectangle = compute_terrain_rectangle(-max_y, max_y, 25, -max_z, 0.1, 13)
        sample_rectangle = np.reshape(sample_rectangle, (-1, 3))
        sample_rectangle_from = np.c_[sample_rectangle[:, 2],
                                      sample_rectangle[:, 0],
                                      sample_rectangle[:, 1]]
        sample_rectangle_to = np.c_[sample_rectangle[:, 2] + 3,
                                    sample_rectangle[:, 0],
                                    sample_rectangle[:, 1]]
        ray_from = base_rotation.apply(sample_rectangle_from) + base_pos
        ray_to = base_rotation.apply(sample_rectangle_to) + base_pos

        results = self.bullet_client.rayTestBatch(ray_from, ray_to, collisionFilterMask=6)

        ray_hit_position = np.array([r[3] for r in results])
        hit_body = np.array([r[0] for r in results])

        ray_hit_pos_correct = np.array([p1 if h > -1 else p2 for h, p1, p2 in zip(hit_body, ray_hit_position, ray_to)])
        ray_dist = np.linalg.norm(ray_hit_pos_correct - ray_from, axis=1)

        return ray_dist.reshape(25, 13)

    def _get_perception_height_v2(self, base_pos, base_rotation):
        max_x = 1.2
        max_y = 0.6
        sample_rectangle = compute_terrain_rectangle(-max_x, max_x, 25, -max_y, max_y, 13)
        sample_rectangle = np.reshape(sample_rectangle, (-1, 3))
        terrain_pos = base_rotation.apply(sample_rectangle) + base_pos
        terrain_pos = terrain_pos[:, :2]
        ray_from = [[x, y, 10.0] for x, y in terrain_pos]
        ray_to = [[x, y, -10.0] for x, y in terrain_pos]
        results = self.bullet_client.rayTestBatch(ray_from, ray_to, collisionFilterMask=6)
        ray_hit_z = np.array([r[3][2] for r in results])
        if self._episodic_noise and 'pos_z_bias' in self._episodic_noise:
            ray_hit_z = np.where(np.logical_and(ray_hit_z > 0.01, ray_hit_z < 0.6),  # TODO(lxhan): trick
                                 ray_hit_z + self._episodic_noise['pos_z_bias'],
                                 0.0 * np.ones_like(ray_hit_z))

        return ray_hit_z.reshape(25, 13)

    def _compute_all_rewards(self, states_infos):
        if self.with_flag[0]:  # MAX0 chases

            reward_chaser = 0.0
            reward_escapee = 0.0
            reward = [reward_chaser + float(self.switch_flag_at_this_frame),
                      reward_escapee - float(self.switch_flag_at_this_frame)]
        else:
            reward_chaser = 0.0
            reward_escapee = 0.0
            reward = [reward_escapee - float(self.switch_flag_at_this_frame),
                      reward_chaser + float(self.switch_flag_at_this_frame)]
        return reward

    def _compute_chaser_reward(self, chaser_state, escapee_state):
        # dist reward
        current_positions = [list(info['base_pos']) for info in [chaser_state, escapee_state]]
        global_pos_diff = (np.array(current_positions[1]) - np.array(current_positions[0]))[:2]
        pos_diff_len = np.linalg.norm(global_pos_diff)
        reward_dist = (pos_diff_len - self.last_two_rob_pos_diff_len) * 100  # when visible, max dist should be 10
        self.last_two_rob_pos_diff_len = pos_diff_len

        reward_chaser = -reward_dist
        reward_chaser /= float(self.max_steps)
        return reward_chaser

    def _compute_escapee_reward(self, escapee_state, flag_visible):
        flag_state = {'base_pos': self.target_pos,
                      'base_lin_vel': [0.0, 0.0, 0.0]}

        current_positions = [list(info['base_pos']) for info in [escapee_state, flag_state]]
        global_pos_diff = (np.array(current_positions[1]) - np.array(current_positions[0]))[:2]

        pos_diff_len = np.linalg.norm(global_pos_diff)
        reward_dist = (pos_diff_len - self.last_esc_flag_pos_diff_len) * 100  # when visible, max dist should be 10
        self.last_esc_flag_pos_diff_len = pos_diff_len
        reward_escapee = -reward_dist
        reward_escapee /= float(self.max_steps)
        reward_escapee *= float(flag_visible)

        return reward_escapee

    @staticmethod
    def _compute_chaser_common_reward(chaser_state, escapee_state):
        states_infos = [chaser_state, escapee_state]
        # vel reward
        current_positions = [list(info['base_pos']) for info in states_infos]
        current_vels = [list(info['base_lin_vel']) for info in states_infos]
        global_pos_diff = (np.array(current_positions[1]) - np.array(current_positions[0]))[:2]
        global_pos_diff_direction = global_pos_diff / np.linalg.norm(global_pos_diff)

        chaser_vel = np.array(current_vels[0])
        spd = chaser_vel[0] * global_pos_diff_direction[0] + chaser_vel[1] * global_pos_diff_direction[1]
        reward_vel = np.exp(-np.abs(spd - 5.0) * 0.5)

        # rotation reward
        r_reach = R.from_quat(states_infos[0]["base_orn"])
        mat_reach = r_reach.as_matrix()
        yaw_reach = np.arctan2(mat_reach[1, 0], mat_reach[0, 0])

        reward_rotation = np.exp((np.cos(yaw_reach) * global_pos_diff_direction[0] +
                                  np.sin(yaw_reach) * global_pos_diff_direction[1] - 1.0) * 2.0)
        return reward_vel * 0.0 + reward_rotation * 0.5
