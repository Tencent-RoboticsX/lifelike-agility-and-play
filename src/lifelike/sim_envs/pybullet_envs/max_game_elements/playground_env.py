from collections import OrderedDict
from collections import deque

import gym
import numpy as np
import pybullet
from gym import spaces
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R
from lifelike.sim_envs.pybullet_envs.legged_robot.legged_robot import LeggedRobot
from lifelike.utils.constants import compute_terrain_rectangle
from lifelike.sim_envs.pybullet_envs.randomizer.push_randomizer import PushRandomizer
from lifelike.sim_envs.pybullet_envs.max_game_elements.bullet_static_entities import BulletStatics


def apply_zaxis_rotation(zaxis_angle, quat):
    euler_rot = [zaxis_angle, 0, 0]
    quat_rot = R.from_euler('zyx', euler_rot, degrees=True).as_quat()
    return (R.from_quat(quat) * R.from_quat(quat_rot)).as_quat()


class RayCast(object):
    def __init__(self, num_rays, len_ray, bullet_client, color=None):
        self.num_rays = num_rays
        self.len_ray = len_ray
        self.ray_hit_color = color if color is not None else [1, 0, 0]
        self.ray_mis_color = [1, 1, 1]
        self.ray_ids = []
        self.bullet_client = bullet_client

    def compute_rays(self, x0, y0, z0, yaw):
        ray_from = []
        ray_to = []
        for i in range(self.num_rays):
            ray_from.append([x0, y0, z0])
            ray_to.append([
                x0 + self.len_ray * np.cos(yaw + 2. * np.pi * float(i) / self.num_rays),
                y0 + self.len_ray * np.sin(yaw + 2. * np.pi * float(i) / self.num_rays),
                z0,
            ])
        return ray_from, ray_to

    def draw_rays(self, ray_from, ray_to):
        results = self.bullet_client.rayTestBatch(ray_from, ray_to, collisionFilterMask=6)
        ray_hit = []
        for i in range(self.num_rays):
            hid_obj_id = results[i][0]

            if hid_obj_id < 0:
                hit_pos = [0, 0, 0]
            else:
                hit_pos = results[i][3]
            ray_hit.append(hit_pos)
        return ray_hit


class PlayGroundEnv(gym.Env):
    def __init__(self, enable_render=False,
                 control_freq=50,
                 kp=50.0,
                 kd=1.0,
                 max_tau=16,
                 prop_type=None,
                 stack_frame_num=3,
                 max_steps=1000,
                 obs_randomization=None,
                 env_randomize_config=None,
                 ):
        self.reward_type = None
        # randomization
        self._obs_randomization = obs_randomization
        self._episodic_noise = None

        # bullet env part
        self._enable_render = enable_render
        if self._enable_render:
            self.bullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.bullet_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        self.bullet_client.configureDebugVisualizer(self.bullet_client.COV_ENABLE_RENDERING, 1)
        self._env_randomize_config = env_randomize_config
        self._static_objs = BulletStatics(self.bullet_client, auxiliary_radius=env_randomize_config['auxiliary_radius'])

        # Inner loop PD controller time step
        self.time_step = 1.0 / 500.0
        # Outer loop controller time step
        policy_step = 1.0 / control_freq
        # Number of time of inner loops within one out loop
        self._num_env_steps = int(policy_step / self.time_step)

        self._max_tau = max_tau
        foot_lateral_friction = np.random.uniform(*env_randomize_config['friction_range'])
        self.legged_robot = LeggedRobot(
            bullet_client=self.bullet_client,
            time_step=self.time_step,
            mode="dynamic",
            init_pos=[0.0, 0.0, 0.5],
            init_orn=[0.0, 0.0, 0.0, 1.0],
            kp=kp,
            kd=kd,
            foot_lateral_friction=foot_lateral_friction,
            max_tau=max_tau)
        self._target_pos = self._static_objs.get_target_pos()
        self._rays_cast = RayCast(num_rays=128, len_ray=20.0, bullet_client=self.bullet_client, color='r')
        # agent part
        self._prop_type = prop_type
        self._stack_frame_num = stack_frame_num
        self._max_steps = max_steps
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
        prop_size *= self._stack_frame_num
        prop_a_size = 12 * self._stack_frame_num
        percep_front_size = (25, 13)
        percep_2d_size = (25, 13)
        percep_1d_size = 128
        target_size = 3
        dict_obs_space = OrderedDict({
            'prop': spaces.Box(0, 0, shape=(prop_size,)),
            'prop_a': spaces.Box(0, 0, shape=(prop_a_size,)),
            'percep_2d': spaces.Box(0, 0, shape=percep_2d_size),
            'percep_1d': spaces.Box(0, 0, shape=(percep_1d_size,)),
            'percep_front': spaces.Box(0, 0, shape=percep_front_size),
            'target': spaces.Box(0, 0, shape=(target_size,)),
        })
        self.observation_space = spaces.Dict(dict_obs_space)

        # act
        ac_dim = 12
        self.action_space = spaces.Dict(OrderedDict({
            'A_Z': spaces.Discrete(256),
            'A_LLC': spaces.Box(0, 0, shape=(ac_dim,))
        }))

        self.time = 0
        self.counter = 0
        self.history_props = deque(maxlen=self._stack_frame_num)
        self.history_actions = deque(maxlen=self._stack_frame_num)
        self.last_action = None
        self.last_pos_diff_len = 0
        self.init_pos_diff_len = 0
        # adaptation
        if 'disturb_force_config' in env_randomize_config:
            self.force_randomizer = PushRandomizer(env=self, **env_randomize_config['disturb_force_config'])
        else:
            self.force_randomizer = None
        self._apply_force_curr_ctl_freq = False
        self.wall_width_offset = [0.02, 0.5]
        self.wall_gap_offset = [1.0, 20.0]

        # joystick
        self.cmd_vary_freq_range = env_randomize_config.get('cmd_vary_freq_range', [25, 200])  # [0.5s, 4s] used in jsv4
        self.cmd_vary_freq = None
        self.target_angle = 0
        self.target_spd = 0
        self.max_spd = 0.0
        self.total_spd = 0.0
        self.episodic_reward = OrderedDict({
            'reward_vel': 0.0,
            'reward_rotation': 0.0,
            'reward_dist': 0.0,
            'reward_avg_spd': 0.0,
        })

    def sample_episodic_noise(self):
        if self._obs_randomization:
            self._episodic_noise = {
                k: np.random.uniform(*self._obs_randomization[k]) for k in self._obs_randomization}

    def randomize_init_states(self):
        # randomize pos and orn
        zaxis_angle = 360 * np.random.rand()
        init_states_info = self.legged_robot.get_init_states_info()
        init_states_info['base_orn'] = list(
            apply_zaxis_rotation(zaxis_angle, np.array(init_states_info['base_orn'])))
        init_states_info['base_pos'] = [0.0, 0.0, 0.5]
        self.legged_robot.set_states_info(init_states_info)
        self.last_pos_diff_len = np.linalg.norm(
            (np.array(init_states_info['base_pos']) - np.array(self._target_pos))[:2])
        if self._element_id == 0:  # joystick, where target is randomized within an episode
            self.init_pos_diff_len = None
        else:
            self.init_pos_diff_len = self.last_pos_diff_len

    def reset(self, **kwargs):
        """ reset game_mgr """
        # randomize element
        self._static_objs.element_id = self._env_randomize_config['element_id']
        self._element_id = self._static_objs.element_id
        if self._element_id == 0:
            self.reward_type = 'joystick'
        else:
            self.reward_type = 'average_speed'
        self._static_objs.element_config = {}
        if self._element_id == 2:  # holes
            self._static_objs.element_config = self._env_randomize_config['hole_config']
        # randomize friction
        foot_friction = np.random.uniform(*self._env_randomize_config['friction_range'])
        self.legged_robot.set_foot_lateral_friction(foot_lateral_friction=foot_friction)
        print('Current episodic friction: {}'.format(foot_friction))
        # randomize force
        if self.force_randomizer is not None:
            self.force_randomizer.reset()
        # randomize wall width
        self._static_objs.set_wall_width_offset(self.wall_width_offset)
        # randomize wall gap
        self._static_objs.set_wall_gap_offset(self.wall_gap_offset)
        # reset game_mgr
        self._static_objs.reset()
        self._target_pos = self._static_objs.target_pos
        # joystick
        self.cmd_vary_freq = np.random.randint(*self.cmd_vary_freq_range)

        self.counter = 0
        self.total_spd = 0.0
        self.max_spd = 0.0
        self.episodic_reward = OrderedDict({
            'reward_vel': 0.0,
            'reward_rotation': 0.0,
            'reward_dist': 0.0,
            'reward_avg_spd': 0.0,
        })
        self.last_action = None
        # reset tau_max
        self.legged_robot.max_taus = [np.random.uniform(*self._max_tau) if isinstance(
            self._max_tau, list) else self._max_tau] * self.legged_robot.num_joints
        # reset state
        self.time = 0
        self.sample_episodic_noise()
        # randomize init states
        self.randomize_init_states()

        self.history_props.clear()
        self.history_actions.clear()
        states_info = self.legged_robot.get_states_info()
        drill = self._prepare_drill(states_info)
        obs = self._prepare_obs(states_info, drill, np.zeros(12, ))
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

    def _prepare_obs(self, state, drill, action):
        prop = self._cfg_prop(self._prepare_full_prop(state), self._prop_type)
        while len(self.history_props) < self._stack_frame_num:
            self.history_props.append(prop)
        self.history_props.append(prop)
        props = np.concatenate(self.history_props, axis=-1)

        while len(self.history_actions) < self._stack_frame_num:
            self.history_actions.append(action)
        self.history_actions.append(action)
        acts = np.concatenate(self.history_actions)

        obs = OrderedDict({
            'prop': props,
            'prop_a': acts,
            'percep_2d': drill['self_percep_2d'],
            'percep_1d': drill['self_percep_1d'],
            'percep_front': drill['self_percep_front'],
            'target': drill['target_info'],
        })
        return obs

    def bullet_update(self):
        if self.force_randomizer is not None:
            self.force_randomizer.step()
            self._apply_force_curr_ctl_freq |= self.force_randomizer.apply_force_curr
        self.bullet_client.stepSimulation()

    def step(self, rl_action):
        if self._element_id == 0 and self.counter % self.cmd_vary_freq == 0:  # rl control freq 50HZ, outer 2HZ
            states_info = self.legged_robot.get_states_info()
            robot_pos = states_info['base_pos']
            # compute target_pos given target_angle
            """ target_angle only used for plotting arrow """
            self.target_angle = np.random.uniform(0, 2 * np.pi)
            radius = 100.0
            self._target_pos = [robot_pos[0] + np.cos(self.target_angle) * radius,
                               robot_pos[1] + np.sin(self.target_angle) * radius, 0.0]
            # once update target_pos, need to update last_pos_diff_len
            self.last_pos_diff_len = np.linalg.norm(
                (np.array(states_info['base_pos']) - np.array(self._target_pos))[:2])
        if self.counter % self.cmd_vary_freq == 0:
            self.target_spd = np.random.uniform(*self._env_randomize_config['target_spd_range'])
        if self._element_id != 0:
            # compute global target_angle given target_pos
            states_info = self.legged_robot.get_states_info()
            diff = (np.array(self._target_pos) - np.array(states_info['base_pos']))[:2]
            dir = diff / np.linalg.norm(diff)
            self.target_angle = np.arctan2(dir[1], dir[0])

        rl_action = np.array(rl_action['A_LLC']) if 'A_LLC' in rl_action else np.array(rl_action)
        tgt_joint_pos = self.legged_robot.get_joint_states_info()[0] + rl_action

        # apply inner loop action
        self._apply_force_curr_ctl_freq = False
        for _ in range(self._num_env_steps):
            self.bullet_client.setTimeStep(self.time_step)
            self.legged_robot.apply_action(tgt_joint_pos)
            self.bullet_update()
            self.time += self.time_step
        if self._enable_render:
            self.legged_robot.camera_on()

        # get state
        states_info = self.legged_robot.get_states_info()
        drill = self._prepare_drill(states_info)
        obs = self._prepare_obs(states_info, drill, rl_action)

        # info
        info = {}
        self.counter += 1
        done, done_dict = self._check_terminate(states_info)
        if self.reward_type == 'average_speed':
            reward = self._compute_avg_spd_reward(states_info, done_dict=done_dict)
        elif self.reward_type == 'joystick':
            reward = self._compute_joystick_reward(states_info)
        else:
            raise ValueError('Unknown reward type.')
        if done:
            info['ave_spd'] = self.total_spd / self.counter
            info['max_spd'] = self.max_spd
            info['reward_vel'] = self.episodic_reward['reward_vel']
            info['reward_rotation'] = self.episodic_reward['reward_rotation']
            info['reward_dist'] = self.episodic_reward['reward_dist']
            info['reward_avg_spd'] = self.episodic_reward['reward_avg_spd']
        return obs, reward, done, info

    def _check_terminate(self, states_info):
        done_robot = self.legged_robot.check_terminate(states_info)
        done_time = self.counter >= self._max_steps
        current_position = list(states_info['base_pos'])
        global_pos_diff = np.array(self._target_pos) - np.array(current_position)
        done_reach = np.linalg.norm(global_pos_diff[:2]) < 0.5
        done = done_robot or done_time or done_reach
        done_dict = {
            "done_robot": done_robot,
            "done_time": done_time,
            "done_reach": done_reach,
        }
        return done, done_dict

    def _prepare_drill(self, states_info):
        # self global info
        current_position = list(states_info['base_pos'])
        rotation = R.from_quat(states_info["base_orn"])
        r_b = rotation.as_matrix()
        yaw = np.arctan2(r_b[1, 0], r_b[0, 0])
        if self._episodic_noise:
            if 'pos_x_bias' in self._episodic_noise:
                current_position = [current_position[0] + self._episodic_noise['pos_x_bias'],
                                    current_position[1] + self._episodic_noise['pos_y_bias'],
                                    current_position[2]]
            if 'yaw_bias' in self._episodic_noise:
                yaw += self._episodic_noise['yaw_bias']
        # terrain map
        self_percep_2d = self._get_perception_height(current_position, rotation)
        # ray
        rays_from, rays_to = self._rays_cast.compute_rays(*current_position, yaw)
        rays_hit = self._rays_cast.draw_rays(rays_from, rays_to)
        hit_distance = np.sum((np.array(rays_hit) - np.array(rays_from)) ** 2, axis=-1) ** 0.5
        self_percep_1d = np.array(hit_distance)
        self_percep_2d_front = self._get_perception_front(current_position, rotation)

        # target info
        global_pos_diff = np.array(self._target_pos) - np.array(current_position)
        target_info = np.array(list(rotation.inv().apply(global_pos_diff)))[:2]
        target_info /= np.linalg.norm(target_info)
        target_info = np.concatenate([target_info, np.array([self.target_spd])])
        drill_dict = OrderedDict({
            'self_percep_2d': self_percep_2d,
            'self_percep_1d': self_percep_1d,
            'self_percep_front': self_percep_2d_front,
            'target_info': target_info,
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
        ray_hit_pos = np.array([p1 if h > -1 else p2 for h, p1, p2 in zip(hit_body, ray_hit_position, ray_to)])
        ray_dist = np.linalg.norm(ray_hit_pos - ray_from, axis=1)
        return ray_dist.reshape(25, 13)

    def _get_perception_height(self, base_pos, base_rotation):
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

    def _compute_common_reward(self, states_info):
        # vel reward
        current_position = list(states_info['base_pos'])
        current_vel = list(states_info['base_lin_vel'])
        global_pos_diff = (np.array(self._target_pos) - np.array(current_position))[:2]
        global_pos_diff_direction = global_pos_diff / np.linalg.norm(global_pos_diff)
        spd = np.abs(current_vel[0] * global_pos_diff_direction[0] + current_vel[1] * global_pos_diff_direction[1])
        self.total_spd += spd
        if spd > self.max_spd:
            self.max_spd = spd
        reward_vel = np.exp(-np.abs(spd - 5.0) * 0.5)
        # rotation reward
        r_reach = R.from_quat(states_info["base_orn"])
        mat_reach = r_reach.as_matrix()
        yaw_reach = np.arctan2(mat_reach[1, 0], mat_reach[0, 0])

        reward_rotation = np.exp((np.cos(yaw_reach) * global_pos_diff_direction[0] +
                                  np.sin(yaw_reach) * global_pos_diff_direction[1] - 1.0) * 2.0)
        # dist reward
        pos_diff_len = np.linalg.norm(global_pos_diff)
        reward_dist = (pos_diff_len - self.last_pos_diff_len) * 100  # when visible, max dist should be 10
        self.last_pos_diff_len = pos_diff_len

        reward = reward_vel * 0.0 + reward_rotation
        reward -= reward_dist
        reward /= float(self._max_steps)
        self.episodic_reward['reward_rotation'] += reward_rotation / float(self._max_steps)
        self.episodic_reward['reward_dist'] -= reward_dist / float(self._max_steps)
        return reward

    def _compute_joystick_reward(self, states_info):
        # vel reward
        current_position = list(states_info['base_pos'])
        current_vel = list(states_info['base_lin_vel'])
        global_pos_diff = (np.array(self._target_pos) - np.array(current_position))[:2]
        global_pos_diff_direction = global_pos_diff / np.linalg.norm(global_pos_diff)
        spd = np.abs(current_vel[0] * global_pos_diff_direction[0] + current_vel[1] * global_pos_diff_direction[1])
        self.total_spd += spd
        if spd > self.max_spd:
            self.max_spd = spd
        reward_vel = np.exp(-np.abs(spd - self.target_spd))
        # rotation reward
        r_reach = R.from_quat(states_info["base_orn"])
        mat_reach = r_reach.as_matrix()
        yaw_reach = np.arctan2(mat_reach[1, 0], mat_reach[0, 0])

        reward_rotation = np.exp((np.cos(yaw_reach) * global_pos_diff_direction[0] +
                                  np.sin(yaw_reach) * global_pos_diff_direction[1] - 1.0) * 5.0)

        reward = reward_vel * reward_rotation
        reward /= float(self._max_steps)
        self.episodic_reward['reward_rotation'] += reward_rotation / float(self._max_steps)
        self.episodic_reward['reward_vel'] += reward_vel / float(self._max_steps)
        return reward

    def _compute_avg_spd_reward(self, states_info, done_dict):
        current_position = list(states_info['base_pos'])
        current_vel = list(states_info['base_lin_vel'])
        global_pos_diff = (np.array(self._target_pos) - np.array(current_position))[:2]
        global_pos_diff_direction = global_pos_diff / np.linalg.norm(global_pos_diff)
        spd = np.abs(current_vel[0] * global_pos_diff_direction[0] + current_vel[1] * global_pos_diff_direction[1])
        self.total_spd += spd
        if spd > self.max_spd:
            self.max_spd = spd
        # rotation reward
        r_reach = R.from_quat(states_info["base_orn"])
        mat_reach = r_reach.as_matrix()
        yaw_reach = np.arctan2(mat_reach[1, 0], mat_reach[0, 0])
        reward_rotation = np.exp((np.cos(yaw_reach) * global_pos_diff_direction[0] +
                                  np.sin(yaw_reach) * global_pos_diff_direction[1] - 1.0) * 5.0)
        # dist reward
        pos_diff_len = np.linalg.norm(global_pos_diff)
        if self.init_pos_diff_len is not None:
            reward_dist = (pos_diff_len - self.last_pos_diff_len) / self.init_pos_diff_len
        else:
            reward_dist = 0.0
        self.last_pos_diff_len = pos_diff_len

        scaled_reward_rotation = reward_rotation / float(self._max_steps) * 0.1
        scaled_reward_dist = - reward_dist * 0.1
        reward = scaled_reward_rotation * 2.0 + scaled_reward_dist
        self.episodic_reward['reward_rotation'] += scaled_reward_rotation * 2.0
        self.episodic_reward['reward_dist'] += scaled_reward_dist

        done_cond = done_dict['done_reach']
        if done_cond:
            avg_spd = self.total_spd / self.counter
            reward_avg_spd = np.exp(-np.abs(avg_spd - self.target_spd))
            reward += reward_avg_spd
            self.episodic_reward['reward_avg_spd'] += reward_avg_spd
        return reward
