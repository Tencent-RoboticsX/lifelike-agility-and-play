import os
import time
import json
import numpy as np
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from tleague.utils import logger
from lifelike.utils.obstacle import obstacles_in_frame


class MotionLib(object):
    def __init__(self, data_path, policy_step):
        self._data_path = data_path
        self._policy_step = policy_step
        self._open_all_mocap_datas()
        self.frame_id = 0
        self.frame_fraction = 0.

    def _open_all_mocap_datas(self):
        while True:
            if os.path.exists(self._data_path):
                if os.path.isdir(self._data_path):
                    mocap_files = [f for f in os.listdir(self._data_path) if f.endswith("txt")]
                    data_file = [os.path.join(self._data_path, file) for file in sorted(mocap_files)]
                else:
                    data_file = [self._data_path]
                break
            else:
                logger.log('File or dir not exists. Wait 10s and recheck. Check if the data_path is wrong.')
                time.sleep(10)
        datas = [json.load(open(file, 'r')) for file in data_file]
        self.data_len = [len(data['Frames']) for data in datas]
        self.frame_step = datas[0]['FrameDuration']
        self.frame_rate = int(1. / self.frame_step)
        self.margin = int(np.ceil(self._policy_step / self.frame_step)) + self.frame_rate + 2

        self.data = []
        self.obstacles_info = []
        for data in datas:
            frames = data['Frames']
            self.data.append(frames)
            self.obstacles_info.append(obstacles_in_frame(frames, self.frame_rate))

        self.time_future = [1. / 30., 1. / 15., 1. / 3., 1.]
        self.max_steps = [(len(data) - self.margin) * self.frame_step / self._policy_step for data in self.data]
        self.prioritized_sample_probability = np.ones(len(self.data)) / len(self.data)

    def reset(self):
        self._sample_mocap_data()
        motion_duration = self.frame_step * (self.num_frames - self.margin - 1)
        sampled_time = np.random.uniform(0, 1) * motion_duration
        self.frame_id = int(np.floor(sampled_time / self.frame_step))
        self.frame_fraction = (sampled_time - self.frame_id * self.frame_step) / self.frame_step
        frame_c = self.frames[self.frame_id]
        frame_n = self.frames[self.frame_id + 1]
        states_info = self._get_states_info_by_interpolation(frame_c, frame_n, self.frame_fraction)
        return sampled_time, states_info

    def _sample_mocap_data(self):
        self.sampled_data_idx = np.random.choice(range(len(self.data)), p=self.prioritized_sample_probability)
        self.frames = self.data[self.sampled_data_idx]
        self.num_frames = len(self.frames)
        self.obstacle = self.obstacles_info[self.sampled_data_idx]

    def step(self, t):
        self.frame_id = int(np.floor(t / self.frame_step))
        self.frame_fraction = (t - self.frame_id * self.frame_step) / self.frame_step

    def get_states_info(self, free_joint=True):
        frame_c = self.frames[self.frame_id]
        frame_n = self.frames[self.frame_id + 1]
        states_info = self._get_states_info_by_interpolation(frame_c, frame_n, self.frame_fraction, free_joint)
        return states_info

    def get_states_info_future(self):
        future_states_info = []
        frame_data_future = self.frames[self.frame_id:self.frame_id + self.frame_rate + 2]
        for i in range(len(self.time_future)):
            # interpolate for goal from expert data by time moment
            t = self.frame_step * self.frame_fraction + self.time_future[i]
            frame_id = int(np.floor(t / self.frame_step))
            frame_frac = t / self.frame_step - frame_id
            states_info = self._get_states_info_by_interpolation(
                frame_data_future[frame_id], frame_data_future[frame_id + 1], frame_frac, free_joint=True)
            future_states_info.append(states_info)
        return future_states_info

    def _get_states_info_by_interpolation(self, frame_data_c, frame_data_n, frame_frac, free_joint=True,
                                         interpolation=True):
        if interpolation:
            base_pos = self.base_pos_interpolation(frame_data_c[0:3], frame_data_n[0:3], frame_frac)
            base_orn = self.base_orn_interpolation(frame_data_c[3:7], frame_data_n[3:7], frame_frac)
            base_lin_vel = self.base_lin_vel_interpolation(frame_data_c[0:3], frame_data_n[0:3], self.frame_step)
            base_ang_vel = self.base_ang_vel_interpolation(frame_data_c[3:7], frame_data_n[3:7], self.frame_step)
            joint_pos, joint_vel = self.joint_interpolation(frame_data_c[7:], frame_data_n[7:], frame_frac,
                                                            self.frame_step, free_joint)
        else:
            # TODO: base_ang_vel and joint vel are set to 0 for sanity check, NEEDS TO BE MODIFIED
            base_pos = frame_data_c[0:3]
            base_orn = frame_data_c[3:7]
            base_lin_vel = self.base_lin_vel_interpolation(frame_data_c[0:3], frame_data_n[0:3], self.frame_step)
            base_ang_vel = [0, 0,
                            0]  # self.base_ang_vel_interpolation(frame_data_c[3:7], frame_data_n[3:7], self.frame_step)
            # joint_pos, joint_vel = self.joint_interpolation(frame_data_c[7:], frame_data_n[7:], frame_frac,
            #                                                 self.frame_step, free_joint)
            joint_pos = frame_data_c[7:]
            joint_vel = [0] * 12

        states_info = {'base_pos': base_pos,
                       'base_orn': base_orn,
                       'base_lin_vel': base_lin_vel,
                       'base_ang_vel': base_ang_vel,
                       'joint_pos': joint_pos,
                       'joint_vel': joint_vel}
        return states_info

    @staticmethod
    def base_pos_interpolation(base_pos_c, base_pos_n, frac):
        base_pos = [
            base_pos_c[0] + frac * (base_pos_n[0] - base_pos_c[0]),
            base_pos_c[1] + frac * (base_pos_n[1] - base_pos_c[1]),
            base_pos_c[2] + frac * (base_pos_n[2] - base_pos_c[2]),
        ]
        return base_pos

    @staticmethod
    def base_orn_interpolation(base_orn_c, base_orn_n, frac):
        key_rots = R.from_quat([base_orn_c, base_orn_n])
        key_times = [0, 1]
        times = frac
        slerp = Slerp(key_times, key_rots)
        interp_rots = slerp(times)
        base_orn = interp_rots.as_quat()
        return list(base_orn)

    @staticmethod
    def base_lin_vel_interpolation(base_pos_c, base_pos_n, delta_t):
        base_lin_vel = [(base_pos_n[0] - base_pos_c[0]) / delta_t, (base_pos_n[1] - base_pos_c[1]) / delta_t,
                        (base_pos_n[2] - base_pos_c[2]) / delta_t]
        return base_lin_vel

    @staticmethod
    def base_ang_vel_interpolation(base_orn_c, base_orn_n, delta_t):
        rotvec = (R.from_quat(base_orn_n) * R.from_quat(base_orn_c).inv()).as_rotvec()
        angle = np.sqrt(np.sum(rotvec ** 2))
        axis = rotvec / (angle + 1e-8)
        #
        base_ang_vel = [(axis[0] * angle) / delta_t, (axis[1] * angle) / delta_t, (axis[2] * angle) / delta_t]
        return base_ang_vel

    @staticmethod
    def joint_interpolation(joint_pos_c, joint_pos_n, frac, delta_t, free_joint=True):
        joint_pos, joint_vel = [], []
        for i in range(len(joint_pos_c)):
            joint_pos_c_i = joint_pos_c[i]
            joint_pos_n_i = joint_pos_n[i]
            joint_pos_i = joint_pos_c_i + frac * (joint_pos_n_i - joint_pos_c_i)
            joint_vel_i = (joint_pos_n_i - joint_pos_c_i) / delta_t
            joint_pos.append(joint_pos_i)
            joint_vel.append(joint_vel_i)
        if not free_joint:
            def _add_wheel(data):
                return data[0:3] + [0.] + data[3:6] + [0.] + data[6:9] + [0.] + data[9:12] + [0.]
            joint_pos = _add_wheel(joint_pos)
            joint_vel = _add_wheel(joint_vel)
        return joint_pos, joint_vel

    def is_ended(self):
        terminate = False
        if self.frame_id >= self.num_frames - self.margin - 1:
            terminate = True
        return terminate
