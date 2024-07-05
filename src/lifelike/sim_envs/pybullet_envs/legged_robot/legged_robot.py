import os
import random
from typing import Optional, Mapping
import numpy as np
from scipy.spatial.transform import Rotation as R
from lifelike.sim_envs.pybullet_envs.legged_robot import get_urdf_path
from lifelike.utils.constants import STATES_INFO_12, STATES_INFO_12_RUN_0
from lifelike.utils.constants import (
    LEG_JOINT_NAMES,
    SHANK_JOINT_NAMES,
    FOOT_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    HANDLE_JOINT_NAMES,
)


def clip_value(value, minimum, maximum):
    return max(minimum, min(value, maximum))


class LeggedRobot(object):
    def __init__(
            self,
            bullet_client,
            time_step,
            mode,
            init_pos,
            init_orn,
            kp=50.0,
            kd=1.0,
            foot_lateral_friction=0.5,
            max_tau=18,
            color=None,
    ):
        """The legged robot class

        :param bullet_client: int, pybullet client
        :param time_step: float, simulation step time
        :param mode: string, 'kinematic' or 'dynamic' mode
        :param init_pos: list of 3, initial base position
        :param init_orn: list of 4, initial base orientation
        """
        self._bullet_client = bullet_client
        self._time_step = time_step
        self._kp = kp
        self._kd = kd
        self._foot_lateral_friction = foot_lateral_friction
        self._max_tau = max_tau
        self._color = color

        robot_path = os.path.join(get_urdf_path(), 'max.urdf')
        if mode == "dynamic":
            self._init_dynamic_model(robot_path, init_pos, init_orn)
        elif mode == "kinematic":
            self._init_kinematic_model(robot_path, init_pos, init_orn)
        else:
            raise ValueError(f'Not support mode `{mode}`.')

        self.end_effector_velocity = np.zeros((4, 3))
        self.end_effector_position = np.zeros((4, 3))

    def set_states_info(self, states_info: Optional[Mapping[str, np.ndarray]] = None):
        """Set the states information for the robot

        :param states_info: dictionary, contains the state of the robot, including base position ('base_pos'),
                            base orientation ('base orn'), base linear velocity ('base_lin_vel'),
                            base angular velocity ('base_ang_vel'), joint positions ('joint_pos') and
                            joint velocities ('joint_vel')
        :return: None
        """
        if states_info is None:
            states_info = STATES_INFO_12

        base_pos, base_orn = states_info["base_pos"], states_info["base_orn"]
        base_lin_vel, base_ang_vel = states_info["base_lin_vel"], states_info["base_ang_vel"]

        self._bullet_client.resetBasePositionAndOrientation(self.robot_id, base_pos, base_orn)
        self._bullet_client.resetBaseVelocity(self.robot_id, base_lin_vel, base_ang_vel)

        joint_positions, joint_velocities = states_info["joint_pos"], states_info["joint_vel"]
        for i in range(self._num_actuated_joints):
            self._bullet_client.resetJointState(
                self.robot_id, self.actuated_joint_indices[i], joint_positions[i], joint_velocities[i]
            )

    def get_states_info(self):
        """Get the states information for the robot

        :return: states_info: dictionary, contains the state of the robot, including base position ('base_pos'),
                              base orientation ('base orn'), base linear velocity ('base_lin_vel'),
                              base angular velocity ('base_ang_vel'), joint positions ('joint_pos') and
                              joint velocities ('joint_vel')
        """
        base_pos, base_orn = self._bullet_client.getBasePositionAndOrientation(self.robot_id)
        base_lin_vel, base_ang_vel = self._bullet_client.getBaseVelocity(self.robot_id)
        joint_pos, joint_vel = self.get_joint_states_info()

        states_info = {
            "base_pos": base_pos,
            "base_orn": base_orn,
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
        }
        return states_info

    def get_joint_states_info(self):
        joint_pos, joint_vel = [], []
        for state in self._bullet_client.getJointStates(self.robot_id, self.actuated_joint_indices):
            joint_pos.append(state[0])
            joint_vel.append(state[1])
        return joint_pos, joint_vel

    @staticmethod
    def get_init_states_info():
        return STATES_INFO_12_RUN_0

    def apply_action(self, tgt_joint_pos, noise=None):
        """Apply torque computed from PD controller

        :param tgt_joint_pos: array, target joint positions
        :param noise: dict, the max noise value of the joint positions and joint velocities
        :return: None
        """
        max_tgt_joint_pos = np.array([3.] * tgt_joint_pos.shape[0])
        tgt_joint_pos = np.clip(tgt_joint_pos, -max_tgt_joint_pos, max_tgt_joint_pos)
        tgt_joint_vel = [0.0] * tgt_joint_pos.shape[0]
        joint_pos, joint_vel = self.get_joint_states_info()
        if noise is not None:
            joint_pos_noise = noise.get('joint_pos', 0.)
            joint_vel_noise = noise.get('joint_vel', 0.)
            for i in range(self._num_actuated_joints):
                joint_pos[i] += joint_pos_noise * random.uniform(-1., 1.)
                joint_vel[i] += joint_vel_noise * random.uniform(-1., 1.)

        taus = []
        for i in range(self._num_actuated_joints):
            tau = self._kp * (tgt_joint_pos[i] - joint_pos[i]) + self._kd * (tgt_joint_vel[i] - joint_vel[i])
            tau = clip_value(tau, -self._max_taus[i], self._max_taus[i])
            taus.append(tau)

        self._bullet_client.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.actuated_joint_indices,
            controlMode=self._bullet_client.TORQUE_CONTROL,
            forces=taus,
        )

    def get_convex_points_position(self):
        convex_indices = self.foot_indices + self.wheel_indices + self.handle_indices
        return [state[0] for state in self._bullet_client.getLinkStates(self.robot_id, convex_indices)]

    def get_head_points_position(self):
        head_indices = [self.handle_indices[0]]
        return [state[0] for state in self._bullet_client.getLinkStates(self.robot_id, head_indices)]

    @staticmethod
    def check_terminate(states_info):
        """Check if the robot falls

        :param states_info: dictionary, contains the state of the robot, including base position ('base_pos'),
                            base orientation ('base orn'), base linear velocity ('base_lin_vel'),
                            base angular velocity ('base_ang_vel'), joint positions ('joint_pos') and
                            joint velocities ('joint_vel')
        :return: bool, True for terminate
        """
        terminates = False
        base_orn = states_info["base_orn"]
        base_rot_mat = R.from_quat(base_orn).as_matrix()
        fwd, up = base_rot_mat[:, 0], base_rot_mat[:, 2]
        left_z = up[0] * fwd[1] - up[1] * fwd[0]
        if left_z > np.sin(45.0 * np.pi / 180.0) or left_z < np.sin(-45.0 * np.pi / 180.0):
            print("Terminates because of roll", np.arcsin(left_z) / np.pi * 180.0)
            terminates = True
        if up[2] < np.cos(60.0 * np.pi / 180.0):
            print("Terminates because of pitch", np.arccos(up[2]) / np.pi * 180.0)
            terminates = True
        return terminates

    def camera_on(self, enable_single_step_rendering=False, yaw_pitch_dist=None):
        """Allow the camera follow the robot

        :return: None
        """
        robot_pos, _ = self._bullet_client.getBasePositionAndOrientation(self.robot_id)
        default_yaw_pitch_dist = self._bullet_client.getDebugVisualizerCamera()[8:11]
        # adjust viewer freely around the robot
        if yaw_pitch_dist is None:
            yaw_pitch_dist = default_yaw_pitch_dist
        self._bullet_client.resetDebugVisualizerCamera(yaw_pitch_dist[2],
                                                      yaw_pitch_dist[0],
                                                      yaw_pitch_dist[1],
                                                      robot_pos)
        if enable_single_step_rendering:
            # smooth simulation rendering
            self._bullet_client.configureDebugVisualizer(self._bullet_client.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

    def compute_end_effector_info(self):
        foot_info = self._bullet_client.getLinkStates(self.robot_id,
                                                     self.foot_indices,
                                                     computeForwardKinematics=True,
                                                     computeLinkVelocity=True)
        self.end_effector_position = np.asarray([x[0] for x in foot_info])
        self.end_effector_velocity = np.asarray([x[6] for x in foot_info])

    def _init_dynamic_model(self, robot_path, init_pos, init_orn):
        self.robot_id: int = self._bullet_client.loadURDF(
            robot_path,
            init_pos,
            init_orn,
            flags=(
                    self._bullet_client.URDF_MAINTAIN_LINK_ORDER
                    + self._bullet_client.URDF_USE_SELF_COLLISION
                    + self._bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                    + self._bullet_client.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
            ),
            globalScaling=1.0,
            useFixedBase=False,
        )
        self.num_joints = self._bullet_client.getNumJoints(self.robot_id)
        joint_infos = [self._bullet_client.getJointInfo(self.robot_id, j) for j in range(self.num_joints)]
        joint_names = [info[1].decode('utf-8') for info in joint_infos]
        self.body_indices = [-1]
        self.leg_indices = [j for j in range(self.num_joints) if joint_names[j] in LEG_JOINT_NAMES]
        self.shank_indices = [j for j in range(self.num_joints) if joint_names[j] in SHANK_JOINT_NAMES]
        self.foot_indices = [j for j in range(self.num_joints) if joint_names[j] in FOOT_JOINT_NAMES]
        self.wheel_indices = [j for j in range(self.num_joints) if joint_names[j] in WHEEL_JOINT_NAMES]
        self.handle_indices = [j for j in range(self.num_joints) if joint_names[j] in HANDLE_JOINT_NAMES]

        self.actuated_joint_indices = [*self.leg_indices]

        for j in range(-1, self.num_joints):
            self._bullet_client.setCollisionFilterGroupMask(
                self.robot_id, j, collisionFilterGroup=9, collisionFilterMask=9
            )
            if self._color == 'r':
                self._bullet_client.changeVisualShape(self.robot_id, j, rgbaColor=[1, 0, 0, 1])
            elif self._color == 'g':
                self._bullet_client.changeVisualShape(self.robot_id, j, rgbaColor=[0, 1, 0, 1])
            elif self._color == 'b':
                self._bullet_client.changeVisualShape(self.robot_id, j, rgbaColor=[0, 0, 1, 1])

        max_tau = self._max_tau if not isinstance(self._max_tau, list) else np.random.uniform(*self._max_tau)
        self._max_taus = np.ones_like(self.actuated_joint_indices) * max_tau

        """ you must run a position control with max_tau = 0 to enable torque control """
        # see more at https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12644
        self._num_actuated_joints = num_actuated_joints = len(self.actuated_joint_indices)
        self._bullet_client.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.actuated_joint_indices,
            controlMode=self._bullet_client.POSITION_CONTROL,
            targetPositions=[0] * num_actuated_joints,
            targetVelocities=[0] * num_actuated_joints,
            forces=[0.0] * num_actuated_joints,
            positionGains=[0.0] * num_actuated_joints,
            velocityGains=[0.0] * num_actuated_joints,
        )
        self._bullet_client.setGravity(0, 0, -9.80665)
        self._bullet_client.setPhysicsEngineParameter(numSolverIterations=10)
        self.set_foot_lateral_friction(self._foot_lateral_friction)
        self._bullet_client.setTimeStep(self._time_step)
        self._bullet_client.setPhysicsEngineParameter(numSubSteps=1)

    def _init_kinematic_model(self, robot_path, init_pos, init_quat):
        self.robot_id = self._bullet_client.loadURDF(
            robot_path,
            init_pos,
            init_quat,
            globalScaling=1.0,
            flags=(self._bullet_client.URDF_MAINTAIN_LINK_ORDER +
                   self._bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )

        self.num_joints = self._bullet_client.getNumJoints(self.robot_id)
        joint_infos = [self._bullet_client.getJointInfo(self.robot_id, j) for j in range(self.num_joints)]
        joint_names = [info[1].decode('utf-8') for info in joint_infos]
        self.leg_indices = [j for j in range(self.num_joints) if joint_names[j] in LEG_JOINT_NAMES]
        self.shank_indices = [j for j in range(self.num_joints) if joint_names[j] in SHANK_JOINT_NAMES]
        self.foot_indices = [j for j in range(self.num_joints) if joint_names[j] in FOOT_JOINT_NAMES]
        self.actuated_joint_indices = [*self.leg_indices]
        self._num_actuated_joints = len(self.actuated_joint_indices)

        alpha = 0.4
        for j in range(-1, self._bullet_client.getNumJoints(self.robot_id)):
            self._bullet_client.setCollisionFilterGroupMask(
                self.robot_id, j, collisionFilterGroup=0, collisionFilterMask=0
            )
            self._bullet_client.changeDynamics(
                self.robot_id,
                j,
                activationState=(
                        self._bullet_client.ACTIVATION_STATE_SLEEP
                        + self._bullet_client.ACTIVATION_STATE_ENABLE_SLEEPING
                        + self._bullet_client.ACTIVATION_STATE_DISABLE_WAKEUP
                ),
                linearDamping=0,
                angularDamping=0,
            )
            self._bullet_client.changeVisualShape(self.robot_id, j, rgbaColor=[1, 1, 1, alpha])

    def set_foot_lateral_friction(self, foot_lateral_friction):
        for i in range(len(self.foot_indices)):
            self._bullet_client.changeDynamics(
                self.robot_id, linkIndex=self.foot_indices[i], lateralFriction=foot_lateral_friction
            )
