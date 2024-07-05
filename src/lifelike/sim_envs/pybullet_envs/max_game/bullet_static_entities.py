import os
import numpy as np

from scipy.spatial.transform import Rotation as R
from lifelike.sim_envs.pybullet_envs.max_game import get_urdf_path


class BulletStatics:
    def __init__(self, bullet_client, small=False):
        self._pre_path = ""
        self._small = small
        if small:
            self._pre_path = "small/"
        self.bullet_client = bullet_client
        # self.static_config = static_config
        self._init_model()

    def _create_walls(self):
        init_quat = [0.0, 0.0, 0.0, 1.0]
        unit_path = os.path.join(get_urdf_path(), self._pre_path + "plane.urdf")
        self.bullet_client.loadURDF(
            unit_path,
            [0.0, 0.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )

        wall_path = os.path.join(get_urdf_path(), self._pre_path + 'walls.urdf')
        init_height = 1.0
        self.wall_n_id = self.bullet_client.loadURDF(wall_path)
        pos_n = [0, 5, init_height] if not self._small else [0, 2.5, init_height]
        orn_n_euler = [0, 0, 0]
        orn_n_quat = R.from_euler('zyx', orn_n_euler, degrees=True).as_quat()
        self.bullet_client.resetBasePositionAndOrientation(self.wall_n_id, pos_n, orn_n_quat)

        self.wall_e_id = self.bullet_client.loadURDF(wall_path)
        pos_e = [5, 0, init_height] if not self._small else [3.0, 0, init_height]
        orn_e_euler = [90, 0, 0]
        orn_e_quat = R.from_euler('zyx', orn_e_euler, degrees=True).as_quat()
        self.bullet_client.resetBasePositionAndOrientation(self.wall_e_id, pos_e, orn_e_quat)

        self.wall_s_id = self.bullet_client.loadURDF(wall_path)
        pos_s = [0, -5, init_height] if not self._small else [0, -2.5, init_height]
        orn_s_euler = [0, 0, 0]
        orn_s_quat = R.from_euler('zyx', orn_s_euler, degrees=True).as_quat()
        self.bullet_client.resetBasePositionAndOrientation(self.wall_s_id, pos_s, orn_s_quat)

        self.wall_w_id = self.bullet_client.loadURDF(wall_path)
        pos_w = [-5, 0, init_height] if not self._small else [-3.0, 0, init_height]
        orn_w_euler = [-90, 0, 0]
        orn_w_quat = R.from_euler('zyx', orn_w_euler, degrees=True).as_quat()
        self.bullet_client.resetBasePositionAndOrientation(self.wall_w_id, pos_w, orn_w_quat)

        wall_path = os.path.join(get_urdf_path(), self._pre_path + 'mid_walls.urdf')
        self.wall_m1_id = self.bullet_client.loadURDF(wall_path)
        pos_s = [0, -3, init_height] if not self._small else [0, -1.5, 0]
        orn_s_euler = [0, 0, 0]
        orn_s_quat = R.from_euler('zyx', orn_s_euler, degrees=True).as_quat()
        self.bullet_client.resetBasePositionAndOrientation(self.wall_m1_id, pos_s, orn_s_quat)

        self.wall_m2_id = self.bullet_client.loadURDF(wall_path)
        pos_s = [0, 3, init_height] if not self._small else [0, 1.5, 0]
        orn_s_euler = [0, 0, 0]
        orn_s_quat = R.from_euler('zyx', orn_s_euler, degrees=True).as_quat()
        self.bullet_client.resetBasePositionAndOrientation(self.wall_m2_id, pos_s, orn_s_quat)

    def _create_cubes(self):
        init_quat = [0.0, 0.0, 0.0, 1.0]

        unit_path = os.path.join(get_urdf_path(), self._pre_path + "cube.urdf")
        self.bullet_client.loadURDF(
            unit_path,
            [0.0, 0.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )

        unit_path = os.path.join(get_urdf_path(), self._pre_path + "stamp1.urdf")
        self.bullet_client.loadURDF(
            unit_path,
            [2.0, 0.0, 0.0] if not self._small else [1.0, 0.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )

        unit_path = os.path.join(get_urdf_path(), self._pre_path + "stamp2.urdf")
        self.bullet_client.loadURDF(
            unit_path,
            [3.0, 0.0, 0.0] if not self._small else [1.75, 0.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )

        if not self._small:
            unit_path = os.path.join(get_urdf_path(), self._pre_path + "stamp3.urdf")
            self.bullet_client.loadURDF(
                unit_path,
                [3.75, 0.0, 0.0],
                init_quat,
                flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
                useFixedBase=True,
            )

        unit_path = os.path.join(get_urdf_path(), self._pre_path + "stamp1.urdf")
        self.bullet_client.loadURDF(
            unit_path,
            [-2.0, 0.0, 0.0] if not self._small else [-1.0, 0.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )

        unit_path = os.path.join(get_urdf_path(), self._pre_path + "stamp2.urdf")
        self.bullet_client.loadURDF(
            unit_path,
            [-3.0, 0.0, 0.0] if not self._small else [-1.75, 0.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )

        if not self._small:
            unit_path = os.path.join(get_urdf_path(), self._pre_path + "stamp3.urdf")
            self.bullet_client.loadURDF(
                unit_path,
                [-3.75, 0.0, 0.0],
                init_quat,
                flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
                useFixedBase=True,
            )

    def _create_hurdles(self):
        init_quat = [0.0, 0.0, 0.0, 1.0]

        unit_path = os.path.join(get_urdf_path(), self._pre_path + "hurdle1.urdf")
        self.bullet_client.loadURDF(
            unit_path,
            [-2.0, -4.0, 0.0] if not self._small else [-1.0, -2.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )

        if not self._small:
            unit_path = os.path.join(get_urdf_path(), self._pre_path + "hurdle2.urdf")
            self.bullet_client.loadURDF(
                unit_path,
                [0.0, -4.0, 0.0] if not self._small else [0.0, -2.0, 0.0],
                init_quat,
                flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
                useFixedBase=True,
            )

        unit_path = os.path.join(get_urdf_path(), self._pre_path + "hurdle3.urdf")
        self.bullet_client.loadURDF(
            unit_path,
            [2.0, -4.0, 0.0] if not self._small else [1.0, -2.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )

        if not self._small:
            unit_path = os.path.join(get_urdf_path(), self._pre_path + "hurdle3.urdf")
            self.bullet_client.loadURDF(
                unit_path,
                [-4.0, -3.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
                useFixedBase=True,
            )

            unit_path = os.path.join(get_urdf_path(), self._pre_path + "hurdle3.urdf")
            self.bullet_client.loadURDF(
                unit_path,
                [4.0, -3.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
                useFixedBase=True,
            )

        # another side
        unit_path = os.path.join(get_urdf_path(), self._pre_path + "hurdle1.urdf")
        self.bullet_client.loadURDF(
            unit_path,
            [-2.0, 4.0, 0.0] if not self._small else [-1.0, 2.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )

        if not self._small:
            unit_path = os.path.join(get_urdf_path(), self._pre_path + "hurdle2.urdf")
            self.bullet_client.loadURDF(
                unit_path,
                [0.0, 4.0, 0.0] if not self._small else [0.0, 2.0, 0.0],
                init_quat,
                flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
                useFixedBase=True,
            )

        unit_path = os.path.join(get_urdf_path(), self._pre_path + "hurdle3.urdf")
        self.bullet_client.loadURDF(
            unit_path,
            [2.0, 4.0, 0.0] if not self._small else [1.0, 2.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )

        if not self._small:
            unit_path = os.path.join(get_urdf_path(), self._pre_path + "hurdle3.urdf")
            self.bullet_client.loadURDF(
                unit_path,
                [-4.0, 3.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
                useFixedBase=True,
            )

            unit_path = os.path.join(get_urdf_path(), self._pre_path + "hurdle3.urdf")
            self.bullet_client.loadURDF(
                unit_path,
                [4.0, 3.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
                useFixedBase=True,
            )

    def _create_slope_stair(self):
        init_quat = [0.0, 0.0, 0.0, 1.0]

        slope_stair_path = os.path.join(get_urdf_path(), self._pre_path + "slope_stair.urdf")
        self.robot_id = self.bullet_client.loadURDF(
            slope_stair_path,
            [-3.0, -0.9, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        for i in range(self.bullet_client.getNumJoints(self.robot_id)):
            self.bullet_client.changeVisualShape(
                self.robot_id, i, rgbaColor=[1, 0, 0, 1]
            )
        self.bullet_client.setCollisionFilterGroupMask(
            self.robot_id, -1, collisionFilterGroup=3, collisionFilterMask=3
        )

    def _init_model(self):
        self._create_walls()
        self._create_cubes()
        self._create_hurdles()
        # self._create_slope_stair()


class BulletStaticsV2:
    def __init__(self, bullet_client, holes=False):
        self.bullet_client = bullet_client
        self.holes = holes
        # self.static_config = static_config
        self.objs_id = []
        self.objs_pos = []
        self._init_model()

    def _create_walls(self):
        init_quat = [0.0, 0.0, 0.0, 1.0]
        unit_path = os.path.join(get_urdf_path(), "small_v2/plane.urdf")
        self.bullet_client.loadURDF(
            unit_path,
            [0.0, 0.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )

        wall_path = os.path.join(get_urdf_path(), 'small_v2/walls.urdf')
        init_height = 1.0
        self.wall_n_id = self.bullet_client.loadURDF(wall_path)
        pos_n = [0, 2.5, init_height]
        orn_n_euler = [0, 0, 0]
        orn_n_quat = R.from_euler('zyx', orn_n_euler, degrees=True).as_quat()
        self.bullet_client.resetBasePositionAndOrientation(self.wall_n_id, pos_n, orn_n_quat)

        self.wall_e_id = self.bullet_client.loadURDF(wall_path)
        pos_e = [3.0, 0, init_height]
        orn_e_euler = [90, 0, 0]
        orn_e_quat = R.from_euler('zyx', orn_e_euler, degrees=True).as_quat()
        self.bullet_client.resetBasePositionAndOrientation(self.wall_e_id, pos_e, orn_e_quat)

        self.wall_s_id = self.bullet_client.loadURDF(wall_path)
        pos_s = [0, -2.5, init_height]
        orn_s_euler = [0, 0, 0]
        orn_s_quat = R.from_euler('zyx', orn_s_euler, degrees=True).as_quat()
        self.bullet_client.resetBasePositionAndOrientation(self.wall_s_id, pos_s, orn_s_quat)

        self.wall_w_id = self.bullet_client.loadURDF(wall_path)
        pos_w = [-3.0, 0, init_height]
        orn_w_euler = [-90, 0, 0]
        orn_w_quat = R.from_euler('zyx', orn_w_euler, degrees=True).as_quat()
        self.bullet_client.resetBasePositionAndOrientation(self.wall_w_id, pos_w, orn_w_quat)

        # wall_path = os.path.join(get_urdf_path(), 'small_v2/mid_walls.urdf')
        # self.wall_m2_id = self.bullet_client.loadURDF(wall_path)
        # pos_s = [0, 1.5, 0]
        # orn_s_euler = [0, 0, 0]
        # orn_s_quat = R.from_euler('zyx', orn_s_euler, degrees=True).as_quat()
        # self.bullet_client.resetBasePositionAndOrientation(self.wall_m2_id, pos_s, orn_s_quat)
        #
        # wall_path = os.path.join(get_urdf_path(), 'small_v2/mid_walls2.urdf')
        # self.wall_m1_id = self.bullet_client.loadURDF(wall_path)
        # pos_s = [-2.0, 0.0, 0]
        # orn_s_euler = [0, 0, 0]
        # orn_s_quat = R.from_euler('zyx', orn_s_euler, degrees=True).as_quat()
        # self.bullet_client.resetBasePositionAndOrientation(self.wall_m1_id, pos_s, orn_s_quat)
        #
        # wall_path = os.path.join(get_urdf_path(), 'small_v2/mid_walls2.urdf')
        # self.wall_m3_id = self.bullet_client.loadURDF(wall_path)
        # pos_s = [2.0, 0.0, 0]
        # orn_s_euler = [0, 0, 0]
        # orn_s_quat = R.from_euler('zyx', orn_s_euler, degrees=True).as_quat()
        # self.bullet_client.resetBasePositionAndOrientation(self.wall_m3_id, pos_s, orn_s_quat)

        wall_path = os.path.join(get_urdf_path(), 'small_v2/mid_walls3.urdf')
        self.wall_m4_id = self.bullet_client.loadURDF(wall_path)
        pos_s = [0.0, 0.0, 0]
        orn_s_euler = [0, 0, 0]
        orn_s_quat = R.from_euler('zyx', orn_s_euler, degrees=True).as_quat()
        self.bullet_client.resetBasePositionAndOrientation(self.wall_m4_id, pos_s, orn_s_quat)

    def _create_cubes(self):
        init_quat = [0.0, 0.0, 0.0, 1.0]

        unit_path = os.path.join(get_urdf_path(), "small_v2/cube.urdf")
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [0.0, 2.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([0.0, 2.0, 0.0])

        unit_path = os.path.join(get_urdf_path(), "small_v2/stamp1.urdf")
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [1.0, 2.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([1.0, 2.0, 0.0])

        unit_path = os.path.join(get_urdf_path(), "small_v2/stamp2.urdf")
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [1.75, 2.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([1.75, 2.0, 0.0])

        unit_path = os.path.join(get_urdf_path(), "small_v2/stamp1.urdf")
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [-1.0, 2.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([-1.0, 2.0, 0.0])

        unit_path = os.path.join(get_urdf_path(), "small_v2/stamp2.urdf")
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [-1.75, 2.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([-1.75, 2.0, 0.0])

    def _create_hurdles(self):
        init_quat = [0.0, 0.0, 0.0, 1.0]

        unit_path = os.path.join(get_urdf_path(), "small_v2/hurdle1.urdf")
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [-2.5, 1.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([-2.5, 1.0, 0.0])

        unit_path = os.path.join(get_urdf_path(), "small_v2/hurdle3.urdf")
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [-2.5, -1.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([-2.5, -1.0, 0.0])

    def _create_holes(self):
        init_quat = [0.0, 0.0, 0.0, 1.0]

        unit_path = os.path.join(get_urdf_path(), "small_v2/hurdle2.urdf")
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [-1.0, -2.0, 0.4],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([-1.0, -2.0, 0.4])

        id_ = self.bullet_client.loadURDF(
            unit_path,
            [1.0, -2.0, 0.4],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([1.0, -2.0, 0.4])

        # another line
        init_quat = [0.0, 0.0, 1.0, 1.0]

        unit_path = os.path.join(get_urdf_path(), "small_v2/hurdle2.urdf")
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [2.5, 1.0, 0.5],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([2.5, 1.0, 0.5])

        id_ = self.bullet_client.loadURDF(
            unit_path,
            [2.5, -1.0, 0.5],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([2.5, -1.0, 0.5])

    def _create_slope_stair(self):
        init_quat = [0.0, 0.0, 0.0, 1.0]

        slope_stair_path = os.path.join(get_urdf_path(), "small_v2/slope_stair.urdf")
        self.robot_id = self.bullet_client.loadURDF(
            slope_stair_path,
            [-3.0, -0.9, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        for i in range(self.bullet_client.getNumJoints(self.robot_id)):
            self.bullet_client.changeVisualShape(
                self.robot_id, i, rgbaColor=[1, 0, 0, 1]
            )
        self.bullet_client.setCollisionFilterGroupMask(
            self.robot_id, -1, collisionFilterGroup=3, collisionFilterMask=3
        )

    def _init_model(self):
        self._create_walls()
        self._create_cubes()
        self._create_hurdles()
        if self.holes:
            self._create_holes()
        # self._create_slope_stair()

    def randomize_height(self, offset_range_low, offset_range_high):
        for obj, pos in zip(self.objs_id, self.objs_pos):
            _, orn = self.bullet_client.getBasePositionAndOrientation(obj)
            offset = np.random.uniform(offset_range_low, offset_range_high)
            self.bullet_client.resetBasePositionAndOrientation(obj, pos[:2] + [pos[2] + offset], orn)


class BulletStaticsV3:
    def __init__(self, bullet_client):
        self.bullet_client = bullet_client
        self.objs_id = []
        self.objs_pos = []
        self._init_model()

    def _create_walls(self):
        init_quat = [0.0, 0.0, 0.0, 1.0]
        unit_path = os.path.join(get_urdf_path(), "small_v3/plane.urdf")
        self.bullet_client.loadURDF(
            unit_path,
            [0.0, 0.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )

        wall_path = os.path.join(get_urdf_path(), 'small_v3/walls1.urdf')
        init_height = 1.0
        self.wall_n_id = self.bullet_client.loadURDF(wall_path)
        pos_n = [0, 3, init_height]
        orn_n_euler = [0, 0, 0]
        orn_n_quat = R.from_euler('zyx', orn_n_euler, degrees=True).as_quat()
        self.bullet_client.resetBasePositionAndOrientation(self.wall_n_id, pos_n, orn_n_quat)

        self.wall_e_id = self.bullet_client.loadURDF(wall_path)
        pos_e = [3.5, 0, init_height]
        orn_e_euler = [90, 0, 0]
        orn_e_quat = R.from_euler('zyx', orn_e_euler, degrees=True).as_quat()
        self.bullet_client.resetBasePositionAndOrientation(self.wall_e_id, pos_e, orn_e_quat)

        self.wall_s_id = self.bullet_client.loadURDF(wall_path)
        pos_s = [0, -3, init_height]
        orn_s_euler = [0, 0, 0]
        orn_s_quat = R.from_euler('zyx', orn_s_euler, degrees=True).as_quat()
        self.bullet_client.resetBasePositionAndOrientation(self.wall_s_id, pos_s, orn_s_quat)

        self.wall_w_id = self.bullet_client.loadURDF(wall_path)
        pos_w = [-3.5, 0, init_height]
        orn_w_euler = [-90, 0, 0]
        orn_w_quat = R.from_euler('zyx', orn_w_euler, degrees=True).as_quat()
        self.bullet_client.resetBasePositionAndOrientation(self.wall_w_id, pos_w, orn_w_quat)

    def _create_cubes(self):
        init_quat = [0.0, 0.0, 1.0, 1.0]

        unit_path = os.path.join(get_urdf_path(), "small_v3/cube.urdf")
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [2.5, 2.0, -0.1],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([2.5, 2.0, -0.1])

        unit_path = os.path.join(get_urdf_path(), "small_v3/cube.urdf")  # 40cm
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [-2.5, 2.0, -0.1],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([-2.5, 2.0, -0.1])

        unit_path = os.path.join(get_urdf_path(), "small_v3/cube.urdf")  # 40cm
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [-1, 1.5, -0.1],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([-1, 1.5, -0.1])

        unit_path = os.path.join(get_urdf_path(), "small_v3/cube.urdf")  # 40cm
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [1, 1.5, -0.1],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([1, 1.5, -0.1])

        unit_path = os.path.join(get_urdf_path(), "small_v3/cube.urdf")  # 40cm
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [1, 0.0, -0.1],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([1, 0.0, -0.1])

        unit_path = os.path.join(get_urdf_path(), "small_v3/cube.urdf")  # 40cm
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [-1, 0.0, -0.1],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([-1, 0.0, -0.1])

        unit_path = os.path.join(get_urdf_path(), "small_v3/cube.urdf")  # 40cm
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [-1, -1.5, -0.1],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([-1, -1.5, -0.1])

        unit_path = os.path.join(get_urdf_path(), "small_v3/cube.urdf")  # 40cm
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [1, -1.5, -0.1],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([1, -1.5, -0.1])

        unit_path = os.path.join(get_urdf_path(), "small_v3/stamp2.urdf")  # 25cm
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [-2.5, 1.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([-2.5, 1.0, 0.0])

        unit_path = os.path.join(get_urdf_path(), "small_v3/stamp3.urdf")  # 10cm
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [-2.5, 0.25, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([-2.5, 0.25, 0.0])

        unit_path = os.path.join(get_urdf_path(), "small_v3/stamp2.urdf")  # 25cm
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [2.5, 1.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([2.5, 1.0, 0.0])

        unit_path = os.path.join(get_urdf_path(), "small_v3/stamp3.urdf")  # 10cm
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [2.5, 0.25, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([2.5, 0.25, 0.0])

        unit_path = os.path.join(get_urdf_path(), "small_v3/walls2.urdf")
        self.wall_w_id = self.bullet_client.loadURDF(unit_path)
        pos_w = [-2.0, 1.25, 0.0]
        orn_w_euler = [-90, 0, 0]
        orn_w_quat = R.from_euler('zyx', orn_w_euler, degrees=True).as_quat()
        self.bullet_client.resetBasePositionAndOrientation(self.wall_w_id, pos_w, orn_w_quat)

        self.wall_w_id = self.bullet_client.loadURDF(unit_path)
        pos_w = [2.0, 1.25, 0.0]
        orn_w_euler = [-90, 0, 0]
        orn_w_quat = R.from_euler('zyx', orn_w_euler, degrees=True).as_quat()
        self.bullet_client.resetBasePositionAndOrientation(self.wall_w_id, pos_w, orn_w_quat)

    def _create_hurdles(self):
        h_init_quat = [0.0, 0.0, 0.0, 1.0]
        v_init_quat = [0.0, 0.0, 1.0, 1.0]

        unit_path = os.path.join(get_urdf_path(), "small_v3/hurdle3.urdf")  # 15cm
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [0, -1.5, 0.0],
            h_init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([0, -1.5, 0.0])

        unit_path = os.path.join(get_urdf_path(), "small_v3/hurdle3.urdf")  # 15cm
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [0, 0, 0],
            h_init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([0, 0, 0])

        unit_path = os.path.join(get_urdf_path(), "small_v3/hurdle3.urdf")  # 15cm
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [0, 1.5, 0],
            h_init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([0, 1.5, 0])

        unit_path = os.path.join(get_urdf_path(), "small_v3/hurdle3.urdf")  # 15cm
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [-1, -1, 0],
            v_init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([-1, -1, 0])

        unit_path = os.path.join(get_urdf_path(), "small_v3/hurdle3.urdf")  # 15cm
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [1, 1, 0],
            v_init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([1, 1, 0])

    def _create_holes(self):
        init_quat = [0.0, 0.0, 0.0, 1.0]

        unit_path = os.path.join(get_urdf_path(), "small_v3/hurdle2.urdf")
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [-1.0, -2.0, 0.4],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([-1.0, -2.0, 0.4])

        id_ = self.bullet_client.loadURDF(
            unit_path,
            [1.0, -2.0, 0.4],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([1.0, -2.0, 0.4])

        # another line
        init_quat = [0.0, 0.0, 1.0, 1.0]

        unit_path = os.path.join(get_urdf_path(), "small_v3/hurdle2.urdf")
        id_ = self.bullet_client.loadURDF(
            unit_path,
            [2.5, 1.0, 0.5],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([2.5, 1.0, 0.5])

        id_ = self.bullet_client.loadURDF(
            unit_path,
            [2.5, -1.0, 0.5],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        self.objs_id.append(id_)
        self.objs_pos.append([2.5, -1.0, 0.5])

    def _create_slope_stair(self):
        init_quat = [0.0, 0.0, 0.0, 1.0]

        slope_stair_path = os.path.join(get_urdf_path(), "small_v3/slope_stair.urdf")
        self.robot_id = self.bullet_client.loadURDF(
            slope_stair_path,
            [-3.0, -0.9, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )
        for i in range(self.bullet_client.getNumJoints(self.robot_id)):
            self.bullet_client.changeVisualShape(
                self.robot_id, i, rgbaColor=[1, 0, 0, 1]
            )
        self.bullet_client.setCollisionFilterGroupMask(
            self.robot_id, -1, collisionFilterGroup=3, collisionFilterMask=3
        )

    def _init_model(self):
        self._create_walls()
        self._create_cubes()
        self._create_hurdles()
        # self._create_slope_stair()

    def randomize_height(self, offset_range_low, offset_range_high):
        for obj, pos in zip(self.objs_id, self.objs_pos):
            _, orn = self.bullet_client.getBasePositionAndOrientation(obj)
            offset = np.random.uniform(offset_range_low, offset_range_high)
            self.bullet_client.resetBasePositionAndOrientation(obj, pos[:2] + [pos[2] + offset], orn)


class BulletStaticsV4:
    def __init__(self, bullet_client, element_config: dict):
        self.bullet_client = bullet_client
        self.objs_id = []
        self.objs_pos = []
        self.element_config = element_config
        self._init_model()

    def reset(self):
        for o_id in self.objs_id:
            self.bullet_client.removeBody(o_id)
        self.objs_id.clear()
        self.objs_pos.clear()
        # recreate
        self._create_walls()
        if 'rand_cube' in self.element_config and self.element_config['rand_cube']:
            self._create_rand_cube_block()
        if 'hurdle' in self.element_config and self.element_config['hurdle']:
            self._create_hurdles()
        if 'hole' in self.element_config and self.element_config['hole']:
            self._create_holes()

    def _create_plane(self):
        init_quat = [0.0, 0.0, 0.0, 1.0]
        unit_path = os.path.join(get_urdf_path(), "small_v3/plane.urdf")
        self.bullet_client.loadURDF(
            unit_path,
            [0.0, 0.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )

    def _create_walls(self):
        width = 0.01
        length = 5.0
        height = 2.0

        def generate_one_wall(p, l, w, h):
            visual_shape = self.bullet_client.createVisualShape(
                shapeType=self.bullet_client.GEOM_BOX,
                halfExtents=[
                    l / 2,
                    w / 2,
                    h / 2,
                ],
                rgbaColor=[0, 1, 1, 1]
            )
            collision_shape = self.bullet_client.createCollisionShape(
                shapeType=self.bullet_client.GEOM_BOX,
                halfExtents=[
                    l / 2,
                    w / 2,
                    h / 2,
                ],
            )
            # position = [5, 1.0, 1.0]
            id = self.bullet_client.createMultiBody(
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=p,
            )
            self.objs_id.append(id)
            self.objs_pos.append(p)

        position = [0, 2.5, height / 2]
        generate_one_wall(position, length, width, height)
        position = [0, -2.5, height / 2]
        generate_one_wall(position, length, width, height)
        position = [2.5, 0, height / 2]
        generate_one_wall(position, width, length, height)
        position = [-2.5, 0, height / 2]
        generate_one_wall(position, width, length, height)

    def _create_rand_cube_block(self):
        """ random cube height and cube num """
        def _gen_cube(p, l, w, h):
            visual_shape = self.bullet_client.createVisualShape(
                shapeType=self.bullet_client.GEOM_BOX,
                halfExtents=[
                    l / 2,
                    w / 2,
                    h / 2,
                ],
                rgbaColor=[1.0, 0.65, 0.0, 1.0]
            )
            collision_shape = self.bullet_client.createCollisionShape(
                shapeType=self.bullet_client.GEOM_BOX,
                halfExtents=[
                    l / 2,
                    w / 2,
                    h / 2,
                ],
            )
            id = self.bullet_client.createMultiBody(
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=p,
            )
            self.objs_id.append(id)
            self.objs_pos.append(p)

        max_n_cube = 6
        min_n_cube = 5
        n_cube = np.random.randint(min_n_cube, max_n_cube)
        for _ in range(n_cube):
            height = np.random.uniform(0.05, 0.25)
            position = [np.random.uniform(-2.0, 2.0),
                        np.random.uniform(-2.0, 2.0),
                        height / 2]
            length = np.random.uniform(0.5, 1.0)
            width = np.random.uniform(0.5, 1.0)
            _gen_cube(position, length, width, height)

    def _create_hurdles(self):
        """ random hurdle height """
        width = 5.0
        length = 0.1
        max_height = 0.15
        min_height = 0.05

        height = np.random.uniform(min_height, max_height)
        visual_shape = self.bullet_client.createVisualShape(
            shapeType=self.bullet_client.GEOM_BOX,
            halfExtents=[
                length / 2,
                width / 2,
                height / 2,
            ],
            rgbaColor=[0, 1, 1, 1]
        )
        collision_shape = self.bullet_client.createCollisionShape(
            shapeType=self.bullet_client.GEOM_BOX,
            halfExtents=[
                length / 2,
                width / 2,
                height / 2,
            ],
        )

        position = [0, 0, height / 2]
        id = self.bullet_client.createMultiBody(
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
        )
        self.objs_id.append(id)
        self.objs_pos.append(position)

    def _create_holes(self):
        """ random gap height """
        width = 0.1
        length = 5.0
        height = 0.3
        max_gap_height = 0.3
        min_gap_height = 0.25

        visual_shape = self.bullet_client.createVisualShape(
            shapeType=self.bullet_client.GEOM_BOX,
            halfExtents=[
                length / 2,
                width / 2,
                height / 2,
            ],
            rgbaColor=[0, 1, 1, 1]
        )
        collision_shape = self.bullet_client.createCollisionShape(
            shapeType=self.bullet_client.GEOM_BOX,
            halfExtents=[
                length / 2,
                width / 2,
                height / 2,
            ],
        )
        gap_height = np.random.uniform(min_gap_height, max_gap_height)
        position = [0, 0, height / 2 + gap_height]
        id = self.bullet_client.createMultiBody(
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
        )
        self.objs_id.append(id)
        self.objs_pos.append(position)

    def _init_model(self):
        self._create_plane()
        self._create_walls()

    def randomize_height(self, offset_range_low, offset_range_high):
        pass
