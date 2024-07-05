import os

import numpy as np
from scipy.spatial.transform import Rotation as R
from lifelike.sim_envs.pybullet_envs.max_game_elements import get_urdf_path


class BulletStatics:
    def __init__(self, bullet_client, auxiliary_radius=None, element_id=None, element_config=None):
        self.bullet_client = bullet_client
        self.element_id = element_id
        self.objs_id = []
        self.auxiliary_obj_id = []
        self.objs_pos = []
        self.element_config = element_config if element_config else {}
        self.auxiliary_radius = auxiliary_radius
        self._wall_width_offset = [0.5, 0.5]
        self._wall_gap_offset = [1.0, 5.0]
        self.target_pos = [8.0, 0.0, 0.0]
        self._init_model()

    def reset(self):
        for o_id in self.objs_id:
            self.bullet_client.removeBody(o_id)
        for o_id in self.auxiliary_obj_id:
            self.bullet_client.removeBody(o_id)
        self.objs_id.clear()
        self.auxiliary_obj_id.clear()
        self.objs_pos.clear()
        if self.element_id != 0:
            self._generate_random_width_walls()
        self._create_elements(self.element_id)

        for id_ in self.auxiliary_obj_id:
            for j in range(-1, self.bullet_client.getNumJoints(id_)):  # terrain with multiple links
                self.bullet_client.setCollisionFilterGroupMask(id_, j,
                                                               collisionFilterGroup=9,
                                                               collisionFilterMask=9)
                # used for detection, the friction does not matter.
                self.bullet_client.changeDynamics(id_, j,
                                                  lateralFriction=0.9)

    def _create_auxiliary_obj(self, position, length, width, height, flag):
        def _generate_cylinder(p, l, quat):
            visual_shape = self.bullet_client.createVisualShape(
                shapeType=self.bullet_client.GEOM_CYLINDER,
                radius=self.auxiliary_radius,
                length=l,
                visualFrameOrientation=quat,
                rgbaColor=[211 / 255, 1 / 255, 1 / 255, 0.1],
            )
            collision_shape = self.bullet_client.createCollisionShape(
                shapeType=self.bullet_client.GEOM_CYLINDER,
                radius=self.auxiliary_radius,
                height=l,
                collisionFrameOrientation=quat,
            )
            return self.bullet_client.createMultiBody(
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=p,
            )

        def _generate_box(p, l, w, h, yaw):
            quat = R.from_euler('XYZ', [0, 0, yaw]).as_quat()
            visual_shape = self.bullet_client.createVisualShape(
                shapeType=self.bullet_client.GEOM_BOX,
                halfExtents=[
                    l / 2,
                    w / 2,
                    h / 2,
                ],
                visualFrameOrientation=quat,
                rgbaColor=[211 / 255, 1 / 255, 1 / 255, 0.1],
            )
            collision_shape = self.bullet_client.createCollisionShape(
                shapeType=self.bullet_client.GEOM_BOX,
                halfExtents=[
                    l / 2,
                    w / 2,
                    h / 2,
                ],
                collisionFrameOrientation=quat,
            )
            return self.bullet_client.createMultiBody(
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=p,
            )

        ids = []
        quat = R.from_euler('XYZ', [np.pi / 2, 0, 0]).as_quat()
        p1 = position.copy()
        p1[0] -= length / 2
        # ids.append(_generate_box(p1, self.auxiliary_radius * 2, width, height, 0))
        p1[2] += height / 2 * flag
        ids.append(_generate_cylinder(p1, width, quat))
        p2 = position.copy()
        p2[0] += length / 2
        # ids.append(_generate_box(p2, self.auxiliary_radius * 2, width, height, 0))
        p2[2] += height / 2 * flag
        ids.append(_generate_cylinder(p2, width, quat))

        return ids

    def _create_target(self, pos):
        self.target_pos = pos
        visual_shape = self.bullet_client.createVisualShape(
            shapeType=self.bullet_client.GEOM_BOX,
            halfExtents=[
                0.05 / 2,
                0.05 / 2,
                0.05 / 2,
            ],
            rgbaColor=[0, 0, 1, 1]
        )
        collision_shape = self.bullet_client.createCollisionShape(
            shapeType=self.bullet_client.GEOM_BOX,
            halfExtents=[
                0 / 2,
                0 / 2,
                0 / 2,
            ],
        )
        self.target_id = self.bullet_client.createMultiBody(
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.target_pos,
        )
        self.objs_id.append(self.target_id)
        self.objs_pos.append(self.target_pos)

    def _create_plane(self):
        init_quat = [0.0, 0.0, 0.0, 1.0]
        unit_path = os.path.join(get_urdf_path(), "plane.urdf")
        self.bullet_client.loadURDF(
            unit_path,
            [0.0, 0.0, 0.0],
            init_quat,
            flags=(self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES),
            useFixedBase=True,
        )

    def _create_walls(self):
        wall_path = os.path.join(get_urdf_path(), 'walls.urdf')
        init_height = 1.0
        # randomly create walls
        # if np.random.randint(2) == 0:
        wall_id = self.bullet_client.loadURDF(wall_path)
        pos_n = [5, 1.0, init_height]
        orn_n_euler = [0, 0, 0]
        orn_n_quat = R.from_euler('zyx', orn_n_euler, degrees=True).as_quat()
        self.bullet_client.resetBasePositionAndOrientation(wall_id, pos_n, orn_n_quat)
        self.objs_id.append(wall_id)
        self.objs_pos.append(pos_n)
        # if np.random.randint(2) == 0:
        wall_id = self.bullet_client.loadURDF(wall_path)
        pos_s = [5, -1.0, init_height]
        orn_s_euler = [0, 0, 0]
        orn_s_quat = R.from_euler('zyx', orn_s_euler, degrees=True).as_quat()
        self.bullet_client.resetBasePositionAndOrientation(wall_id, pos_s, orn_s_quat)
        self.objs_id.append(wall_id)
        self.objs_pos.append(pos_s)

    def _generate_random_width_walls(self):
        width = np.random.uniform(*self._wall_width_offset)
        length = 200
        height = 2
        gap = np.random.uniform(*self._wall_gap_offset)
        self._cur_wall_gap = gap

        def generateerate_one_wall(position):
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
            # position = [5, 1.0, 1.0]
            id = self.bullet_client.createMultiBody(
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position,
            )
            self.objs_id.append(id)
            self.objs_pos.append(position)

        position = [5, gap / 2.0 + width / 2.0, 1.0]
        generateerate_one_wall(position)
        position = [5, -(gap / 2.0 + width / 2.0), 1.0]
        generateerate_one_wall(position)

    def _create_cubes(self, easy=False, num_set=None):
        cur_len = 0.0
        generate_func = self._generate_one_cube_set
        if num_set is None:
            num_set = np.random.randint(1, 5)
        for _ in range(num_set):
            cur_len = generate_func(cur_len, easy=easy)
        self._create_target([cur_len + np.random.uniform(-3.0, 3.0), 0.0, 0.0])
        # below are more objects to avoid feature fitting to discriminate terminate
        for _ in range(num_set):
            cur_len = generate_func(cur_len, easy=easy)

    def _create_hurdles(self, num_obj=None):
        cur_len = 0.0
        generate_func = self._generate_one_hurdle
        n = num_obj if num_obj is not None else np.random.randint(1, 10)
        for _ in range(n):
            cur_len = generate_func(cur_len)
        self._create_target([cur_len + np.random.uniform(-1.0, 1.0), 0.0, 0.0])
        # below are more objects to avoid feature fitting to discriminate terminate
        for _ in range(n):
            cur_len = generate_func(cur_len)

    def _create_holes(self, num_obj=None):
        cur_len = 0.0
        generate_func = self._generate_one_hole
        n = num_obj if num_obj is not None else np.random.randint(1, 10)
        for _ in range(n):
            cur_len = generate_func(cur_len)
        self._create_target([cur_len + np.random.uniform(-1.0, 1.0), 0.0, 0.0])
        # below are more objects to avoid feature fitting to discriminate terminate
        for _ in range(n):
            cur_len = generate_func(cur_len)

    def _update_target_pos(self, pos):
        self.bullet_client.resetBasePositionAndOrientation(
            self.target_id, pos, [0.0, 0.0, 0.0, 1.0])
        self.target_pos = pos

    def _create_elements(self, e_id=None):
        # elements
        if e_id is None:
            e_id = np.random.randint(5)
        if e_id == 0:  # joystick
            self._create_target([8.0, 0.0, 0.0])
        elif e_id == 1:
            self._create_hurdles()
        elif e_id == 2:
            self._create_holes()
        elif e_id == 3:
            self._create_cubes(easy=True)
        else:
            raise ValueError('Unknown element id.')

    def _create_hurdle_block(self, offset, blk_range=5):
        for i in range(1, blk_range):
            hurdle_config = {
                "width": 0.1 + 2 * i,
                "length": 0.1,
                "max_height": 0.15,
                "min_height": 0.05,
            }
            cur_offset = [offset[0] + i, offset[1]]
            self._generate_one_hurdle(offset=cur_offset, custom_config=hurdle_config)
            cur_offset = [offset[0] - i, offset[1]]
            self._generate_one_hurdle(offset=cur_offset, custom_config=hurdle_config)
        for i in range(1, blk_range):
            hurdle_config = {
                "width": 0.1,
                "length": 0.1 + 2 * i,
                "max_height": 0.15,
                "min_height": 0.05,
            }
            cur_offset = [offset[0], offset[1] + i]
            self._generate_one_hurdle(offset=cur_offset, custom_config=hurdle_config)
            cur_offset = [offset[0], offset[1] - i]
            self._generate_one_hurdle(offset=cur_offset, custom_config=hurdle_config)

    def _create_hole_block(self, offset, blk_range=5):
        for i in range(1, blk_range):
            hole_config = {
                "width": 0.1 + 2 * i,
                "length": 0.1,
                "max_gap_height": 0.3
                if 'max_gap_height' not in self.element_config else self.element_config['max_gap_height'],
                "min_gap_height": 0.25
                if 'min_gap_height' not in self.element_config else self.element_config['min_gap_height'],
            }
            cur_offset = [offset[0] + i, offset[1]]
            self._generate_one_hole(offset=cur_offset, custom_config=hole_config)
            cur_offset = [offset[0] - i, offset[1]]
            self._generate_one_hole(offset=cur_offset, custom_config=hole_config)
        for i in range(1, blk_range):
            hole_config = {
                "width": 0.1,
                "length": 0.1 + 2 * i,
                "max_gap_height": 0.3
                if 'max_gap_height' not in self.element_config else self.element_config['max_gap_height'],
                "min_gap_height": 0.25
                if 'min_gap_height' not in self.element_config else self.element_config['min_gap_height'],
            }
            cur_offset = [offset[0], offset[1] + i]
            self._generate_one_hole(offset=cur_offset, custom_config=hole_config)
            cur_offset = [offset[0], offset[1] - i]
            self._generate_one_hole(offset=cur_offset, custom_config=hole_config)

    def _generate_one_hurdle(self, cur_len=0.0, offset=None, custom_config=None):
        if custom_config is None:
            width = self._cur_wall_gap
            length = self.element_config.get("length", 0.1)
            max_height = self.element_config.get("max_height", 0.15)
            min_height = self.element_config.get("min_height", 0.05)
            max_distance = self.element_config.get("max_distance", 3.0)
            min_distance = self.element_config.get("min_distance", 1.0)
        else:
            width = custom_config.get("width", 1.0)
            length = custom_config.get("length", 0.1)
            max_height = custom_config.get("max_height", 0.15)
            min_height = custom_config.get("min_height", 0.05)
            max_distance = custom_config.get("max_distance", 0.0)
            min_distance = custom_config.get("min_distance", 0.0)

        height = np.random.uniform(min_height, max_height)
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

        distance = np.random.uniform(min_distance, max_distance)

        position = [cur_len + distance / 2, 0, height / 2]
        if offset:
            position = [position[0] + offset[0], position[1] + offset[1], position[-1]]
        cur_len += distance + length

        id = self.bullet_client.createMultiBody(
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
        )
        self.objs_id.append(id)
        self.objs_pos.append(position)

        if self.auxiliary_radius is not None:
            ids = self._create_auxiliary_obj(position, length, width, height, flag=1)
            self.auxiliary_obj_id.extend(ids)

        return cur_len

    def _generate_one_hole(self, cur_len=0.0, offset=None, custom_config=None):
        if custom_config is None:
            width = self._cur_wall_gap
            length = self.element_config.get("length", 0.1)
            height = self.element_config.get("height", 0.3)
            max_gap_height = self.element_config.get("max_gap_height", 0.3)
            min_gap_height = self.element_config.get("min_gap_height", 0.25)
            max_distance = self.element_config.get("max_distance", 3.0)
            min_distance = self.element_config.get("min_distance", 1.0)
        else:
            width = custom_config.get("width", 1.0)
            length = custom_config.get("length", 0.1)
            height = custom_config.get("height", 0.3)
            max_gap_height = custom_config.get("max_gap_height", 0.3)
            min_gap_height = custom_config.get("min_gap_height", 0.25)
            max_distance = custom_config.get("max_distance", 0.0)
            min_distance = custom_config.get("min_distance", 0.0)

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

        distance = np.random.uniform(min_distance, max_distance)
        gap_height = np.random.uniform(min_gap_height, max_gap_height)

        position = [cur_len + distance / 2, 0, height / 2 + gap_height]
        if offset:
            position = [position[0] + offset[0], position[1] + offset[1], position[-1]]
        cur_len += distance + length

        id = self.bullet_client.createMultiBody(
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
        )
        self.objs_id.append(id)
        self.objs_pos.append(position)

        if self.auxiliary_radius is not None:
            ids = self._create_auxiliary_obj(position, length, width, height, flag=-1)
            self.auxiliary_obj_id.extend(ids)

        return cur_len

    def _generate_one_cube_set(self, cur_len=0.0, easy=False):
        def _generate_cube(p, l, w, h):
            visual_shape = self.bullet_client.createVisualShape(
                shapeType=self.bullet_client.GEOM_BOX,
                halfExtents=[
                    l / 2,
                    w / 2,
                    h / 2,
                ],
                rgbaColor=[1.0, 0.65, 1.0, 1.0]
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

            if self.auxiliary_radius is not None:
                ids = self._create_auxiliary_obj(p, l, w, h, flag=1)
                self.auxiliary_obj_id.extend(ids)

        width = self._cur_wall_gap
        max_gap = self.element_config.get("max_gap", 0.75)
        min_gap = self.element_config.get("min_gap", 0.25)
        max_distance = self.element_config.get("max_distance", 1.0)
        min_distance = self.element_config.get("min_distance", 0.0)

        distance = np.random.uniform(min_distance, max_distance)
        cur_len += distance

        if not easy:
            pos = [2.75 + cur_len, 0.0, 0.4 / 2]
            _generate_cube(pos, 1.0, width, 0.4)  # 40cm

        pos = [1.75 + cur_len, 0.0, 0.25 / 2]
        _generate_cube(pos, 0.5, width, 0.25)  # 25cm

        pos = [1.0 + cur_len, 0.0, 0.1 / 2]
        _generate_cube(pos, 0.5, width, 0.1)  # 10cm

        if not easy:
            cur_len += 2.75  # cube center
            cur_len += 0.5  # cube width / 2
            while True:
                if np.random.randint(2) == 0:
                    gap = np.random.uniform(min_gap, max_gap)
                    cur_len += gap
                    pos = [cur_len + 0.5, 0.0, 0.4 / 2]
                    _generate_cube(pos, 1.0, width, 0.4)  # 40cm
                    cur_len += 1.0
                else:
                    break
        else:
            cur_len += 1.75  # 25cm block center
            cur_len += 0.25  # 25cm block width / 2

        pos = [cur_len + 0.5, 0.0, 0.25 / 2]
        _generate_cube(pos, 0.5, width, 0.25)  # 25cm

        if not easy:
            pos = [cur_len + 1.5, 0.0, 0.1 / 2]
            _generate_cube(pos, 0.5, width, 0.1)  # 10cm
        else:
            pos = [cur_len + 1.25, 0.0, 0.1 / 2]
            _generate_cube(pos, 0.5, width, 0.1)  # 10cm

        return cur_len + 3.0

    def _init_model(self, e_id=None):
        # create once
        self._create_plane()

    def randomize_height(self, offset_range_low, offset_range_high):
        for obj, pos in zip(self.objs_id, self.objs_pos):
            _, orn = self.bullet_client.getBasePositionAndOrientation(obj)
            offset = np.random.uniform(offset_range_low, offset_range_high)
            self.bullet_client.resetBasePositionAndOrientation(obj, pos[:2] + [pos[2] + offset], orn)

    def get_target_pos(self):
        return self.target_pos

    def set_target_pos(self, pos):
        self.target_pos = pos

    def set_wall_width_offset(self, offset):
        self._wall_width_offset = offset

    def set_wall_gap_offset(self, offset):
        self._wall_gap_offset = offset
