from math import pi, sin, cos
import numpy as np

COLOR = np.array([0.80392157, 0.0627451, 0.4627451])  # pink


class PushRandomizer:
    """Add random forces to the base of the robot during the simulation steps."""

    def __init__(
            self,
            env,
            start_time=0.,
            interval_time=5.,
            duration_time=0.5,
            horizontal_force=20,
            vertical_force=5,
            force_position=np.zeros(3),
            # force_position=np.array([0.08, 0.25, 0]),
            push_strength_ratio=1.,
            render=False,
    ):
        """Initializes the randomizer.

        Args:
          start_time: No push force before the env has advanced this amount of steps.
          interval_time: The step interval between applying push forces.
          duration_time: The duration of the push force.
          horizontal_force: The applied force magnitude when projected in the horizontal plane.
          vertical_force: The z component of the applied push force (positive:↓).
        """
        assert duration_time <= interval_time
        self.env = env
        self._start_time = start_time
        self._interval_time = interval_time
        self._duration_time = duration_time
        self._horizontal_force_range = horizontal_force
        self._vertical_force_range = vertical_force
        self._horizontal_force = 0
        self._vertical_force = 0
        self._force_position = force_position
        self.push_strength_ratio = push_strength_ratio
        self._is_render = render
        self._link_id = 0
        self._duration_step = self._duration_time // env.time_step
        self._interval_step = self._interval_time // env.time_step
        self.force = np.zeros(3)
        self._randomized_force = None
        self._count = None
        self.apply_force_curr = False

    def reset(self):
        self._count = -self._start_time // self.env.time_step
        self.randomize_force()

    def step(self):
        """Randomize simulation steps.

        Will be called at every time step. May add random forces/torques to the base of the robot.

        """
        self._count += 1
        self.force = np.zeros(3)
        self.apply_force_curr = False
        if self._count > 0:
            if self._count % self._interval_step == 0:
                self.randomize_force()
                self._count = 0
            if self._count < self._duration_step:
                self.apply_force_curr = True
                self.force = self._randomized_force * self.push_strength_ratio
                if hasattr(self.env, 'legged_robot'):
                    self.env.bullet_client.applyExternalForce(objectUniqueId=self.env.legged_robot.robot_id,
                                                              linkIndex=self._link_id,
                                                              forceObj=self.force,
                                                              posObj=self._force_position,
                                                              flags=self.env.bullet_client.LINK_FRAME)
                else:
                    assert hasattr(self.env, 'legged_robots')
                    for rob in self.env.legged_robots:
                        self.env.bullet_client.applyExternalForce(objectUniqueId=rob.robot_id,
                                                                  linkIndex=self._link_id,
                                                                  forceObj=self.force,
                                                                  posObj=self._force_position,
                                                                  flags=self.env.bullet_client.LINK_FRAME)
                        self.randomize_force()
                        self.force = self._randomized_force * self.push_strength_ratio

    def randomize_force(self):
        theta = np.random.uniform(0, 2 * pi)
        assert isinstance(self._horizontal_force_range, list)
        assert isinstance(self._vertical_force_range, list)
        self._horizontal_force = np.random.uniform(*self._horizontal_force_range)
        self._vertical_force = np.random.uniform(*self._vertical_force_range)
        self._randomized_force = np.array([
            self._horizontal_force * cos(theta),
            self._horizontal_force * sin(theta),
            self._vertical_force
        ])

    @property
    def curr_force(self):
        return self._randomized_force
