from lifelike.sim_envs.pybullet_envs.max_game.bullet_static_entities import BulletStatics, BulletStaticsV2, \
    BulletStaticsV3, BulletStaticsV4


class GameManager:
    def __init__(self, bullet_client, element_config: dict, holes=False, version='v2'):
        self.bullet_client = bullet_client
        if version == 'v2':
            self.static_objs = BulletStaticsV2(bullet_client, holes=holes)
        elif version == 'v3':
            self.static_objs = BulletStaticsV3(bullet_client)
        elif version == 'v4':
            self.static_objs = BulletStaticsV4(bullet_client, element_config)
        else:
            self.static_objs = BulletStatics(bullet_client)

    def reset(self, offset_range_low=0.0, offset_range_high=0.0):
        self.static_objs.randomize_height(offset_range_low, offset_range_high)
