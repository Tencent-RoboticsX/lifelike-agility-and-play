"""
Example of load trained model and test it in simulation
"""
import importlib
import joblib
from absl import app
from absl import flags
from lifelike.sim_envs.pybullet_envs.create_pybullet_envs import create_tracking_env
from tleague.actors.agent import PGAgent

flags.DEFINE_string("model_path", 'data/models/primitive_level.model', "the trained PMC model path")
flags.DEFINE_string("data_path", 'data/mocap_data', "the mocap data path")

FLAGS = flags.FLAGS


def main(_):
    reward_weights = {
        'joint_pos': 0.3,
        'joint_vel': 0.05,
        'end_effector': 0.1,
        'root_pose': 0.5,
        'root_vel': 0.05,
    }
    env_config = {
        'arena_id': 'LeggedRobotTracking',
        'render': True,
        'data_path': FLAGS.data_path,
        'control_freq': 50.0,
        'prop_type': ['joint_pos', 'joint_vel', 'root_ang_vel_loc', 'root_lin_vel_loc', 'e_g'],
        'prioritized_sample_factor': 3.0,
        'set_obstacle': True,
        'obstacle_height': 0.2,
        'kp': 50.0,
        'kd': 0.5,
        'max_tau': 18,
        'reward_weights': reward_weights,
    }
    policy_config = {
        'batch_size': 1,
        'rollout_len': 1,
        'test': True,
        'use_loss_type': 'none',
        'z_prior_type': 'VQ',
        'use_value_head': True,
        'rms_momentum': 0.0001,
        'append_hist_a': True,
        'main_activation_func': 'relu',
        'n_v': 1,
        'z_len': 32,
        'num_embeddings': 256,
        'conditional': True,
        'norm_z': False,
        'bot_neck_z_embed_size': 32,
        'bot_neck_prop_embed_size': 64,
        'logvar_prior': 0.0,
    }
    policy = "lifelike.networks.legged_robot.pmc_net.pmc_net"

    # env, policy and model file
    env = create_tracking_env(**env_config)
    ob_space, ac_space = env.observation_space, env.action_space

    policy_module, _ = policy.rsplit(".", 1)
    policy = importlib.import_module(policy_module)

    agent = PGAgent(policy, ob_space, ac_space, policy_config=policy_config)

    model = joblib.load(FLAGS.model_path)
    agent.load_model(model.model)

    obs = env.reset()
    agent.reset(obs[0])

    # evaluate model
    epi_length = 0
    returns = 0
    sum_r = 0
    n_episodes = 0
    obss = [obs]
    while True:
        act = agent.step(obs, argmax=True)
        obs, rwd, done, _ = env.step([act])
        obss.append(obs)
        returns += rwd[0]
        epi_length += 1

        if done:
            obs = env.reset()
            agent.reset(obs[0])
            sum_r += returns
            n_episodes += 1
            print("reward_sum: {}, episode length: {}, n_episodes: {}, average_reward_sum: {:.2f}".format(
                returns, epi_length, n_episodes, sum_r / float(n_episodes)))
            epi_length = 0
            returns = 0


if __name__ == '__main__':
    app.run(main)
