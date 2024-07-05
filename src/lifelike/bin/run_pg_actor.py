import os
import random

from absl import app
from absl import flags

from tleague.actors.ppo_actor import PPOActor
from lifelike.learning.actors.distill_actor import PureDistillActor
from tleague.utils import read_config_dict
from tleague.utils import import_module_or_data
from tleague.utils import kill_sc2_processes_v2

FLAGS = flags.FLAGS
flags.DEFINE_string("league_mgr_addr", "localhost:10005",
                    "League manager address.")
flags.DEFINE_string("model_pool_addrs", "localhost:10003:10004",
                    "Model Pool address.")
flags.DEFINE_string("learner_addr", "localhost:10001:10002",
                    "Learner address")
# RL related
flags.DEFINE_integer("unroll_length", 32, "unroll length")
flags.DEFINE_integer("n_v", 1, "value length")
flags.DEFINE_integer("update_model_freq", 32, "update model every n steps")
flags.DEFINE_string("env", None, "task env, e.g., sc2")
flags.DEFINE_string("outer_env", None, "envs that are not in TLeague")
flags.DEFINE_string("env_config", "",
                    "python dict config used for env. "
                    "e.g., {'replay_dir': '/root/replays/ext471_zvz'}")
flags.DEFINE_string("interface_config", "",
                    "python dict config used for Arena interface. "
                    "e.g., {'zstat_data_src_path': '/root/replay_ds/zstat'}")
flags.DEFINE_string("policy", "tpolicies.ppo.policies.DeepMultiHeadMlpPolicy",
                    "policy used")
flags.DEFINE_string("policy_config", "", "config used for policy")
flags.DEFINE_string("type", "PPO", "PPO|PPO2|Vtrace actor type")
flags.DEFINE_string("self_infserver_addr", "", "infserver_addr self_agent used.")
flags.DEFINE_string("distill_infserver_addr", "", "infserver_addr distill_agent used.")
flags.DEFINE_string("post_process_data", None,
                    "post process of (X, A), drop useless mask in SC2.")
flags.DEFINE_boolean("compress", True, "whether data is compressed for infserver")
flags.DEFINE_boolean("rwd_shape", True, "do reward shape in actor")
flags.DEFINE_boolean("distillation", False, "use distillation policy")
flags.DEFINE_string("distill_policy", "tpolicies.ppo.policies.DeepMultiHeadMlpPolicy", "distill policy")
flags.DEFINE_string("distill_policy_config", "", "config used for distill policy")
flags.DEFINE_string("pure_distill_model_path", None, "directly provide the pure distill model")
flags.DEFINE_string("student_env", None, "in pure distillation, the env in the learner side")
flags.DEFINE_string("student_env_config", "", "python dict config used for student env. ")
flags.DEFINE_string("student_policy", None, "in pure distillation, the policy in the learner side")
flags.DEFINE_string("student_policy_config", "", "config used for student policy")
flags.DEFINE_string("pure_distill_type", "standard", "pure distill type: standard or any non-standard")
# printing/logging
flags.DEFINE_integer("verbose", 11,
                     "verbosity. The smaller, the noisier. Reference:"
                     "10:DEBUG, 20:INFO, 30: WARN, 40: ERROR, 50:DISABLED")
flags.DEFINE_integer("log_interval_steps", 51,
                     "frequency of printing log in steps")
flags.DEFINE_string("replay_dir", "", "replay dir when available")
#
flags.DEFINE_boolean(
    "reboot_on_failure", False,
    "Should actor reboot on failure. NOTE: Before rebooting, it kills ALL the "
    "Children SC2 processes on the machine. Use it carefully. Some hints:"
    "For k8s, we don't need rebooting as k8s can do it. "
)
flags.DEFINE_string("data_server_version", "v2", "v2|v1")


def main(_):
    if FLAGS.replay_dir:
        os.makedirs(FLAGS.replay_dir, exist_ok=True)

    if FLAGS.pure_distill_model_path is not None:
        pure_distill_model_path = eval(FLAGS.pure_distill_model_path)
        pure_distill_model_path = random.choice(pure_distill_model_path)
        pure_distill_model_path, env_config_update = pure_distill_model_path
    else:
        assert FLAGS.type != 'Distill', 'DistillActor must be provided with model paths'
        pure_distill_model_path, env_config_update = None, {}
    env_config = read_config_dict(FLAGS.env_config)
    env_config.update(env_config_update)
    if FLAGS.outer_env is not None:
        create_env_func = import_module_or_data(FLAGS.outer_env)
        env = create_env_func(**env_config)
    else:
        raise NotImplementedError("No environment defined.")
    policy = import_module_or_data(FLAGS.policy)
    policy_config = read_config_dict(FLAGS.policy_config)
    if FLAGS.distillation:
        distill_policy = import_module_or_data(FLAGS.distill_policy)
        distill_policy_config = read_config_dict(FLAGS.distill_policy_config)
    else:
        distill_policy = None
        distill_policy_config = {}

    if FLAGS.student_env is not None:
        student_env_config = read_config_dict(FLAGS.student_env_config)
        create_student_env_func = import_module_or_data(FLAGS.student_env)
        student_env = create_student_env_func(**student_env_config)
    else:
        student_env = None
    if FLAGS.student_policy is not None:
        student_policy = import_module_or_data(FLAGS.student_policy)
        student_policy_config = read_config_dict(FLAGS.student_policy_config)
    else:
        student_policy = None
        student_policy_config = None

    post_process_data = None
    if FLAGS.post_process_data is not None:
        post_process_data = import_module_or_data(FLAGS.post_process_data)
    if FLAGS.type == 'PPO':
        actor_class = PPOActor
    elif FLAGS.type == 'PureDistill':
        actor_class = PureDistillActor
    else:
        raise KeyError(f'Not recognized learner type {FLAGS.type}!')
    actor = actor_class(env, policy,
                        policy_config=policy_config,
                        league_mgr_addr=FLAGS.league_mgr_addr or None,
                        model_pool_addrs=FLAGS.model_pool_addrs.split(','),
                        learner_addr=FLAGS.learner_addr,
                        unroll_length=FLAGS.unroll_length,
                        update_model_freq=FLAGS.update_model_freq,
                        n_v=FLAGS.n_v,
                        verbose=FLAGS.verbose,
                        log_interval_steps=FLAGS.log_interval_steps,
                        rwd_shape=FLAGS.rwd_shape,
                        distillation=FLAGS.distillation,
                        distill_policy=distill_policy,
                        distill_policy_config=distill_policy_config,
                        replay_dir=FLAGS.replay_dir,
                        compress=FLAGS.compress,
                        self_infserver_addr=FLAGS.self_infserver_addr or None,
                        distill_infserver_addr=FLAGS.distill_infserver_addr or None,
                        post_process_data=post_process_data,
                        pure_distill_model_path=pure_distill_model_path,
                        student_env=student_env,
                        student_policy=student_policy,
                        student_policy_config=student_policy_config,
                        pure_distill_type=FLAGS.pure_distill_type,
                        data_server_version=FLAGS.data_server_version)

    n_failures = 0
    while True:
        try:
            actor.run()
        except ValueError:
            raise ValueError
        except RuntimeError as e:
            if not FLAGS.reboot_on_failure:
                raise e
            print("Actor crushed no. {}, the exception:\n{}".format(n_failures, e))
            n_failures += 1
            print("Rebooting...")
            kill_sc2_processes_v2()


if __name__ == '__main__':
    app.run(main)
