from tleague.learners.pg_learner import PGLearner, tf
from tleague.utils.data_structure import DistillData


class PureDistillLearner(PGLearner):
    """Learner for Pure Distillation.

    'Pure' is used to avoid confusion with the original TLeague's distillation for TStarBot-X

    Reuse PGLearner without using the RL stuffs therein.
    Note: PGLearner pushes model to model_pool periodically by default. Normally,
    distill learner does not need to push model to model_pool, because
    the actor never use the model. However, when the student network in distill
    learner contains rnn structures, it requires hidden state, while
    teacher might not have an rnn structure. In such cases, the most correct
    way to learn the rnn is that the learner has to push its model to
    model_pool, and the distill actor pulls the student model and use it to
    pass the obs from the teacher to get hidden state and push the data back
    to learner together with the obs, action from teacher. This is exactly the same
    as the imitation learner in TStarBot-X (see imitation_learner3 in TLeague).
    In imitation_learner3, there is:

    self.should_push_model = (self.rnn and self.rank == 0)

    then, you can see what it has done. Accordingly, the replay_actor takes an arg named
    use_policy=True or False, indicating that it wether pulls learner model for computing
    hidden state. Here, the DistillLearner inherits PGLearner,
    so it pushes model to model_pool by default, no matter whether it uses rnn (whether
    the distill actor pulls model), for simplicity
    (imitation_learner3 is better, because without rnn, pushing model to
    model_pool increases additional costs).
    """

    def __init__(self, league_mgr_addr, model_pool_addrs, learner_ports, rm_size,
                 batch_size, ob_space, ac_space, policy, gpu_id, **kwargs):
        super(PureDistillLearner, self).__init__(
            league_mgr_addr, model_pool_addrs, learner_ports, rm_size, batch_size,
            ob_space, ac_space, policy, gpu_id, data_type=DistillData, **kwargs)

    def _build_train_op(self):
        params_pi = tf.trainable_variables('model')  # notice
        grads_and_vars = self.trainer.compute_gradients(self.loss, params_pi)

        grads_and_vars, self.clip_grad_norm, self.nonclip_grad_norm = self.clip_grads_vars(
          grads_and_vars, self.clip_vars, self.max_grad_norm)

        self._train_batch = self.trainer.apply_gradients(grads_and_vars)
