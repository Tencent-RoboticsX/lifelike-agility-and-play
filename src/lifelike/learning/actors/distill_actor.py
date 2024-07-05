import pickle
import time
import numpy as np
from tleague.actors.actor import Actor, logger
from tleague.actors.agent import DistillAgent
from tleague.utils.data_structure import DistillData
from tleague.utils.tl_types import is_inherit


class PureDistillActor(Actor):
    """ For pure distillation task usage.

    'Pure' is used to avoid confusion with the original TLeague's distillation for TStarBot-X.
    In this Actor, it requests model from model_pool only if the student policy has an rnn
    structure, and the Actor pulls the student model, not the teacher model which is given fixed
    at the beginning. So, this Actor reuses the common RL base Actor framework, but with
    different motivation.
    """
    # TODO (lxhan): all codes for n_agents > 1 are copied from TLeague and not checked
    def __init__(self, env, policy, league_mgr_addr, model_pool_addrs, pure_distill_model_path,
                 student_env, student_policy, student_policy_config=None, student_infserver_addr=None,
                 **kwargs):
        """ All kwargs are for base Actor. Do not touch them or move args in them before. """
        super(PureDistillActor, self).__init__(env, policy, league_mgr_addr, model_pool_addrs,
                                               data_type=DistillData,
                                               age_cls=DistillAgent, **kwargs)
        self.pure_distill_model_path = pure_distill_model_path
        self.model_path = None
        self._teacher_ratio = 1.0 if 'teacher_ratio' not in kwargs else kwargs['teacher_ratio']
        # Create student agent for rnn
        self.student_model = None
        # Whenever student policy either uses lstm or it use z_mlp style (temporal) policy, use_lstm = True
        self.use_student_policy = student_policy_config['use_lstm'] or student_policy_config['used_by_actor']
        self.distill_type = kwargs['pure_distill_type']
        if self.use_student_policy:
            # student_ob_space = student_env.observation_space.spaces[self._learning_agent_id]
            # student_ac_space = student_env.action_space.spaces[self._learning_agent_id]
            student_ob_space = student_env.observation_space if student_env is not None else \
            self.env.observation_space.spaces[self._learning_agent_id]
            student_ac_space = student_env.action_space if student_env is not None else \
            self.env.action_space.spaces[self._learning_agent_id]
            student_policy_config = {} if student_policy_config is None else student_policy_config
            student_policy_config['use_loss_type'] = 'none'
            student_policy_config['use_self_fed_heads'] = True
            # student_agent uses n_v=1, scope_name="student", use_gpu_id=-1, and compress=True by Default
            self.student_agent = DistillAgent(
                student_policy, student_ob_space, student_ac_space, n_v=1, scope_name="student",
                policy_config=student_policy_config, use_gpu_id=-1,
                infserver_addr=student_infserver_addr, compress=True)
            self.student_ds = DistillData(
                student_ob_space, student_ac_space, n_v=1, use_lstm=student_policy_config['use_lstm'],
                hs_len=1, distill_type=kwargs['pure_distill_type'])
        else:
            self._update_model_freq = 0  # do not update any model in the rollout loop
            self.student_policy = None

    def _update_agents_model(self, task):
        """ Fixed self model; never update it in this episode;
        once done the actor will relaunch a new episode; see what's done in Actor
        """
        logger.log('entering _update_agents_model', 'steps: {}'.format(self._steps),
                   level=logger.DEBUG + 5)
        if self.self_infserver_addr is None:
            if self.model_path is None:
                # Only update teacher model once
                self.model_path = self.pure_distill_model_path
                model1 = pickle.load(open(self.model_path, 'rb'))
                me_id = self._learning_agent_id  # short name
                self.agents[me_id].load_model(model1.model)
                self.self_model = model1
            if self._should_update_model(self.student_model, task.model_key1):
                # Update student model periodically
                model1 = self._model_pool_apis.pull_model(task.model_key1)
                self.student_agent.load_model(model1.model)
                self.student_model = model1
        if self.n_agents > 1 and self._should_update_model(self.oppo_model, task.model_key2):
            model2 = self._model_pool_apis.pull_model(task.model_key2)
            oppo_id = self._oppo_agent_id  # short name
            for agt in self.agents[oppo_id:]:
                agt.load_model(model2.model)
            self.oppo_model = model2
        logger.log('leaving _update_agents_model', level=logger.DEBUG + 5)

    def _push_data_to_learner(self, data_queue):
        """ In PGLearner, the TD-lambda return is computed here. In distill learner,
        there is no need to compute rewards

        :param data_queue:
        :return:
        """
        logger.log('entering _push_data_to_learner',
                   'steps: {}'.format(self._steps),
                   level=logger.DEBUG + 5)
        me_id = self._learning_agent_id  # short name
        # oppo_id = self._oppo_agent_id  # short name

        # initialize
        while True:
            # This loop is to deal with the case that future trajectory will be
            # used as the current feature. Under this case, online data collection
            # should wait for the teacher goes further while the students receives None
            last_obs, actions, reward, info, done, other_vars = data_queue.get()
            if last_obs[me_id] is None:
                # when student policy takes future states as input and
                # future states have not been collected yet
                continue
            break

        if self.use_student_policy:
            # no need to update student_agent here;
            # it will be updated in the main thread periodically
            self.student_agent.reset(last_obs[me_id])

        # loop infinitely to make the unroll on and on
        push_times = 0
        t0 = time.time()
        while True:
            data_model_id = self.task.model_key1
            unroll = []
            infos = []
            mask = False  # For the first frame in an unroll, there is no need to care
            # about whether it is just a start of a new episode, because even if it is a
            # new start, hidden state is zero and this is equivalent to mask=True. For
            # other cases, mask must be False. So, just set mask=False here.
            while True:
                # extend the unroll until a desired length
                me_action = actions[me_id]
                if isinstance(me_action, list):
                    me_action = tuple(me_action)
                # Make a `data` for this time step. The `data` is a DistillData compatible list,
                # see DistillData definition
                data = [last_obs[me_id], me_action]
                if self.distill_type == 'standard':
                    flatparam = other_vars['flatparam']
                    data += [flatparam]
                if self.use_student_policy:
                    # hidden state and temporal mask for student policy's rnn
                    data.extend([other_vars['student_state'], np.array(mask, np.bool)])
                data = self.student_ds.structure(data)
                mask = done
                unroll.append(data)

                if reward is not None:
                    # DevOps
                    pass

                while True:
                    last_obs, actions, reward, info, done, other_vars = data_queue.get()
                    if last_obs[me_id] is None:
                        continue
                    break

                if done:
                    if info:
                        infos.append(info)
                if mask and self.use_student_policy:
                    # no need to update student_agent here;
                    # it will be updated in the main thread periodically
                    self.student_agent.reset(last_obs[me_id])

                if len(unroll) >= self._unroll_length:
                    # need to collect a complete Noop duration
                    break
            unroll = [self.student_ds.flatten(_data) for _data in unroll]
            shapes = tuple(data.shape for data in unroll[0])
            unroll_np = np.concatenate([b.reshape(-1) for a in unroll for b in a])
            self._learner_apis.push_data((data_model_id, unroll_np, infos, shapes))

            logger.log(f"Pushed one unroll to learner at time "
                       f"{time.strftime('%Y%m%d%H%M%S')}",
                       level=logger.DEBUG + 5)
            push_times += 1
            if push_times % 10 == 0:
                push_fps = push_times * self._unroll_length / (time.time() - t0 + 1e-8)
                t0 = time.time()
                logger.log("push fps: {}".format(push_fps))

    def _rollout_an_episode(self):
        """Perform trajectory rollout until an episode ends.

        Data are produced by env-agent interaction at each time step. The data are
        put in the _data_queue, and will be sent to (remote) Learner in a separate
        Thread.
        """
        self._steps = 0
        me_id = self._learning_agent_id  # short name
        oppo_id = self._oppo_agent_id  # short name
        logger.log('episode begins with the task: {}'.format(str(self.task)))

        # passing me and oppo hyperparams to the arena interface
        assert self.task.hyperparam is not None
        oppo_hyperparam = None
        if self.n_agents > 1:
            logger.log('pulling oppo hyperparam of model key {}'.format(
                self.task.model_key2))
            oppo_hyperparam = self._model_pool_apis.pull_attr(attr='hyperparam',
                                                              key=self.task.model_key2)
            logger.log('Done pulling oppo hyperparam')
        oppo_inter_kwargs = ({} if oppo_hyperparam is None
                             else oppo_hyperparam.__dict__)
        inter_kwargs = ([self.task.hyperparam.__dict__]
                        + [oppo_inter_kwargs] * (self.n_agents - 1))

        # agent, env reset
        obs = self.env.reset(inter_kwargs=inter_kwargs)
        for agt, ob in zip(self.agents, obs):
            agt.reset(ob)
        self._update_agents_model(self.task)  # for agent Neural Net parameters
        obs_student = (self.env.to2so(obs[self._learning_agent_id]), ) if hasattr(self.env, 'to2so') \
            else obs[self._learning_agent_id]
        me_reward_sum = 0.0
        self.time_beg = time.time()
        self._update_hyperparam(self.task)
        if hasattr(self.task.hyperparam, 'discounted_teacher_ratio'):
            discounted_teacher_ratio = self.task.hyperparam.discounted_teacher_ratio
            self._teacher_ratio *= discounted_teacher_ratio
        if self._teacher_ratio < 1:
            assert self.use_student_policy
        apply_teacher_action = np.random.rand() < self._teacher_ratio
        logger.log('Right now {} policy is used to sample. The ratio of using teacher policy to sample is {}'.format(
            'teacher' if apply_teacher_action else 'student', self._teacher_ratio))
        self._changed_task = False
        t0 = time.time()
        while True:
            self._steps += 1
            # predictions for each agent
            predictions = self._parallel.run((self._agent_pred, ob, i)
                                             for i, ob in enumerate(obs))
            me_prediction = predictions[me_id]
            action, extra_vars = me_prediction[0], me_prediction[1]
            me_action = action
            # book-keep obs in previous step
            actions = [me_action] + predictions[oppo_id:]
            label_action = self.env.ta2sa(me_action) if hasattr(self.env, 'ta2sa') else me_action
            distill_sa_pair = [(obs_student, ), (label_action,)]
            if self.use_student_policy:
                if apply_teacher_action:
                    # hidden state and temporal mask for student policy's rnn
                    extra_vars['student_state'] = self.student_agent.state
                    self.student_agent.update_state(obs_student)
                else:
                    action_student, extra_vars_student = self.student_agent.forward_squeezed(obs_student)
                    extra_vars.update({'student_' + key: extra_vars_student[key] for key in extra_vars_student})
                    actions[0] = action_student  # replace the executed action with student action
            # agent-env interaction; note that env.step should be able to distinguish teacher
            # and student actions by their shape or type below
            obs, reward, done, info = self.env.step(actions)
            obs_student = (self.env.to2so(obs[self._learning_agent_id]),) if hasattr(self.env, 'to2so') \
                else obs[self._learning_agent_id]

            me_rwd_scalar = self._reward_shape(reward[me_id])
            me_reward_sum += me_rwd_scalar

            if self._enable_push:
                # put the interested data (obs, rwd, act, ... for each agent) into the
                # _data_queue, which is watched in another Thread (the _push_data_to_learner()
                # method) that the data are dequeued and sent to remote Learner
                rwd_to_push = me_rwd_scalar if self.rwd_shape else reward[me_id]
                rwd_to_push = np.asarray(rwd_to_push, np.float32)
                if rwd_to_push.shape == ():
                    rwd_to_push = np.asarray([rwd_to_push], np.float32)
                if self.use_oppo_obs:
                    extra_vars['oppo_state'] = self.agents[self._oppo_agent_id]._last_state
                if done:
                    if not isinstance(info, dict):
                        info = {}
                data_tuple = (*distill_sa_pair, rwd_to_push, info, done, extra_vars)
                if self._data_queue.full():
                    logger.log("Actor's queue is full.", level=logger.WARN)
                self._data_queue.put(data_tuple)
                logger.log('successfully put one tuple.', level=logger.DEBUG)

            if self._steps % self._log_interval_steps == 0:
                logger.log('_rollout_an_episode,', 'steps: {},'.format(self._steps),
                           'data qsize: {}'.format(self._data_queue.qsize()),
                           'rollout fps: {}'.format(self._steps / (time.time() - t0 + 1e-8)))

            if done:
                # an episode ends
                if self._replay_dir:
                    self._save_replay()
                if not apply_teacher_action:
                    self.log_kvs(me_reward_sum, info)
                    info['teacher_ratio'] = self._teacher_ratio
                # if self._changed_task:
                #     # if the actor task has changed during an episode, then it indicates
                #     # that the model has changed during the episode and so the info['outcome']
                #     # should not be counted for that player in league
                #     return None, info
                # else:
                #     return None, info
                return None, info

            if self._update_model_freq and self._steps % self._update_model_freq == 0:
                # time to update the model for each agent
                if (self._enable_push and
                        self._model_pool_apis.pull_attr(
                            'freezetime', self.task.model_key1) is not None):
                    # Current task (learning period) finishes, start a new task or continue
                    self._finish_task(self.task, None)  # notify early abort
                    last_task = self.task
                    self.task = self._request_task()  # try to continue
                    if not is_inherit(last_task.model_key1, self.task.model_key1):
                        self.log_kvs(me_reward_sum, info)
                        return None, info
                    if last_task.model_key2 != self.task.model_key2:
                        self._changed_task = True
                self._update_agents_model(self.task)
