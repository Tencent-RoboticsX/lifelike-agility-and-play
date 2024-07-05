import logging
from collections import namedtuple
import numpy as np
import tensorflow as tf


PMCInputs = namedtuple('TrackingZInputs', [
    'X',
    'A',
    'neglogp',
    'discount',
    'r',
    'R',
    'V',
    'flatparam',  # logits or mean and std
])

PMCOutputs = namedtuple('TrackingZOutputs', [
    'self_fed_heads',
    'outer_fed_heads',
    'loss',
    'vars',
    'endpoints',
    'value_head',
    'ob_rms',
])

PMCTrainableVariables = namedtuple('TrackingZTrainableVariables', [
    'all_vars',
    'vf_vars',
    'pf_vars',
    'ob_stat',
    'lstm_vars',
])

PMCLosses = namedtuple('TrackingZLosses', [
    'total_reg_loss',
    'pg_loss',
    'value_loss',
    'entropy_loss',
    'loss_endpoints'  # OrderedDict
])


def reset_hs_func(hs_len=0, z_len=0):
    return np.concatenate([np.zeros(shape=(hs_len-z_len,)), np.random.randn(z_len)])


class PMCConfig(object):
    def __init__(self, ob_space, ac_space, **kwargs):
        # logical settings
        self.test = False  # negate is_training
        self.batch_size = None

        # network architecture related
        self.use_lstm = True
        self.use_value_head = False  # for RL training/testing
        self.use_loss_type = 'none'  # {'rl' | 'none'}
        self.use_self_fed_heads = False
        self.rms_momentum = 0.0001  # rms_v2 momentum
        self.main_activation_func = 'relu'
        self.append_hist_a = False

        # lstm settings
        self.nrollout = None
        self.rollout_len = 1
        self.hs_len = 64 * 3  # for separate pi, vf, hs_z_enc or z_prev
        self.nlstm = 32
        self.z_len = 8
        self.reset_hs_func = reset_hs_func
        self.norm_z = True
        self.forget_bias = 1.0  # usually it's 1.0
        self.lstm_duration = 4  # this is for k_lstm
        self.lstm_dropout_rate = 0.0
        self.lstm_cell_type = 'lstm'
        self.lstm_layer_norm = True

        # z settings
        self.z_len = 8
        self.num_embeddings = 256
        self.norm_z = False
        self.bot_neck_z_embed_size = 64
        self.bot_neck_prop_embed_size = 64
        self.z_prior_type = 'VQ'

        # hyper-parameters
        self.alpha = 0.95
        if self.z_prior_type == 'Gaussian':
            self.z_logvar_prior = 0.0
        elif self.z_prior_type == 'TemporalGaussian':
            self.z_logvar_prior = np.log(1 - self.alpha ** 2)
        else:
            pass
        self.logstd_init = -2.0

        # value head related
        self.n_v = 1
        self.reward_weights = None
        self.lam = 0.95

        # ppo related
        self.clip_range = 0.1
        self.clip_range_lower = 0.1

        # common embedding settings
        self.embed_dim = 256

        # regularization settings
        self.weight_decay = None  # None means not use. use value, e.g., 0.0005

        # loss related settings
        self.sync_statistics = None

        # finally, cache the spaces
        self.ob_space = ob_space
        self.ac_space = ac_space

        self.conditional = False
        self.total_timesteps = None
        # parameter for weight decay
        self.z_prior_param1 = 0.003
        self.z_prior_param2 = 6 * 10e8
        self.z_prior_param3 = 0.2

        # allow partially overwriting
        for k, v in kwargs.items():
            if k not in self.__dict__:
                logging.info(
                    'unrecognized config k: {}, v: {}, ignored'.format(k, v))
            self.__dict__[k] = v

        # consistency check
        if self.use_lstm:
            assert self.batch_size is not None, 'lstm requires a specific batch_size.'
            self.nrollout = self.batch_size // self.rollout_len
            assert (self.rollout_len % self.lstm_duration == 0 or self.rollout_len == 1)
            if self.lstm_cell_type == 'k_lstm':
                if self.z_prior_type == 'Gaussian':
                    assert self.hs_len == 3 * 2 * self.nlstm + 1
                elif self.z_prior_type == 'TemporalGaussian':
                    assert self.hs_len - self.z_len == 2 * 2 * self.nlstm + 1
                else:
                    raise NotImplementedError
            elif self.lstm_cell_type == 'lstm':
                if self.z_prior_type == 'Gaussian':
                    assert self.hs_len == 3 * 2 * self.nlstm
                elif self.z_prior_type == 'TemporalGaussian':
                    assert self.hs_len - self.z_len == 2 * 2 * self.nlstm
                elif self.z_prior_type == 'VQ':
                    assert self.hs_len == 3 * 2 * self.nlstm
                else:
                    raise NotImplementedError
            else:
                raise NotImplemented('Unknown lstm_cell_type {}'.format(
                    self.lstm_cell_type))

        # activate func
        if self.main_activation_func == 'relu':
            self.main_activation_func_op = tf.nn.relu
        elif self.main_activation_func == 'tanh':
            self.main_activation_func_op = tf.nn.tanh
        else:
            raise NotImplementedError

        # update reset_hs_func args
        if self.reset_hs_func is not None:
            self.reset_hs_func.__defaults__ = (self.hs_len, self.z_len)
