import logging
from collections import namedtuple
import tensorflow as tf


SEPMCInputs = namedtuple('HMLMultiHeadV2Inputs', [
    'X',
    'A',
    'neglogp',
    'R',
    'V',
    'discount',
    'r',
    'S',  # rnn hidden state
    'M',  # rnn hidden state mask
    'flatparam',  # for distillation
])

SEPMCOutputs = namedtuple('HMLMultiHeadV2Outputs', [
    'self_fed_heads',
    'outer_fed_heads',
    'loss',
    'vars',
    'endpoints',
    'value_head',
    'ob_rms',
    'S',
])

SEPMCTrainableVariables = namedtuple('HMLMultiHeadV2TrainableVariables', [
    'all_vars',
    'vf_vars',
    'pf_vars',
    'ob_stat',
    'lstm_vars',
])

SEPMCLosses = namedtuple('HMLMultiHeadV2Losses', [
    'total_reg_loss',
    'pg_loss',
    'value_loss',
    'entropy_loss',
    'distill_loss',
    'loss_endpoints'  # OrderedDict
])


class SEPMCConfig(object):
    def __init__(self, ob_space, ac_space, **kwargs):
        # logical settings
        self.test = False  # negate is_training
        self.batch_size = None
        self.distillation = False
        self.llc_param_type = 'freeze'  # freeze or trainable
        self.mlc_param_type = 'freeze'  # freeze or trainable
        self.distill_llc = False
        self.distill_z = False

        # network architecture related
        self.use_lstm = True
        self.use_value_head = False  # True for testing; False for rl training
        self.use_loss_type = 'none'  # {'rl' | 'none'}
        self.use_self_fed_heads = False
        self.outer_control_spd = False
        self.rms_momentum = 0.0001  # rms_v2 momentum
        self.main_activation_func = 'relu'
        self.append_hist_a = False
        self.take_percept_1d = True

        # llc
        self.llc_light = False

        # lstm settings
        self.nrollout = None
        self.rollout_len = 1
        self.hs_len = 64 * 4  # for separate pi and vf lstm, and z
        self.nlstm = 32
        self.z_len = 256
        self.z_len_llc = 32
        self.discrete_z = False
        self.norm_z = True
        self.forget_bias = 1.0  # usually it's 1.0
        self.lstm_duration = 4  # this is for k_lstm
        self.lstm_dropout_rate = 0.0
        self.lstm_cell_type = 'lstm'
        self.lstm_layer_norm = True

        self.expert_lstm = True

        # z settings
        self.z_len = 8
        self.norm_z = True
        self.bot_neck_z_embed_size = 64
        self.bot_neck_prop_embed_size = 64

        # hyper-parameters
        self.isolate_z_logvar = False
        self.z_logvar_init = 0.0
        self.logstd_init = -2.0
        self.clip_range = 0.1
        self.clip_range_lower = 100.0

        # value head related
        self.n_v = 1
        self.reward_weights = None
        self.lam = 0.95

        # common embedding settings
        self.embed_dim = 256

        # regularization settings
        self.weight_decay = None  # None means not use. use value, e.g., 0.0005

        # loss related settings
        self.sync_statistics = None

        # finally, cache the spaces
        self.ob_space = ob_space
        self.ac_space = ac_space

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
                assert self.hs_len == 4 * 2 * self.nlstm + 1
            elif self.lstm_cell_type == 'lstm':
                assert self.hs_len == 4 * 2 * self.nlstm
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
