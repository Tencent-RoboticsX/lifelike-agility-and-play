import logging
import math
from collections import namedtuple

ZLSTMMLPInputs = namedtuple('ZLSTMMLPInputs', [
    'X',
    'A',
    'neglogp',
    'R',
    'V',
    'S',  # rnn hidden state
    'M',  # rnn hidden state mask
    'flatparam',  # logits or mean and std
])

ZLSTMMLPOutputs = namedtuple('ZLSTMMLPOutputs', [
    'self_fed_heads',
    'outer_fed_heads',
    'loss',
    'vars',
    'endpoints',
    'value_head',
    'ob_rms',
    'S',  # rnn hidden state
])

ZLSTMMLPTrainableVariables = namedtuple('ZLSTMMLPTrainableVariables', [
    'all_vars',
    'encoder_vars',
    'decoder_vars',
    'lstm_vars',
])

ZLSTMMLPLosses = namedtuple('ZLSTMMLPLosses', [
    'total_reg_loss',
    'pg_loss',
    'value_loss',
    'entropy_loss',
    'loss_endpoints'  # OrderedDict
])


class ZLSTMMLPConfig(object):
    def __init__(self, ob_space, ac_space, **kwargs):
        # logical settings
        self.test = False  # negate is_training
        self.batch_size = None

        # network architecture related
        self.use_loss_type = 'distill'
        self.distill_loss_type = 'supervised'  # standard, action_var or supervised
        self.action_var = 0.15
        self.bn_type = 'rms_v2'  # batch normalization type
        self.rms_momentum = 0.0001  # rms_v2 momentum

        # lstm settings
        self.nrollout = None
        self.rollout_len = 1
        self.hs_len = self.z_len = 8  # for z

        # common embedding settings
        self.enc_dim = 128
        self.dec_dim = 256
        self.command_dim = 3

        # hyper-parameters
        self.alpha = 0.95
        self.beta = 0.1
        self.sigma = math.sqrt(1 - self.alpha ** 2)
        self.logvar_prior = math.log(self.sigma ** 2)

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
        assert self.batch_size is not None, 'z requires a specific batch_size.'
        assert self.batch_size % self.rollout_len == 0, 'batch size must be divided by rollout_len'
        self.nrollout = self.batch_size // self.rollout_len
