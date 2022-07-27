from gym import utils as gym_utils

from wam_envs.envs import WAMWipe
import numpy as np

from gym.envs.registration import registry, register, make, spec

from nht.expNHT import NHT
import tensorflow as tf
from wam_envs import utils
from baselines.common.tf_util import get_session


class NHT_WAMWipeEnv(WAMWipe.WAMWipeEnv, gym_utils.EzPickle):
    def __init__(self, action_dim=2, NHT_path=None, reward_type='sparse'):
        self.action_dim = action_dim

        #cond_size = 7  # 7 joint angles
        cond_size = 10 # 7 joint angles, 3x1 EE frame origin pos
        #cond_size = 19 # 7 joint angles, 3x3 EE frame orientation, 3x1 EE frame origin pos

        # placeholders to construct NHT computational graph
        self.cond_inp = tf.placeholder(shape=[None, cond_size], dtype=tf.float32)
        self.z = tf.placeholder(shape=[None, action_dim], dtype=tf.float32)

        # NHT model (outputs basis for actuation subspace given current observation)
        self.NHT_model = NHT(action_dim=action_dim, output_dim=7, cond_dim=cond_size)
        self.NHT_model.h = tf.keras.models.load_model(NHT_path) # loads weights
        print(self.NHT_model.h.summary())

        self.H_hat = self.NHT_model._get_map(self.cond_inp)
        self.tfsess = get_session()

        WAMWipe.WAMWipeEnv.__init__(self, reward_type=reward_type, n_actions=action_dim)
        gym_utils.EzPickle.__init__(self)
    
    def _set_action(self, action):
        '''
            This method is overrides the _set_action method in wam_env.WAMEnv
        '''
        k = self.action_dim
        assert action.shape == (k,)

        o_r = self._get_cond_inp()
        H_hat = self.tfsess.run(self.H_hat, feed_dict={self.cond_inp: o_r})
        a = np.expand_dims(action,1) # turn action from agent to column vector tensor (with batch dimension)
        qdot = np.matmul(H_hat.squeeze(0), a)
        qdot = qdot.squeeze()

        WAMWipe.WAMWipeEnv._set_action(self, qdot)

    def _get_cond_inp(self):
        q = self._get_joint_angles().copy()
        x = utils.WAM_fwd_kinematics(utils.get_joint_angles(self.sim), self.n_joints).copy()
        cond_inp = np.expand_dims(np.concatenate((x,q),0),0)
        return cond_inp



for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    register(
        id='NHT_WAMWipe{}-v1'.format(suffix),
        entry_point='wam_envs.action_interfaces.NHT_WAMWipe:NHT_WAMWipeEnv',
        kwargs=kwargs,
        max_episode_steps=200,
    )