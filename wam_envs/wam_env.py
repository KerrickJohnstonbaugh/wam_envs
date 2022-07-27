import os
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding
from wam_envs import utils

'''
 WAM based analog to RobotEnv class that is distributed with openai gym-0.15.7
'''

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500


class WAMEnv(gym.GoalEnv):
    def __init__(self, model_path, initial_qlist, n_substeps, robot_mode, n_actions=7):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        print(fullpath)
        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        
        

        self.robot_mode = robot_mode
        
        self.n_joints = 7
        self.n_actions = n_actions

        self.distance_threshold = 0.03

        self.joint_names = [
            'wam/base_yaw_joint',
            'wam/shoulder_pitch_joint',
            'wam/shoulder_yaw_joint',
            'wam/elbow_pitch_joint',
            'wam/wrist_yaw_joint',
            'wam/wrist_pitch_joint',
            'wam/palm_yaw_joint'
        ]

        self.initial_qlist = initial_qlist

        # build dict of joint angles for mujoco simulator
        initial_qpos = {}
        for ind, key in enumerate(self.joint_names):
            initial_qpos[key] = self.initial_qlist[ind]
        
        self.initial_qpos = initial_qpos
        self._env_setup()
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.braking = False

        self.goal = self._sample_goal()
        obs = self._get_obs()

        max_abs_action = np.inf
        self.action_space = spaces.Box(-max_abs_action, max_abs_action, shape=(self.n_actions,), dtype='float32')
        
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'q_dot': self._get_joint_vel(),
            'q': self._get_joint_angles(),
            'x': obs['achieved_goal'], # Cartesian position of end-effector
            'goal': self.goal,
            'observation': obs['observation'],
            'braking': self.braking,
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        
        super(WAMEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.reset_goal()
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    # WAM methods
    # ----------------------------

    def _get_joint_angles(self):
        if self.robot_mode == '4dof':
            return utils.get_joint_angles(self.sim)[:4]
        else:
            return utils.get_joint_angles(self.sim)
    
    def _get_joint_vel(self):
        if self.robot_mode == '4dof':
            return utils.get_joint_vels(self.sim)[:4]
        else:
            return utils.get_joint_vels(self.sim)


    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = utils.goal_distance(achieved_goal, goal)

        if self.reward_type == 'sparse':
            return -1+np.squeeze(info['is_success'])
            
        else:
            return -d

    def _set_action(self, action):
        '''
            This method is overridden in action interface subclasses 
        '''
        assert action.shape == (7,)  # not necessarily the same as n_actions, because n_actions < 7 for interface subclasses
        action = action.copy()  # ensure that we don't change the action outside of this scope
        action = self.check_braking(action)
        
        utils.ctrl_set_qvel(self.sim, action, mode=self.robot_mode)


    def _get_obs(self):
        """Returns the observation.
        """
        q = utils.get_joint_angles(self.sim)
        pos = utils.WAM_fwd_kinematics(q, self.n_joints)

        T = utils.get_kinematics_mat(utils.get_joint_angles(self.sim), self.n_joints) # end effector frame matrix, computed according to https://support.barrett.com/wiki/WAM/KinematicsJointRangesConversionFactors
        flat_T = T[:3,:4].flatten()
        
        obs = np.concatenate([
            q, flat_T
        ])
            
        return {
            'observation': obs.copy(),
            'achieved_goal': pos.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        body_id = self.sim.model.body_name2id('wam/wrist_palm_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self._reset_constraints()
        self.sim.set_state(self.initial_state)

        self.sim.forward()

        return True


    def check_braking(self, qdot):
        joint_lims = utils.get_joint_limits()
        q = self._get_joint_angles()
        # dont try to move joints past their limit (if you do, weird stuff happens, e.g. other joints move)
        # store whether or not the robot experienced braking in the info so we can remove these episode from demonstrations?
        # Do we want to remove braking episodes? Maybe not, since the RL agent might have to overcome braking in some cases
        # Unfortunately, sometimes when braking is experienced the inverse Jacobian control fails to recover.
        for joint_ind in range(len(q)):
            if np.abs(q[joint_ind]-joint_lims[joint_ind][0]) < 0.15:
                if qdot[joint_ind] < 0.0:
                    #print('braking')
                    # print('joint:', joint_ind)
                    self.braking = True
                    qdot[joint_ind] = 0.0
            if np.abs(q[joint_ind]-joint_lims[joint_ind][1]) < 0.15:
                if qdot[joint_ind] > 0.0:
                    #print('braking')
                    # print('joint:', joint_ind)
                    self.braking = True
                    qdot[joint_ind] = 0.0
        
        return qdot

    # Extension methods
    # ----------------------------


    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _reset_constraints(self):
        raise NotImplementedError()

    def _env_setup(self):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
