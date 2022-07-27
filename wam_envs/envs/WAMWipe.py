import os
from gym import utils as gym_utils
import numpy as np
from wam_envs import wam_env
from wam_envs import utils
from wam_envs.utils import WAM_fwd_kinematics



MODEL_XML_PATH = os.path.join('wam', 'wam_7dof_wam_bhand_frictionless.xml')

class WAMWipeEnv(wam_env.WAMEnv, gym_utils.EzPickle):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, reward_type='sparse', model_path=MODEL_XML_PATH, robot_mode='7dof', n_substeps=10, n_actions=7
    ):
        """Initializes a new WAM environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        
        self.obj_range = None
        self.target_range = None
        self.reward_type = reward_type

        # task specific constraints
        self.collided = False
        self.lifted = False
        self.tilted = False
        self.braking = False

        initial_qlist = [-2.02362697e-01,  4.33858512e-01,  4.71274239e-02,  2.17864115e+00,
        -3.92220801e-02,  5.29857799e-01, -1.25754495e-01]

        super(WAMWipeEnv, self).__init__(
            model_path=model_path, initial_qlist=initial_qlist, n_substeps=n_substeps, robot_mode=robot_mode, n_actions=n_actions)

        gym_utils.EzPickle.__init__(self)


    # GoalEnv methods
    # ----------------------------
            

    def get_SCL_map(self):
        np_cond_inp = self._get_cond_inp()
        out = self.tfsess.run(self.SCL_map, feed_dict={self.cond_inp: np_cond_inp})
        #print(out)
        return out

    # WAMEnv methods
    # ----------------------------
    def _check_constraints(self):
        q = utils.get_joint_angles(self.sim)

        # CoRL sim
        target_height = 0.2+0.15-0.346+0.06
        T = utils.get_kinematics_mat(q,self.n_joints)
        height = T[2,-1]
        ee_zaxis = T[:3,2]
        angle = np.arccos(-1*ee_zaxis[2])

        if height < (target_height-0.015):
            self.collided = True
        if height > (target_height+0.015):
            self.lifted = True
        if np.abs(angle) > np.pi/16:
            self.tilted = True

    def _step_callback(self):
        self._check_constraints()
        return


    def _render_callback(self):
        # Visualize target.
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal + np.array([0, 0, 0.346*3])#-0.06]) # make note of what each of these numbers represent

        # Add marker to end-effector
        q = utils.get_joint_angles(self.sim)
        ee_pos = WAM_fwd_kinematics(q, self.n_joints)
        ee_site_id = self.sim.model.site_name2id('ee')
        self.sim.model.site_pos[ee_site_id] = ee_pos + np.array([0, 0, 0.346*3])

        self.sim.forward()

    def _reset_constraints(self):
        self.collided = False
        self.lifted = False
        self.tilted = False
        self.braking = False
    

    def reset_goal(self):
        g = self._sample_goal()
        self.goal = g

    def _sample_goal(self):
        goal = np.array([np.random.uniform(0.3,0.7), np.random.uniform(-0.25,0.25), 0.2+0.15-0.346+0.06])
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = utils.goal_distance(achieved_goal, desired_goal)
        
        success = True # innocent until proven guilty
        if d >= self.distance_threshold:
            success = False
        
        if self.collided or self.lifted or self.tilted:
            success = False
            
        return success

    def _env_setup(self):
        initial_qpos = self.initial_qpos # defined in wam_env super class
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
            
        self.sim.forward()


    def render(self, mode='human', width=500, height=500):
        return super(WAMWipeEnv, self).render(mode, width, height)
