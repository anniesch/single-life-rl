import numpy as np
from gym import utils
from . import mujoco_env
from PIL import Image

# Cheetah with short torso, can learn to flip upright
# Has resets every self.max_steps steps
class HalfCheetahEnvShort(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.max_steps = 500
        self.cur_steps = 0
        mujoco_env.MujocoEnv.__init__(self, "half_cheetah_short.xml", 5)
        utils.EzPickle.__init__(self)
        
    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) #/ self.dt
        reward = reward_ctrl + reward_run / self.dt
        rotation = self.sim.data.qpos[2]
        reward=(reward_run/self.dt)+reward_ctrl - np.abs(self.sim.data.qpos[2] - 0)
        done = self.is_successful(ob)
        if not done and self.cur_steps >= self.max_steps:
            done = True
        self.cur_steps += 1
    
        return ob, reward, done, dict(reward_run=reward_run, total_reward=reward)

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
                [-1]
            ]
        ).reshape(-1)

    def reset_model(self):
        self.cur_steps = 0
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        
    def save_im(self, im, name):
        img = Image.fromarray(im.astype(np.uint8)) 
        img.save(name)
        
    def is_successful(self, obs=None):
        x_pos2 =self.get_body_com('bfoot')[0]
        return x_pos2 > 300