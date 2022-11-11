from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

from gym import utils
import numpy as np
from . import mujoco_env
from PIL import Image


# pylint: disable=missing-docstring
class PointMassEnv(mujoco_env.MujocoEnv, utils.EzPickle):

  def __init__(self,
               target=None,
               resets=False,
               wiggly_weight=0.,
               alt_xml=False,
               expose_velocity=True,
               expose_goal=True,
               use_simulator=False,
               model_path='point.xml'):
    self._sample_target = target
    self.goal = np.array([100.0, 0.0])
    self.max_steps = 1000
    self.cur_steps = 0
    self.resets = resets

    self._expose_velocity = expose_velocity
    self._expose_goal = expose_goal
    self._use_simulator = use_simulator
    self._wiggly_weight = abs(wiggly_weight)
    self._wiggle_direction = +1 if wiggly_weight > 0. else -1

    if self._use_simulator:
      mujoco_env.MujocoEnv.__init__(self, model_path, 5)
    else:
      mujoco_env.MujocoEnv.__init__(self, model_path, 1)
    utils.EzPickle.__init__(self)

  def step(self, action):
    action = np.clip(action, np.array([-1.] * 2), np.array([1.] * 2))
    
    if self._use_simulator:
      self.do_simulation(action, self.frame_skip)
    else:
      force = 0.2 * action[0]
      rot = 1.0 * action[1]
      qpos = self.sim.data.qpos.flat.copy()
      dx = action[0]
      dy = action[1] 
      if self._wiggly_weight > 0: # add wind
        dy += np.random.uniform(0.8, 0.9)
        dx -= 0.2
      qpos[0] = np.clip(qpos[0] + dx, -100, 100)
      qpos[1] = np.clip(qpos[1] + dy, -200, 200)
      qvel = self.sim.data.qvel.flat.copy()
      self.set_state(qpos, qvel)

    ob = self._get_obs()
    if False: #self.goal is not None:
      reward = -np.linalg.norm(self.sim.data.qpos.flat[:2] - self.goal)**2
    else:
      reward = float(self.is_successful(obs=ob))
    
    done = self.is_successful(ob)
    if not self.resets:
        done = False
    elif self.cur_steps >= self.max_steps:
        done = True
    
    self.cur_steps += 1
    return ob, reward, done, {}

  def _get_obs(self):
    new_obs = [self.sim.data.qpos.flat]
    if self._expose_velocity:
      new_obs += [self.sim.data.qvel.flat]
    if self._expose_goal and self.goal is not None:
      new_obs += [self.goal]
    new_obs = np.concatenate(new_obs)
    return new_obs

  def reset_model(self):
    self.cur_steps = 0
    if self._wiggly_weight > 0:
        print("Resetting wiggly weight env")
    qpos = self.init_qpos + self.np_random.uniform(low=-.2, high=.2, size=2)
    qvel = self.init_qvel + self.np_random.randn(self.sim.model.nv) * .01
    self.goal = np.array([100.0, 0.0])
    self.set_state(qpos, qvel)
    return self._get_obs()

  def set_qpos(self, state):
    qvel = np.copy(self.sim.data.qvel.flat)
    self.set_state(state, qvel)

  def viewer_setup(self):
    self.viewer.cam.distance = self.model.stat.extent * 0.5
    
  def compute_reward(self, obs):
    reward = float(self.is_successful(obs=obs))
    
  def is_successful(self, obs=None):
    if obs is None:
      obs = self._get_obs()
    return np.linalg.norm(obs[0:2] - obs[-2:]) <= 2