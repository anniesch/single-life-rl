"API to load tabletop and kitchen environments."

import os
import numpy as np
import pickle

from envs.wrappers import persistent_state_wrapper
from envs.wrappers import lifelong_wrapper
from envs import tabletop_manipulation

# for every environment, add an entry for the configuration of the environment
# make a default configuration for environment, the user can change the parameters by passing it to the constructor.

# number of initial states being provided to the user
# for deterministic initial state distributions, it should be 1
# for stochastic initial state distributions, sample the distribution randomly and save those samples for consistency
deployment_eval_config = {
  'tabletop_manipulation': {
    'num_initial_state_samples': 1,
    'num_goals': 4,
    'train_horizon': int(2e10),
    'eval_horizon': 200,
  },
  'kitchen': {
    'num_initial_state_samples': 1,
    'train_horizon': int(2e10),
    'eval_horizon': 400,
    'task': 'all_pairs',
  },
}

class EARLEnvs(object):
  def __init__(self,
               # parameters that need to be set for every environment
               env_name,
               reward_type='sparse',
               reset_train_env_at_goal=False,
               setup_as_lifelong_learning=False,
               train_resets=False,
               # parameters that have default values in the config
               **kwargs):
    self._env_name = env_name
    self._reward_type = reward_type
    self._reset_train_env_at_goal = reset_train_env_at_goal
    self._setup_as_lifelong_learning = setup_as_lifelong_learning
    self._kwargs = kwargs
    self.train_resets = train_resets

    # resolve to default parameters if not provided by the user
    self._train_horizon = kwargs.get('train_horizon', deployment_eval_config[env_name]['train_horizon'])
    self._eval_horizon = kwargs.get('eval_horizon', deployment_eval_config[env_name]['eval_horizon'])
    self._num_initial_state_samples = kwargs.get('num_initial_state_samples', deployment_eval_config[env_name]['num_initial_state_samples'])

    self._train_env = self.get_train_env()
    self._eval_env = self.get_eval_env()

  def get_train_env(self, lifelong=False):
    if self._env_name == 'tabletop_manipulation':
      from envs import tabletop_manipulation
      train_env = tabletop_manipulation.TabletopManipulation(task_list='rc_r-rc_k-rc_g-rc_b',
                                                             reward_type=self._reward_type,
                                                             reset_at_goal=self._reset_train_env_at_goal,
                                                             ns_init=True,
                                                            train_resets=self.train_resets)
    elif self._env_name == 'kitchen':
      from envs import kitchen
      kitchen_task = self._kwargs.get('kitchen_task', deployment_eval_config[self._env_name]['task'])  
      train_env = kitchen.Kitchen(task=kitchen_task, reward_type=self._reward_type, single=False) # Compound tasks (both microwave + kitchen), not single

    train_env = persistent_state_wrapper.PersistentStateWrapper(train_env, episode_horizon=self._train_horizon)
    return train_env

  def get_eval_env(self):
    if self._env_name == 'tabletop_manipulation':
      from envs import tabletop_manipulation
      eval_env = tabletop_manipulation.TabletopManipulation(task_list='rc_r-rc_k-rc_g-rc_b',
                                                            reward_type=self._reward_type,
                                                            ns_init=False,
                                                           train_resets=self.train_resets)
    elif self._env_name == 'kitchen':
      from envs import kitchen
      kitchen_task = self._kwargs.get('kitchen_task', deployment_eval_config[self._env_name]['task'])  
      eval_env = kitchen.Kitchen(task=kitchen_task, reward_type=self._reward_type, single=True)
    return persistent_state_wrapper.PersistentStateWrapper(eval_env, episode_horizon=self._eval_horizon)

  def has_demos(self):
    if self._env_name in ['tabletop_manipulation']: 
        return True
    else:
        return False

  def get_envs(self):
    if not self._setup_as_lifelong_learning:
      return self._train_env, self._eval_env
    else:
      return self._train_env

  def get_initial_states(self, num_samples=None):
    '''
    Always returns initial state of the shape N x state_dim
    '''
    if num_samples is None:
      num_samples = self._num_initial_state_samples

    # TODO: potentially load initial states from disk
    if self._env_name == 'tabletop_manipulation':
      return tabletop_manipulation.initial_states

    elif self._env_name == 'kitchen':
      kitchen_task = self._kwargs.get('kitchen_task', deployment_eval_config[self._env_name]['task'])  
      env = kitchen.Kitchen(task=kitchen_task, reward_type=self._reward_type)
      return env.get_init_states()

    else:
      # make a new copy of environment to ensure that related parameters do not get affected by collection of reset states
      cur_env = self.get_eval_env()
      reset_states = []
      while len(reset_states) < self._num_initial_state_samples:
        reset_states.append(cur_env.reset())
      return np.stack(reset_states)

  def get_goal_states(self):
    if self._env_name == 'tabletop_manipulation':
      from envs import tabletop_manipulation
      return tabletop_manipulation.goal_states

    if self._env_name == 'kitchen':
      from envs import kitchen
      return kitchen.goal_states

  def get_demonstrations(self, envname=None):
    # use the current file to locate the demonstrations
    base_path = os.path.abspath(__file__)
    demo_dir = os.path.join(os.path.dirname(base_path), 'demonstrations')
    name = envname if envname else self._env_name
    try:
      forward_demos = pickle.load(open(os.path.join(demo_dir, name, 'forward/demo_data.pkl'), 'rb'))
      reverse_demos = pickle.load(open(os.path.join(demo_dir, name, 'reverse/demo_data.pkl'), 'rb'))
      return forward_demos, reverse_demos
    except:
      print('please download the demonstrations corresponding to ', self._env_name)
