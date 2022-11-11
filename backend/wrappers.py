import numpy as np
import gym
import dm_env
from dm_env import specs
from bsuite.utils.gym_wrapper import DMEnvFromGym, space2spec

from .timestep import ExtendedTimeStep


_ACTION_SPEC_MUST_BE_BOUNDED_ARRAY = (
    "`env.action_spec()` must return a single `BoundedArray`, got: {}.")
_MUST_BE_FINITE = "All values in `{name}` must be finite, got: {bounds}."
_MUST_BROADCAST = (
    "`{name}` must be broadcastable to shape {shape}, got: {bounds}.")


class DMEnvFromGymWrapper(DMEnvFromGym):
    def __init__(self, gym_env: gym.Env):
        self.gym_env = gym_env
        # Convert gym action and observation spaces to dm_env specs.
        self._observation_spec = space2spec(self.gym_env.observation_space,
                                            name='observation')
        self._action_spec = space2spec(self.gym_env.action_space, name='action')
        self._reset_next_step = True

    def is_successful(self, obs):
        return self.gym_env.is_successful(obs=obs)

    def compute_reward(self, obs):
        reward = self.gym_env.compute_reward(obs=obs)
        if isinstance(reward, list):
            reward = reward[0]
        return reward
    
    def compute_progress_cheetah(self, obs):
        reward = self.gym_env.get_body_com('bfoot')[0]
#         if isinstance(reward, list):
#             reward = reward[0]
        return reward, self.gym_env.get_body_com('bfoot')[0], self.gym_env.get_body_com('ffoot')[0]

    def render(self, mode="rgb_array"):
        return self.gym_env.render(mode=mode)

class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        return self._env.step(action)
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
#             import pdb; pdb.set_trace()
            reward += (time_step.reward or 0.0) * discount
            try:
                discount *= time_step.discount
            except:
                import pdb; pdb.set_trace()
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionScaleWrapper(dm_env.Environment):
  """Wraps a control environment to rescale actions to a specific range."""
  __slots__ = ("_action_spec", "_env", "_transform")

  def __init__(self, env, minimum, maximum):
    """Initializes a new action scale Wrapper.
    Args:
      env: Instance of `dm_env.Environment` to wrap. Its `action_spec` must
        consist of a single `BoundedArray` with all-finite bounds.
      minimum: Scalar or array-like specifying element-wise lower bounds
        (inclusive) for the `action_spec` of the wrapped environment. Must be
        finite and broadcastable to the shape of the `action_spec`.
      maximum: Scalar or array-like specifying element-wise upper bounds
        (inclusive) for the `action_spec` of the wrapped environment. Must be
        finite and broadcastable to the shape of the `action_spec`.
    Raises:
      ValueError: If `env.action_spec()` is not a single `BoundedArray`.
      ValueError: If `env.action_spec()` has non-finite bounds.
      ValueError: If `minimum` or `maximum` contain non-finite values.
      ValueError: If `minimum` or `maximum` are not broadcastable to
        `env.action_spec().shape`.
    """
    action_spec = env.action_spec()
    if not isinstance(action_spec, specs.BoundedArray):
      raise ValueError(_ACTION_SPEC_MUST_BE_BOUNDED_ARRAY.format(action_spec))

    minimum = np.array(minimum)
    maximum = np.array(maximum)
    shape = action_spec.shape
    orig_minimum = action_spec.minimum
    orig_maximum = action_spec.maximum
    orig_dtype = action_spec.dtype

    def validate(bounds, name):
      if not np.all(np.isfinite(bounds)):
        raise ValueError(_MUST_BE_FINITE.format(name=name, bounds=bounds))
      try:
        np.broadcast_to(bounds, shape)
      except ValueError:
        raise ValueError(_MUST_BROADCAST.format(
            name=name, bounds=bounds, shape=shape))

    validate(minimum, "minimum")
    validate(maximum, "maximum")
    validate(orig_minimum, "env.action_spec().minimum")
    validate(orig_maximum, "env.action_spec().maximum")

    scale = (orig_maximum - orig_minimum) / (maximum - minimum)

    def transform(action):
      new_action = orig_minimum + scale * (action - minimum)
#       print("newaction", new_action)
      return new_action.astype(orig_dtype, copy=False)

    dtype = np.result_type(minimum, maximum, orig_dtype)
    self._action_spec = action_spec.replace(
        minimum=minimum, maximum=maximum, dtype=dtype)
    self._env = env
    self._transform = transform

  def step(self, action):
    return self._env.step(self._transform(action))

  def reset(self):
    return self._env.reset()

  def observation_spec(self):
    return self._env.observation_spec()

  def action_spec(self):
    return self._action_spec

  def __getattr__(self, name):
    return getattr(self._env, name)
