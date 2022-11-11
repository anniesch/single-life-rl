import envs.earl_benchmarks as earl_benchmark
import envs.tabletop_manipulation as tabletop_manipulation
from envs.half_cheetah_short_hurdle import HalfCheetahEnvShortHurdle
from envs.half_cheetah_short import HalfCheetahEnvShort
from envs.pointmass import PointMassEnv
import numpy as np
import h5py

from backend.wrappers import (
    ActionRepeatWrapper,
    ActionDTypeWrapper,
    ExtendedTimeStepWrapper,
    ActionScaleWrapper,
    DMEnvFromGymWrapper,
)

def make(name, frame_stack, action_repeat, resets=False, orig_dir=None):
    
    if name == 'pointmass':
        eval_env = PointMassEnv(wiggly_weight=0., resets=True)
        train_env = PointMassEnv(wiggly_weight=1., resets=False)
        reset_states, goal_states, forward_demos = None, None, None
        import pickle
        with open(f'{orig_dir}/data/demos/pointmass/pointmass.pkl', 'rb') as f:
            forward_demos = pickle.load(f)
        forward_demos['observations'] = np.array(forward_demos['observations'][:300]) 
        forward_demos['actions'] = np.array(forward_demos['actions'][:300]).astype(np.float32) 
        forward_demos['rewards'] = np.array(forward_demos['rewards'][:300])
        forward_demos['terminals'] = np.array(forward_demos['terminals'][:300])[np.newaxis, :]
        forward_demos['next_observations'] = np.array(forward_demos['next_observations'][:300])
        
    elif name == 'cheetah':
        eval_env = HalfCheetahEnvShort()
        train_env = HalfCheetahEnvShortHurdle()
        goal_states = None
        reset_states = None
        observation_space = train_env.observation_space
        forward_demos = None
    
    else:
        if name == 'kitchen':
            env_loader = earl_benchmark.EARLEnvs(
            name,
            reward_type="dense",
            reset_train_env_at_goal=False,
            train_resets=resets,
            )
        else:
            env_loader = earl_benchmark.EARLEnvs(
                name,
                reward_type="sparse",
                reset_train_env_at_goal=False,
                train_resets=resets,
            )
        train_env, eval_env = env_loader.get_envs()
        reset_states = env_loader.get_initial_states()
        reset_state_shape = reset_states.shape[1:]
        goal_states = env_loader.get_goal_states()
        forward_demos = None

    # add wrappers
    minimum = -1.0
    maximum = +1.0
    train_env = DMEnvFromGymWrapper(train_env)
    train_env = ActionDTypeWrapper(train_env, np.float32)
    train_env = ActionRepeatWrapper(train_env, action_repeat)
    train_env = ActionScaleWrapper(train_env, minimum=minimum, maximum=maximum)
    train_env = ExtendedTimeStepWrapper(train_env) 

    eval_env = DMEnvFromGymWrapper(eval_env)
    eval_env = ActionDTypeWrapper(eval_env, np.float32)
    eval_env = ActionRepeatWrapper(eval_env, action_repeat)
    eval_env = ActionScaleWrapper(eval_env, minimum=minimum, maximum=maximum)
    eval_env = ExtendedTimeStepWrapper(eval_env)

    return train_env, eval_env, reset_states, goal_states, forward_demos