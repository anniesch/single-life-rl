import os
import pickle
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from backend.timestep import ExtendedTimeStep

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 12})

class SimpleReplayBuffer:
    
    def __init__(self, data_specs, max_size, batch_size=None, replay_dir=None, discount=0.99,
                 filter_transitions=True, with_replacement=True, time_step=1, q_weights=False, rl_pretraining=False):
        self._data_specs = data_specs
        self._max_size = max_size # assume max_size >= total transitions
        self._batch_size = batch_size
        self.q_weights = q_weights
        self.rl_pretraining = rl_pretraining
        self._discount = discount
        self._replay_dir = replay_dir
        
        if not os.path.exists(self._replay_dir):
            os.makedirs(self._replay_dir)
        self._replay_buffer = {}
        self._filter_transitions = filter_transitions
        self._with_replacement = with_replacement
        self.time_step = time_step
        for spec in self._data_specs:
            self._replay_buffer[spec.name] = np.empty((max_size, *spec.shape), dtype=spec.dtype)
        self._replay_buffer['step_type'] = np.empty((max_size, 1), dtype=np.int32)
        self._replay_buffer['Q_vals'] = np.empty((max_size), dtype=np.float32)
        self._replay_buffer['traj_index'] = np.empty((max_size), dtype=np.int32) # index trajectory for demo data
        self._num_transitions = 0
        self._index = 0

    def __len__(self):
        return self._num_transitions
    
    def add_offline_data(self, demos, default_action, env_name=None):
        obs = demos['observations']
        acts = demos['actions']
        rew = demos['rewards']
        term = demos['terminals']
        next_o = demos['next_observations']
        demo_lengths = np.where(rew == 1.0)[0]

        for idx in range(obs.shape[0]):
            traj_index = 1
            for d in range(len(demo_lengths) - 1):
                if idx >= demo_lengths[d] and idx <= demo_lengths[d+1]:
                    traj_index = (idx - demo_lengths[d])/(demo_lengths[d+1] - demo_lengths[d])
            if env_name == 'pointmass':
                if demos['step_type'][idx] == 0:
                    time_step = ExtendedTimeStep(observation=obs[idx], step_type=0, action=default_action, reward=0.0, discount=1.0)
                time_step = ExtendedTimeStep(observation=obs[idx], step_type=demos['step_type'][idx], action=acts[idx], reward=rew[idx], discount=1.0)
                self.add(time_step, traj_index)
            else:
                if idx == 0 or term[idx - 1][0]:
                    time_step = ExtendedTimeStep(observation=obs[idx], step_type=0, action=default_action, reward=0.0, discount=1.0)
                    self.add(time_step, traj_index)

                if term[idx][0]:
                    time_step = ExtendedTimeStep(observation=next_o[idx], step_type=2, action=acts[idx], reward=rew[idx][0], discount=1.0)
                else:
                    time_step = ExtendedTimeStep(observation=next_o[idx], step_type=1, action=acts[idx], reward=rew[idx][0], discount=1.0)
                self.add(time_step, traj_index)
                
    def add(self, time_step, traj_index=None, q_val=None):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if spec.name == 'discount':
                value = np.expand_dims(time_step.discount * self._discount, 0).astype('float32')
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            np.copyto(self._replay_buffer[spec.name][self._index], value)

        np.copyto(self._replay_buffer['step_type'][self._index], time_step.step_type)
        
        if traj_index:
            self._replay_buffer['traj_index'][self._index] = traj_index 
        if q_val:
            self._replay_buffer['Q_vals'][self._index] = q_val 
        if self._num_transitions < self._max_size:
            self._num_transitions += 1
        if self._index == self._max_size - 1:
            self._index = 0
        else:
            self._index += 1
        
    def label_q_vals(self, critic, normalize=False):
        Q_vals = []
        for ind in range(self._num_transitions):
            obs, action = self._replay_buffer['observation'][ind], self._replay_buffer['action'][ind]
            with torch.no_grad():
                Q1, Q2 = critic(torch.FloatTensor(obs).cuda(), torch.FloatTensor(action).cuda())
            Q_vals.append(Q1.detach().item())
        self._replay_buffer['Q_vals'][:self._num_transitions] = np.array(Q_vals)
        if normalize:
            self._replay_buffer['Q_vals'][:self._num_transitions] = np.array(Q_vals).astype(np.float32) / (np.max(Q_vals) - np.min(Q_vals))
            self._replay_buffer['Q_vals'][:self._num_transitions] -= np.min(self._replay_buffer['Q_vals'][:self._num_transitions])
            self._replay_buffer['Q_vals'][:self._num_transitions] = np.clip(self._replay_buffer['Q_vals'][:self._num_transitions], 0, 1)
        return np.min(Q_vals), np.max(Q_vals)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self, batch_size=None, filter_transitions=None, with_replacement=None):
        batch_size = self._batch_size if batch_size is None else batch_size
        filter_transitions = self._filter_transitions if filter_transitions is None else filter_transitions
        with_replacement = self._with_replacement if with_replacement is None else with_replacement

        if with_replacement:
            idxs = np.random.randint(self.time_step, len(self), size=batch_size)
        else:
            # do not use np.random.choice, it gets much slower as the size increases
            idxs = np.array(random.sample(time_step, range(len(self)), batch_size), dtype=np.int64)

        if filter_transitions:
            filtered_idxs = []
            for idx in idxs:
                if self._replay_buffer['step_type'][idx]:
                    filtered_idxs.append(idx)
            idxs = np.array(filtered_idxs, dtype=np.int64)
            
        time_step = self.time_step
        if self.time_step == 0:
            time_step = np.random.randint(1, 20)
            
        if self.q_weights:
            if self.rl_pretraining:
                q_vals = self._replay_buffer['Q_vals'][idxs]
            else:
                q_vals = self._replay_buffer['traj_index'][idxs][:, np.newaxis]
        else:
            q_vals = np.ones(self._replay_buffer['reward'][idxs].shape)

        return (self._replay_buffer['observation'][idxs - time_step],
                self._replay_buffer['action'][idxs],
                self._replay_buffer['reward'][idxs],
                self._replay_buffer['discount'][idxs],
                self._replay_buffer['observation'][idxs],
                self._replay_buffer['step_type'][idxs - time_step].squeeze(1),
                self._replay_buffer['step_type'][idxs].squeeze(1),
                idxs,
               q_vals,)

    def save_buffer(self):
        import os
        if not os.path.exists(self._replay_dir):
            os.makedirs(self._replay_dir)
        with open(self._replay_dir / 'episodic_replay_buffer.buf', 'wb') as f:
            pickle.dump(self._replay_buffer, f)
        np.save(self._replay_dir / 'num_transitions.npy', self._num_transitions)
        np.save(self._replay_dir / 'index.npy', self._index)

    def load_buffer(self, replay_buffer_dir, num_transitions=None, env_name=None):
        if num_transitions:
            self._num_transitions = num_transitions
        else:
            self._num_transitions = np.load(f'{replay_buffer_dir}/num_transitions.npy').tolist()
        self._index = np.load(f'{replay_buffer_dir}/index.npy').tolist()
        replay_buffer = pickle.load(open(f'{replay_buffer_dir}/episodic_replay_buffer.buf', 'rb'))
        for key in replay_buffer.keys():
            self._replay_buffer[key][:self._num_transitions] = replay_buffer[key][-self._num_transitions:]
        Q_vals = self._replay_buffer['Q_vals'][:self._num_transitions]
        self._replay_buffer['Q_vals'][:self._num_transitions] = np.array(Q_vals).astype(np.float32) / (np.max(Q_vals) - np.min(Q_vals))
        self._replay_buffer['Q_vals'][:self._num_transitions] -= np.min(self._replay_buffer['Q_vals'][:self._num_transitions])
        self._replay_buffer['Q_vals'][:self._num_transitions] = np.clip(self._replay_buffer['Q_vals'][:self._num_transitions], 0, 1)
        return np.min(replay_buffer['Q_vals'][:self._num_transitions]), np.max(replay_buffer['Q_vals'][:self._num_transitions])
       
            
    def load(self):
        try:
            self._replay_buffer = pickle.load(open(self._replay_dir / 'episodic_replay_buffer.buf'), 'rb')
            self._num_transitions = np.save(self._replay_dir / 'num_transitions.npy').tolist()
        except:
            print('no replay buffer to be restored')

