import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "osmesa"

from pathlib import Path
import env_loader
import hydra
import numpy as np
import torch
import random
import utils
from PIL import Image
import time

from dm_env import specs
from logger import Logger
from simple_replay_buffer import SimpleReplayBuffer
from video import TrainVideoRecorder
from agents import SACAgent, Discriminator
from backend.timestep import ExtendedTimeStep
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 12})

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg, env_name):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.feature_dim = obs_spec.shape[0]
    return SACAgent(obs_shape=cfg.obs_shape,
                action_shape=cfg.action_shape,
                device=cfg.device,
                lr=cfg.lr,
                feature_dim=cfg.feature_dim,
                hidden_dim=cfg.hidden_dim,
                critic_target_tau=cfg.critic_target_tau, 
                reward_scale_factor=cfg.reward_scale_factor,
                use_tb=cfg.use_tb,
                from_vision=cfg.from_vision,
                env_name=env_name)
    
def make_discriminator(obs_spec, action_spec, cfg, env_name, discrim_type, mixup, q_weights, num_discrims):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.feature_dim = obs_spec.shape[0]
    return Discriminator(
            discrim_hidden_size=cfg.discrim_hidden_size,
            obs_shape=cfg.obs_shape,
            action_shape=cfg.action_shape,
            device=cfg.device,
            lr=cfg.lr,
            feature_dim=cfg.feature_dim,
            hidden_dim=cfg.hidden_dim,
            critic_target_tau=cfg.critic_target_tau, 
            reward_scale_factor=cfg.reward_scale_factor,
            use_tb=cfg.use_tb,
            from_vision=cfg.from_vision,
            env_name=env_name,
            discrim_type=discrim_type,
            mixup=mixup,
            q_weights=q_weights,
            num_discrims=num_discrims,)

class Workspace:
    def __init__(self, cfg, orig_dir):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.orig_dir = orig_dir

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent,
                               self.cfg.env_name)
        
        if self.cfg.use_discrim:
            self.discriminator = make_discriminator(self.train_env.observation_spec(),
                                                  self.train_env.action_spec(),
                                                  self.cfg.agent,
                                                  self.cfg.env_name,
                                                     discrim_type=self.cfg.discrim_type,
                                                     mixup=self.cfg.mixup,
                                                     q_weights=self.cfg.q_weights,
                                                      num_discrims=self.cfg.num_discrims)
        
        self.timer = utils.Timer()
        self.timer._start_time = time.time()
        self._global_step = -self.cfg.num_pretraining_steps
        print("Global step", self._global_step)
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        self.train_env, self.eval_env, self.reset_states, self.goal_states, self.forward_demos = env_loader.make(self.cfg.env_name, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.resets, orig_dir=self.orig_dir)
        if self.cfg.resets:
            _, self.train_env, self.reset_states, self.goal_states, self.forward_demos = env_loader.make(self.cfg.env_name, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.resets, orig_dir=self.orig_dir)
            
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage_f = SimpleReplayBuffer(data_specs,
                                                       self.cfg.replay_buffer_size,
                                                       self.cfg.batch_size,
                                                       self.work_dir / 'forward_buffer',
                                                  self.cfg.discount)
    
        self.online_buffer = SimpleReplayBuffer(data_specs,
                                                       self.cfg.replay_buffer_size,
                                                       self.cfg.batch_size,
                                                       self.work_dir / 'forward_buffer',
                                               self.cfg.discount,
                                               time_step=self.cfg.time_step)
        self.prior_buffers = []
        for _ in range(self.cfg.num_discrims):
            self.prior_buffers.append(SimpleReplayBuffer(data_specs,
                                                       self.cfg.prior_buffer_size, 
                                                       self.cfg.batch_size,
                                                       self.work_dir / 'forward_buffer',
                                                         self.cfg.discount,
                                                         time_step=self.cfg.time_step,
                                                       q_weights=self.cfg.q_weights,
                                                       rl_pretraining=self.cfg.rl_pretraining))
        
        self._forward_iter = None 

        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None, self.cfg.env_name)

        # recording metrics for EARL
        np.save(self.work_dir / 'eval_interval.npy', self.cfg.eval_every_frames)
        try:
            self.deployed_policy_eval = np.load(self.work_dir / 'deployed_eval.npy').tolist()
        except:
            self.deployed_policy_eval = []

    @property
    def forward_iter(self):
        if self._forward_iter is None:
            self._forward_iter = iter(self.replay_storage_f)
        return self._forward_iter
    
    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat
    
    def save_im(self, im, name):
        img = Image.fromarray(im.astype(np.uint8)) 
        img.save(name)

    def save_gif(self, ims, name):
        imageio.mimsave(name, ims, fps=len(ims)/100)
    
    
    def update_progress(self, time_step, distances, upside_down, num_upside_down, x_progress, y_progress, agent_x, agent_y, initial_back=0):
        if self.cfg.env_name == 'cheetah':
            backfoot, bfoot, ffoot = self.train_env.compute_progress_cheetah(time_step.observation)
            if self.global_step > 1 and bfoot > ffoot + .5:
                num_upside_down += 1
                upside_down.append(self.global_step)
            total_distance = backfoot - initial_back
            distances.append(total_distance)
        if self.cfg.env_name == 'pointmass':
            loc = time_step.observation[:2]
            x_progress.append(loc[0])
            y_progress.append(loc[1])
            if self.global_step % 1000 == 0:
                print("location", loc[0], loc[1])
        if self.cfg.env_name == 'tabletop_manipulation':
            loc = time_step.observation[2:4]
            goal = time_step.observation[-4:-2]
            x_progress.append(loc[0])
            y_progress.append(loc[1])
            agent_x.append(time_step.observation[0])
            agent_y.append(time_step.observation[1])
        return distances, upside_down, num_upside_down, x_progress, y_progress, agent_x, agent_y
    
    
    def plot_progress(self, obs, distances, upside_down, num_upside_down, x_progress, y_progress, agent_x, agent_y, online_rews, online_qvals, color='timestep'):
        c_vals = []
        c_vals = np.arange(len(x_progress)) # timestep
        if self.cfg.env_name == 'cheetah':
            plt.plot(distances)
            plt.xlabel("Steps")
            plt.ylabel("X distance")
            plt.savefig(f'{self.work_dir}/online_distances.png')
            plt.close()
        if self.cfg.env_name == 'pointmass':
            plt.scatter(x_progress, y_progress, c=c_vals, s=1, cmap='Greens')
            plt.colorbar()
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.savefig(f'{self.work_dir}/xy_coords_{color}.png')
            plt.close()
        if self.cfg.env_name == 'tabletop_manipulation':
            goal = obs[-4:-2]
            plt.scatter(x_progress, y_progress, c=c_vals, label='x progress', s=1, cmap='Greens')
            plt.colorbar()
            plt.scatter(goal[0], goal[1], label='goal', s=10, color='red')
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.savefig(f'{self.work_dir}/mug_xy_coords_{color}.png')
            plt.close()
        return c_vals
        

    def train(self, snapshot_dir=None):
        # predicates
        train_until_step = utils.Until(self.cfg.online_steps, 
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_init_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        if self.cfg.rl_pretraining:
            time_step = self.eval_env.reset()
            _, self.eval_env_pretraining, _, _, _ = env_loader.make(self.cfg.env_name, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.resets, orig_dir=self.orig_dir)
        else:
            time_step = self.train_env.reset()
        dummy_action = time_step.action

        if self.forward_demos and (not self.cfg.rl_pretraining or self.cfg.use_demos):
            self.replay_storage_f.add_offline_data(self.forward_demos, dummy_action, env_name=self.cfg.env_name)
            for buffer in self.prior_buffers:
                _ = buffer.add_offline_data(self.forward_demos, dummy_action, env_name=self.cfg.env_name)
        
        prior_iters = []
        for d in range(self.cfg.num_discrims):
            prior_iters.append(iter(self.prior_buffers[d])) 
        online_iter = iter(self.online_buffer)
        cur_agent = self.agent
        cur_buffer = self.replay_storage_f
        cur_iter = self.forward_iter

        if self.cfg.save_train_video:
            self.train_video_recorder.init(self.train_env)
    
        metrics = None
        episode_step, episode_reward = 0, 0
        distances = []
        num_stuck = 0
        past_timesteps = []
        online_rews = [] # all online rewards
        online_qvals = [] # all online qvals
        x_progress = []
        y_progress = []
        agent_x = []
        agent_y = []
        num_upside_down = 0
        upside_down = []
        prev_actions = []
        cur_reward = torch.tensor(0.0).cuda()
        initial_back = 0. # For cheetah
        while train_until_step(self.global_step):
            
            '''Start single episode'''
            if self.global_step == 0:
                print("Starting single episode")
                time_step = self.train_env.reset()
                cur_buffer.add(time_step)
                x_progress = []
                y_progress = []
                agent_x = []
                agent_y = []
                
                if self.cfg.rl_pretraining:
                    self.agent.load_frozen_critic()
                    # Load prior data into cur_buffer as well as self.prior_buffers[0]
                    min_q, max_q = cur_buffer.load_buffer(f'{snapshot_dir}/', self.cfg.prior_buffer_size, self.cfg.env_name)
                    if self.prior_buffers[0].__len__() == 0:
                        _, _ = self.prior_buffers[0].load_buffer(f'{snapshot_dir}/', self.cfg.prior_buffer_size)
                    
            '''Logging and eval'''
            criteria = self.global_step % 500 == 0 
            if self.cfg.resets and time_step.last():
                episode_step, episode_reward = 0, 0
                time_step = self.train_env.reset()
                cur_buffer.add(time_step)
            if criteria: 
                if self.global_step == 0:
                    episode_step, episode_reward = 0, 0
                self._global_episode += 1
                
                if self.global_step % 1000 == 0 and self.global_step > 0:
                    if self.cfg.save_train_video:
                        self.save_im(self.train_env.render(mode="rgb_array"), f'{self.work_dir}/train_video/train{self.global_frame}.png')
                        self.train_video_recorder.save(f'train{self.global_frame}.gif')
                
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('forward_buffer_size', len(self.replay_storage_f))
                        log('step', self.global_step)
                
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                
             ##############################################################################################  
            '''If online during single episode'''
            if self.global_step >= 0 or self.cfg.rl_pretraining:
                '''Sample action'''
                with torch.no_grad(), utils.eval_mode(cur_agent):
                        action = cur_agent.act(time_step.observation.astype("float32"),
                                   self.global_step,
                                   eval_mode=False)
                
                '''Take env step'''                           
                if self.cfg.rl_pretraining and self.global_step < 0:
                    time_step = self.eval_env.step(action)
                else:
                    time_step = self.train_env.step(action)
                orig_reward = time_step.reward
                
                '''Plot progress'''
                if self.cfg.env_name == 'cheetah':
                    if self.cfg.rl_pretraining:
                        _, eval_bfoot, eval_ffoot = self.eval_env.compute_progress_cheetah(time_step.observation)
                        if self.global_step == -self.cfg.num_pretraining_steps: initial_back = eval_bfoot 
                    elif self.global_step == 0: 
                        backfoot, bfoot, ffoot = self.train_env.compute_progress_cheetah(time_step.observation)
                        initial_back = backfoot
                distances, upside_down, num_upside_down, x_progress, y_progress, agent_x, agent_y = self.update_progress(time_step, distances, upside_down, num_upside_down, x_progress, y_progress, agent_x, agent_y, initial_back=initial_back)
                # Get rew and q val of new online state transition
                online_rews.append(cur_reward.detach().item())
                with torch.no_grad():
                    Q1, Q2 = self.agent.critic_target(torch.FloatTensor(time_step.observation).cuda(), torch.FloatTensor(time_step.action).cuda())
                online_qvals.append(Q1.detach().item())
                if self.global_step % 100 == 0:
                    self.plot_progress(time_step.observation, distances, upside_down, num_upside_down, x_progress, y_progress, agent_x, agent_y, online_rews, online_qvals, color='timestep')            
                    
                '''Terminate if successful'''
                success_criteria = orig_reward > 0.0 and self.global_step > 0 and not self.cfg.resets
                if self.cfg.env_name == 'cheetah': success_criteria = distances[-1] > 300
                if success_criteria or self.global_step == self.cfg.online_steps - 1:
                    
                    time_step = ExtendedTimeStep(observation=time_step.observation,
                                                 step_type=2,
                                                 action=action,
                                                 reward=time_step.reward,
                                                 discount=time_step.discount)
                    print("Completed task in steps", self.global_step, time_step)
                    self.save_im(self.train_env.render(mode="rgb_array"), f'{self.work_dir}/final_completed{self.global_step}.png')
                    with open(f"{self.work_dir}/total_steps.txt", 'w') as f:
                        f.write(str(self.global_step))
                    if self.cfg.save_train_video:
                        self.train_video_recorder.save(f'train{self.global_step}.mp4')
                    
                    # Plot final progress
                    self.plot_progress(time_step.observation, distances, upside_down, num_upside_down, x_progress, y_progress, agent_x, agent_y, online_rews, online_qvals, color='timestep')
                    exit()
                    
                episode_reward += orig_reward
                if self.cfg.biased_update and self.global_step % self.cfg.biased_update == 0:
                    # Use a biased TD update to control critic values
                    time_step = ExtendedTimeStep(observation=time_step.observation,
                                             step_type=2,
                                             action=action,
                                             reward=orig_reward,
                                             discount=time_step.discount)
                
                # Add to buffer
                if self.cfg.rl_pretraining and self.global_step < 0:
                    self.prior_buffers[0].add(time_step)
                else:
                    cur_buffer.add(time_step)
                    self.online_buffer.add(time_step)
                episode_step += 1
                
            if self.cfg.save_train_video and self.global_step < 50000:
                self.train_video_recorder.record(self.train_env)
                
            ##############################################################################################    
            '''Update the agent'''
            if self.cfg.use_discrim:
                if self.global_step % self.cfg.discriminator.train_interval == 0 and self.online_buffer.__len__() > self.cfg.discriminator.batch_size:
                    for k in range(self.cfg.discriminator.train_steps_per_iteration):
                        if self.cfg.rl_pretraining and self.cfg.q_weights:
                            metrics = self.discriminator.update_discriminators(prior_iters, online_iter, self.agent.frozen_critic, cur_reward, time_step, min_q, max_q, baseline=self.cfg.baseline)
                        else:
                            metrics = self.discriminator.update_discriminators(prior_iters, online_iter)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

            if not seed_until_step(self.global_step):
                if self.cfg.use_discrim and self.online_buffer.__len__() > self.cfg.discriminator.batch_size: # ensure enough steps so discriminator is trained
                    trans_tuple, original_reward = self.discriminator.transition_tuple(cur_iter)
                    metrics = cur_agent.update(trans_tuple, self.global_step)
                    metrics['original_reward'] = original_reward.mean()

                    # Log the reward of the latest step
                    if len(past_timesteps) > 10: # for logging
                        del past_timesteps[0]
                    past_timesteps.append(time_step)
                    old_time_step = past_timesteps[0]
                    latest_tuple, original_reward = self.discriminator.transition_tuple(cur_iter, cur_time_step=time_step, old_time_step=old_time_step)
                    _, _, latest_reward = latest_tuple
                    actual_reward, disc_s = latest_reward
                    metrics['latest_r'] = actual_reward
                    metrics['disc_s'] = disc_s
                    cur_reward = disc_s # Use latest discriminator score as baseline val
                else:
                    trans_tuple = cur_agent.transition_tuple(cur_iter)
                    metrics = cur_agent.update(trans_tuple, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
            
            self._global_step += 1    
           
              
    def save_snapshot(self, epoch=None):
        snapshot = self.work_dir / 'snapshot.pt'
        if epoch: snapshot = self.work_dir / f'snapshot{epoch}.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, dirname=None):
        if dirname: 
            payload = torch.load(dirname)
        else: 
            snapshot = self.work_dir / 'snapshot.pt'
            with snapshot.open('rb') as f:
                payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v
        self._global_step = -self.cfg.num_pretraining_steps


@hydra.main(config_path='./', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd
    orig_dir = hydra.utils.get_original_cwd()
    workspace = W(cfg, orig_dir)
    snapshot_dir = None
    if cfg.rl_pretraining and not cfg.save_buffer:
        snapshot= f'{orig_dir}/data/offline/{cfg.env_name}_snapshot.pt'
        print(f'resuming: {snapshot}')
        workspace.load_snapshot(snapshot)
        snapshot_dir = f'{orig_dir}/data/offline/{cfg.env_name}_buffer/'
    workspace.train(snapshot_dir)


if __name__ == '__main__':
    main()
    

    