import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from networks import RandomShiftsAug, Encoder, SACActor, Critic
    

class SACAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau,
                 reward_scale_factor, use_tb, from_vision, env_name):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.lr = lr
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.reward_scale_factor = reward_scale_factor
        self.use_tb = use_tb
        self.from_vision = from_vision
        self.env_name = env_name
        self.log_std_bounds = [-20, 2]
        self.init_temperature = 1.0

        model_repr_dim = -1
        # models
        if self.from_vision:
            self.encoder = Encoder(obs_shape).to(device)
            model_repr_dim = self.encoder.repr_dim

        self.actor = SACActor(model_repr_dim, action_shape, feature_dim,
                        hidden_dim, from_vision, self.log_std_bounds, env_name).to(device)
        
        self.critic = Critic(model_repr_dim, action_shape, feature_dim,
                             hidden_dim, from_vision).to(device)
        self.critic_target = Critic(model_repr_dim, action_shape,
                                    feature_dim, hidden_dim, from_vision).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.tensor(0).to(device)

        # optimizers
        if self.from_vision:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
           # data augmentation
            self.aug = RandomShiftsAug(pad=4)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()
    
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self, training=True):
        self.training = training
        if self.from_vision:
            self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
    
    def load_frozen_critic(self):
         # for when RL pretraining
        self.frozen_critic = Critic(-1, self.action_shape,
                                    self.feature_dim, self.hidden_dim, self.from_vision).to(self.device)
        self.frozen_critic.load_state_dict(self.critic.state_dict())

    def act(self, obs, step, eval_mode, var=None):
        obs = torch.as_tensor(obs, device=self.device)
        if self.from_vision:
            obs = self.encoder(obs.unsqueeze(0))

        dist = self.actor(obs, var=var)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            
        return action.cpu().numpy()

    def update_critic(self, obs, action, reward, discount, next_obs, step, not_done=None):
        metrics = dict()

        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_V -= self.alpha.detach() * log_prob
            target_Q = self.reward_scale_factor * reward + \
                            (discount * target_V * not_done.unsqueeze(1))

        Q1, Q2 = self.critic(obs, action)
        critic_loss = 1.0 * (F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q))

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        if self.from_vision:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 20.0)
        self.critic_opt.step()
        if self.from_vision:
            self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q + (self.alpha.detach() * log_prob)
        actor_loss = actor_loss.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                    (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['alpha_value'] = self.alpha
            
        return metrics
    
    def transition_tuple(self, replay_iter):
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, step_type, next_step_type, idxs, q_vals = utils.to_torch(batch, self.device)

        return (obs, action, reward, discount, next_obs, step_type, next_step_type)

    def update(self, trans_tuple, step):
        metrics = dict()

        obs, action, reward, discount, next_obs, step_type, next_step_type = trans_tuple
        
        obs = obs.float()
        next_obs = next_obs.float()

        not_done = next_step_type.clone()
        not_done[not_done < 2] = 1
        not_done[not_done == 2] = 0

        # augment
        if self.from_vision:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
            # encode
            obs = self.encoder(obs)
            with torch.no_grad():
                next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step, not_done))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)
            
        return metrics

    
class Discriminator(SACAgent):
    def __init__(self, *agent_args, env_name='tabletop_manipulation', discrim_type='state', discrim_hidden_size=128, discrim_lr=3e-4, mixup=False, q_weights=False, num_discrims=1, discrim_eps=1e-10, **agent_kwargs):
        
        super(Discriminator, self).__init__(**agent_kwargs, env_name=env_name)
        self.discrim_hidden_size = discrim_hidden_size
        self.discrim_lr = discrim_lr
        self.discrim_eps  = discrim_eps
        self.discrim_type = discrim_type
        self.mixup = mixup
        self.q_weights = q_weights
        self.num_discrims = num_discrims
        self.env_name = env_name
        if self.discrim_type == 'state':
            input_shape = self.obs_shape[0]
        elif self.discrim_type == 's-a':
            input_shape = self.obs_shape[0]+self.action_shape[0]
        if self.env_name == 'cheetah': input_shape -= 1
        self.discriminators = []
        self.discrim_opts = []
        for d in range(self.num_discrims):
            self.discriminators.append(nn.Sequential(nn.Linear(input_shape, discrim_hidden_size),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(discrim_hidden_size, 1)).to(self.device))
            self.discrim_opts.append(torch.optim.Adam(self.discriminators[d].parameters(), lr=discrim_lr))

    def update_discriminator(self, pos_replay_iter, neg_replay_iter, disc_ind, val_function=None, current_val=0, current_obs=None, min_q=0, max_q=1, baseline=0):
        self.discriminator = self.discriminators[disc_ind]
        self.discrim_opt = self.discrim_opts[disc_ind]
        if self.from_vision:
            print("update_discrim does not support vision")
            exit()

        metrics = dict()
        batch_pos = next(pos_replay_iter[disc_ind])
        obs_pos, act_pos, _, _, next_obs_pos, _, _, pos_idxs, pos_q = utils.to_torch(batch_pos, self.device)
        batch_neg = next(neg_replay_iter)
        obs_neg, act_neg, _, _, next_obs_neg, _, _, neg_idxs, neg_q = utils.to_torch(batch_neg, self.device)
        orig_obs_neg = obs_neg
        if self.env_name == 'cheetah': 
            obs_pos = obs_pos[:, :-1]
            obs_neg = obs_neg[:, :-1]
        
        if self.discrim_type == 'state':
            obs_pos = obs_pos
            obs_neg = obs_neg 
        elif self.discrim_type == 's-a':
            obs_pos = torch.cat((obs_pos, act_pos), axis=-1)
            obs_neg = torch.cat((obs_neg, act_neg), axis=-1)
            
        num_pos = obs_pos.shape[0]
        num_neg = obs_neg.shape[0]
        labels = torch.cat((torch.ones(num_pos, 1), torch.zeros(num_neg, 1)), 0).to(self.device)

        if self.mixup:
            alpha = 1.0
            beta_dist = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
            l = beta_dist.sample([num_pos + num_neg])
            mixup_coef = torch.reshape(l, (num_pos + num_neg, 1)).to(self.device)
            disc_inputs = torch.cat((obs_pos, obs_neg), 0)
            ridxs = torch.randperm(num_pos + num_neg)
            perm_labels = labels[ridxs]
            perm_disc_inputs = disc_inputs[ridxs]

            images = disc_inputs * mixup_coef + perm_disc_inputs * (1 - mixup_coef)
            labels = labels * mixup_coef + perm_labels * (1 - mixup_coef)
        else:
            images = torch.cat((obs_pos, obs_neg), 0)

        m = nn.Sigmoid()
        images = images.float()
        if not self.q_weights:
            loss = torch.nn.BCELoss()
            discrim_loss = loss(m(self.discriminator(images)), labels)
        else:
            loss = torch.nn.BCELoss(reduction='none')
            discrim_loss = loss(m(self.discriminator(images)), labels)
            # Calculate baseline
            if baseline == 0:
                obs_b = torch.tensor(current_obs.observation).cuda()
                action_b = torch.tensor(current_obs.action).cuda()
                b, _ = val_function(obs_b.float(), action_b.float())
                b = (b - min_q) / (max_q - min_q)
            else:
                b = torch.tensor(baseline).cuda()
            pos_q = torch.exp(pos_q - b)
            neg_q = torch.ones((num_neg, 1)).cuda() # no negative weighting
            if len(pos_q.shape) < len(neg_q.shape):
                pos_q = pos_q.unsqueeze(1)
            all_q = torch.cat((pos_q, neg_q), axis=0)
            if self.mixup: 
                all_q = all_q * mixup_coef + all_q[ridxs] * (1 - mixup_coef)
            discrim_loss = all_q * discrim_loss
            discrim_loss = discrim_loss.mean()

        self.discrim_opt.zero_grad(set_to_none=True)
        discrim_loss.backward()
        self.discrim_opt.step()
            
        # For logging
        images = torch.cat((obs_pos, obs_neg), 0).float()
        labels = torch.cat((torch.ones(num_pos, 1), torch.zeros(num_neg, 1)), 0).to(self.device)
        out = m(self.discriminator(images))
        prec1 = float(((out > 0.5) == labels[:]).sum()) / out.shape[0]
        
        if self.use_tb:
            metrics['discriminator_loss'] = discrim_loss.item()
            metrics['discriminator_acc'] = prec1
            if self.q_weights:
                metrics['discriminator_loss'] = discrim_loss.item()
                metrics['discriminator_acc'] = prec1

        return metrics
    
    def update_discriminators(self, pos_replay_iter, neg_replay_iter, val_function=None, current_val=0, current_obs=None, min_q=0, max_q=1, baseline=10):
        for ind, discriminator in enumerate(self.discriminators):
            metrics = self.update_discriminator(pos_replay_iter, neg_replay_iter, ind, val_function, current_val, current_obs, min_q, max_q, baseline)
        return metrics

    def compute_reward(self, obs, act, next_obs, reward=0, q_vals=None):
        obs = obs.float()
        next_obs = next_obs.float()
        act = act.float()
        if self.env_name == 'cheetah': 
            if len(obs.shape) > 1:
                obs = obs[:, :-1]
            else:
                obs = obs[:-1]
        if self.discrim_type == 'state':
            obs = obs
        elif self.discrim_type == 's-a':
            obs = torch.cat((obs, act), axis=-1)
        
        actual_reward = 0.
        for i, discriminator in enumerate(self.discriminators):
            actual_reward += -torch.log(1 - torch.sigmoid(discriminator(obs)) + self.discrim_eps) * (1/len(self.discriminators))
            
        disc_s = torch.sigmoid(self.discriminators[0](obs))
        if self.env_name == 'cheetah' or self.env_name == 'kitchen': actual_reward += reward
        
        return actual_reward, disc_s
        
    def transition_tuple(self, replay_iter, cur_time_step=None, old_time_step=None):
        if cur_time_step:
            pt_time_step = cur_time_step 
            next_obs, action, reward, discount = pt_time_step.observation, pt_time_step.action, pt_time_step.reward, pt_time_step.discount
            pt_time_step = old_time_step
            obs, action, reward, discount = pt_time_step.observation, pt_time_step.action, pt_time_step.reward, pt_time_step.discount
            obs = torch.as_tensor(obs, device=self.device)
            next_obs = torch.as_tensor(next_obs, device=self.device)
            action = torch.as_tensor(action, device=self.device)
            actual_reward, disc_s = self.compute_reward(obs, action, next_obs, reward, q_vals=torch.ones(1).cuda())
            return (obs, action, (actual_reward.detach(), disc_s.detach())), reward
        else:
            batch = next(replay_iter)
            obs, action, reward, discount, next_obs, step_type, next_step_type, idxs, q_vals = utils.to_torch(batch, self.device)
        
        obs = obs.float()
        next_obs = next_obs.float()
        new_r, _ = self.compute_reward(obs, action, next_obs, reward, q_vals)

        return (obs, action, new_r.detach(), discount, next_obs, step_type, next_step_type), reward
