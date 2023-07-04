from unstable_baselines.common.util import second_to_time_str
from unstable_baselines.common.trainer import BaseTrainer
import numpy as np
from tqdm import tqdm
from time import time
import cv2
import os
from tqdm import trange
import torch

class SACTrainer(BaseTrainer):
    def __init__(self, 
            seed,
            agent, 
            train_env, 
            eval_env, 
            buffer,  
            batch_size,
            max_env_steps,
            start_timestep,
            random_policy_timestep, 
            load_path,
            agent_save_frequency,
            agent_save_path,
            return_save_path,
            reset_frequency,
            utd,
            **kwargs):
        super(SACTrainer, self).__init__(agent, train_env, eval_env, **kwargs)
        self.seed = seed
        self.agent = agent
        self.buffer = buffer
        #hyperparameters
        self.batch_size = batch_size
        self.max_env_steps = max_env_steps
        self.start_timestep = start_timestep
        self.random_policy_timestep = random_policy_timestep
        self.agent_save_frequency=agent_save_frequency
        self.agent_save_path=agent_save_path
        self.return_save_path = return_save_path
        self.reset_frequency = reset_frequency
        self.utd = utd
        
        if load_path != "":
            self.load_snapshot(load_path)
            torch.save(self.agent.policy_network, "policy_network.pt")
            torch.save(self.agent.q1_network, "q1_network.pt")
            torch.save(self.agent.q2_network, "q2_network.pt")
            exit(0)
    
    def save_agent(self, timestamp):
        save_dir = self.agent_save_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_save_path = save_dir + "ite_{}_{}.pt".format(timestamp,self.seed)
        #print('debug!')
        # import pdb
        # pdb.set_trace()
        #model_save_path = os.path.join(save_dir, "ite_{}.pt".format(timestamp))
        torch.save(self.agent.state_dict(), model_save_path)

    def train(self):
        train_traj_returns = [0]
        train_traj_lengths = [0]
        tot_env_steps = 0
        traj_return = 0
        traj_length = 0
        done = False
        obs = self.train_env.reset()
        for env_step in trange(self.max_env_steps): # if system is windows, add ascii=True to tqdm parameters to avoid powershell bugs
            self.pre_iter()
            log_infos = {}

            if tot_env_steps < self.random_policy_timestep:
                action = self.train_env.action_space.sample()
            else:
                action = self.agent.select_action(obs)['action']

            next_obs, reward, done, info = self.train_env.step(action)
            traj_length += 1
            traj_return += reward
            if traj_length >= self.max_trajectory_length:
                done = False
            self.buffer.add_transition(obs, action, next_obs, reward, done)
            obs = next_obs
            if done or traj_length >= self.max_trajectory_length:
                obs = self.train_env.reset()
                train_traj_returns.append(traj_return)
                train_traj_lengths.append(traj_length)
                traj_length = 0
                traj_return = 0
            log_infos['performance/train_return'] = train_traj_returns[-1]
            log_infos['performance/train_length'] = train_traj_lengths[-1]
            tot_env_steps += 1
            if tot_env_steps < self.start_timestep:
                continue
            
            for _ in range(self.utd):
                data_batch = self.buffer.sample(self.batch_size)
                train_agent_log_infos = self.agent.update(data_batch)
    
            log_infos.update(train_agent_log_infos)

            self.post_step(tot_env_steps, self.return_save_path)
            self.post_iter(log_infos, tot_env_steps)
            
            if self.agent_save_frequency > 0 and tot_env_steps % self.agent_save_frequency == 0 : # save agent
                
                self.save_agent(timestamp=tot_env_steps)
            
            if self.reset_frequency > 0 and tot_env_steps % self.reset_frequency == 0: # reset agent
                self.agent.q1_network.reset_net()
                self.agent.q2_network.reset_net()
                self.agent.target_q1_network.reset_net()
                self.agent.target_q2_network.reset_net()
                self.agent.policy_network.reset_net()
                
                



