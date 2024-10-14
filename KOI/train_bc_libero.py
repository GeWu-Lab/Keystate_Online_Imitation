#!/usr/bin/env python3

import warnings
import os

from pathlib import Path

import hydra
import numpy as np
import torch

import utils
from logger import Logger
from replay_buffer import make_expert_replay_loader
import pickle
import h5py


warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True
def make_agent(obs_shape, action_shape, cfg):
	cfg.obs_shape = obs_shape
	cfg.action_shape = action_shape
	return hydra.utils.instantiate(cfg)

class WorkspaceIL:
	def __init__(self, cfg):
		self.work_dir = Path.cwd()
		print(f'workspace: {self.work_dir}')

		self.cfg = cfg
		utils.set_seed_everywhere(cfg.seed)
		self.device = torch.device(cfg.device)
		self.setup()

		self.agent = make_agent(obs_shape=cfg.suite.obs_shape,action_shape=cfg.suite.action_shape, cfg = cfg.agent)

		self.cfg.suite.num_train_frames = self.cfg.suite.num_train_frames_bc
		self.cfg.suite.num_seed_frames = 0

		self.expert_replay_loader = make_expert_replay_loader(
			self.cfg.expert_dataset, self.cfg.batch_size // 2, self.cfg.num_demos, self.cfg.obs_type)
		self.expert_replay_iter = iter(self.expert_replay_loader)
			
		self.timer = utils.Timer()
		self._global_step = 0
		self._global_episode = 0
		
	def setup(self):
		# create logger
		self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)

		self._replay_iter = None
		self.expert_replay_iter = None

	@property
	def global_step(self):
		return self._global_step

	@property
	def global_episode(self):
		return self._global_episode

	@property
	def global_frame(self):
		return self.global_step * self.cfg.suite.action_repeat

	@property
	def replay_iter(self):
		return self._replay_iter

	def train_il(self):
		# predicates
		train_until_step = utils.Until(self.cfg.suite.num_train_frames,
									   self.cfg.suite.action_repeat)
		
		seed_until_step = utils.Until(self.cfg.suite.num_seed_frames,
									  self.cfg.suite.action_repeat)
		eval_every_step = utils.Every(self.cfg.suite.eval_every_frames,
									  self.cfg.suite.action_repeat)

		episode_step, episode_reward = 0, 0

		metrics = None

  
		while train_until_step(self.global_step):
			# try to evaluate
			if eval_every_step(self.global_step):
				self.logger.log('eval_total_time', self.timer.total_time(),
								self.global_frame)
				
				self.save_snapshot(episode_step)
    
			# try to update the agent
			if not seed_until_step(self.global_step):
				metrics = self.agent.update(self.replay_iter, self.expert_replay_iter, 
											self.global_step, self.cfg.bc_regularize)
				if self.global_step % 1000 == 0:
					print("the global step is:", metrics)
				self.logger.log_metrics(metrics, self.global_frame, ty='train')

			episode_step += 1
			self._global_step += 1

	def save_snapshot(self, step):
		snapshot = self.work_dir / f'{step}_snapshot.pt'
		keys_to_save = ['timer', '_global_step', '_global_episode']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		payload.update(self.agent.save_snapshot())
		with snapshot.open('wb') as f:
			torch.save(payload, f)

	def load_snapshot(self, snapshot):
		with snapshot.open('rb') as f:
			payload = torch.load(f)
		agent_payload = {}
		for k, v in payload.items():
			if k not in self.__dict__:
				agent_payload[k] = v
		self.agent.load_snapshot(agent_payload)
  
		

@hydra.main(config_path='cfgs', config_name='libero_config')
def main(cfg):
	# from train import WorkspaceIL as W
	root_dir = Path.cwd()
	workspace = WorkspaceIL(cfg)
	
	# Load weights
	if cfg.load_bc:
		snapshot = Path(cfg.bc_weight)
		if snapshot.exists():
			print(f'resuming bc: {snapshot}')
			workspace.load_snapshot(snapshot)

	workspace.train_il()


if __name__ == '__main__':
	main()
