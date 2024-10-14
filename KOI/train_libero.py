#!/usr/bin/env python3

# import torch.multiprocessing as mp
# if mp.get_start_method(allow_none=True) != "spawn":  
#     mp.set_start_method("spawn", force=True)

import warnings
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'glfw'
from pathlib import Path


import hydra
import numpy as np
import torch
from dm_env import specs

import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_expert_replay_loader, make_libero_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import pickle
from dm_env import StepType, specs, TimeStep
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

		self.agent = make_agent(obs_shape=(3,128,128),action_shape=(7,), cfg = cfg.agent)

		if repr(self.agent) == 'bc':
			self.cfg.suite.num_seed_frames = 0

		self.expert_replay_loader = make_expert_replay_loader(
			self.cfg.expert_dataset, self.cfg.batch_size // 2, self.cfg.num_demos, self.cfg.obs_type)
		self.expert_replay_iter = iter(self.expert_replay_loader)
			
		self.timer = utils.Timer()
		self._global_step = 0
		self._global_episode = 0


		expert_obs = []
		expert_action = []

		import h5py
		with h5py.File(self.cfg.expert_dataset, 'r') as f:
			data = f['data']
			# print("data keys:", data[0].keys())
			for i in range(self.cfg.num_demos):
				obs = np.array(data[f'demo_{i}']['obs']['agentview_rgb'])
				obs = np.flip(obs, axis=1)
				obs = np.transpose(obs, (0, 3, 1, 2)).copy()

				expert_obs.append(obs)
				expert_action.append(np.array(data[f'demo_{i}']['actions']))

    

			self.expert_demo = dict(observation=expert_obs, action=expert_action)

		
	def setup(self):
		# create logger
		self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
		# create envs
		self.train_env = hydra.utils.call(self.cfg.suite.task_make_fn)
		self.eval_env = hydra.utils.call(self.cfg.suite.task_make_fn)

		state_shape = specs.BoundedArray(shape=(7,),
													  dtype=np.float32,
													  minimum=-2 * np.pi,
													  maximum=2 * np.pi,
													  name='state')

		# create replay buffer
		data_specs = [
			self.train_env.observation_spec()[self.cfg.obs_type],
			self.train_env.action_spec(),
			state_shape,
			specs.Array((1, ), np.float32, 'reward'),
			specs.Array((1, ), np.float32, 'discount')
		]

		self.replay_storage = ReplayBufferStorage(data_specs,
												  self.work_dir / 'buffer')

		self.replay_loader = make_libero_replay_loader(
			self.work_dir / 'buffer', self.cfg.replay_buffer_size,
			self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
			self.cfg.suite.save_snapshot, self.cfg.nstep, self.cfg.suite.discount)

		self._replay_iter = None
		self.expert_replay_iter = None

		self.video_recorder = VideoRecorder(
			self.work_dir if self.cfg.save_video else None)
		self.train_video_recorder = TrainVideoRecorder(
			self.work_dir if self.cfg.save_train_video else None)

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
		if self._replay_iter is None:
			self._replay_iter = iter(self.replay_loader)
		return self._replay_iter

	def eval(self):
		step, episode, total_reward = 0, 0, 0
		eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)

		paths = []
		while eval_until_episode(episode):
			if self.cfg.suite.name == 'metaworld':
				path = []
			time_step = self.eval_env.reset()
			self.video_recorder.init(self.eval_env, enabled=(episode == 0))
			eval_time = 0
			while not time_step.last() and eval_time < 200:
				# print("the eval time is:", eval_time)
				eval_time += 1
				with torch.no_grad(), utils.eval_mode(self.agent):
					action = self.agent.act(time_step.observation,
											self.global_step,
											eval_mode=True)
				time_step = self.eval_env.step(action)
				if self.cfg.suite.name == 'metaworld':
					path.append(time_step.observation['goal_achieved'])
				self.video_recorder.record(self.eval_env)
				total_reward += time_step.reward
				step += 1

			episode += 1
			self.video_recorder.save(f'{self.global_frame}.mp4')
			paths.append(time_step.observation['goal_achieved'])
		
		with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
			log('episode_reward', total_reward / episode)
			log('episode_length', step * self.cfg.suite.action_repeat / episode)
			log('episode', self.global_episode)
			log('step', self.global_step)

			log("success_percentage", np.mean(paths))
			print("the success percentage is:", np.mean(paths))

	def train_il(self):
		# predicates
		train_until_step = utils.Until(self.cfg.suite.num_train_frames,
									   self.cfg.suite.action_repeat)
		
		seed_until_step = utils.Until(self.cfg.suite.num_seed_frames,
									  self.cfg.suite.action_repeat)
		eval_every_step = utils.Every(self.cfg.suite.eval_every_frames,
									  self.cfg.suite.action_repeat)

		episode_step, episode_reward = 0, 0

		time_steps = list()
		observations = list()
		actions = list()

		time_step = self.train_env.reset()
		
		time_steps.append(time_step)
		observations.append(time_step.observation[self.cfg.obs_type])
		actions.append(time_step.action)
		
		if repr(self.agent) == 'koi':
			if self.agent.auto_rew_scale:
				self.agent.sinkhorn_rew_scale = 1.  # Set after first episode

		self.train_video_recorder.init(time_step.observation[self.cfg.obs_type])
		metrics = None

		action_time = 0
		while train_until_step(self.global_step):
			# print("the seed is:", self.num_train_frames)
			# print("frame:", self.cfg.suite.num_train_frames)
			# print("the train step is:", self.global_step)
			# print("the action repeat is:", self.cfg.suite.action_repeat)
			# print("the util is:", train_until_step(self.global_step))
			if time_step.last() or action_time >= 200:
				action_time = 0
				# print("the timestep is:", time_step.action.shape)
				self._global_episode += 1
				if self._global_episode % 10 == 0:
					self.train_video_recorder.save(f'{self.global_frame}.mp4')
				# wait until all the metrics schema is populated
				observations = np.stack(observations, 0)
				actions = np.stack(actions, 0)
				# print("the observation shape is:", observations.shape)
				# print("the expert demo is:", self.expert_demo[0].shape)
				if repr(self.agent) == 'koi':
					# print("self.expert_demo shape is:", self.expert_demo)
					new_rewards = self.agent.ot_rewarder(
						observations, self.expert_demo, self.global_step)
					# print("the new reward is:", len(new_rewards))
					new_rewards_sum = np.sum(new_rewards)
				elif repr(self.agent) == 'dac':
					new_rewards = self.agent.dac_rewarder(observations, actions)
					new_rewards_sum = np.sum(new_rewards)
				elif repr(self.agent) == 'roboclip':
					new_rewards = self.agent.roboclip_rewarder(observations, self.expert_demo, self.global_step)

				
				if repr(self.agent) == 'koi':
					if self.agent.auto_rew_scale: 
						if self._global_episode == 1:
							self.agent.sinkhorn_rew_scale = self.agent.sinkhorn_rew_scale * self.agent.auto_rew_scale_factor / float(
								np.abs(new_rewards_sum))
							new_rewards = self.agent.ot_rewarder(
								observations, self.expert_demo, self.global_step)
							new_rewards_sum = np.sum(new_rewards)

				for i, elt in enumerate(time_steps):
					elt = elt._replace(state=time_steps[i].observation['state'].astype(np.float32))
					task_finish = time_steps[i].observation['goal_achieved']
					elt = elt._replace(
						observation=time_steps[i].observation[self.cfg.obs_type])
     
					if repr(self.agent) == 'koi' or repr(self.agent) == 'dac':
						elt = elt._replace(reward=new_rewards[i] + 5 * task_finish)
					elif repr(self.agent) == 'roboclip':
						elt = elt._replace(reward=new_rewards + 5 * task_finish) if elt.last() else elt._replace(reward=5*task_finish)
							# elt = elt._replace(reward=5*task_finish)
					if i == len(time_steps) - 1:
						elt = elt._replace(step_type=StepType.LAST)
					# print("i add the elt", i)
					self.replay_storage.add(elt)

				if metrics is not None:
					# log stats
					elapsed_time, total_time = self.timer.reset()
					episode_frame = episode_step * self.cfg.suite.action_repeat
					with self.logger.log_and_dump_ctx(self.global_frame,
													  ty='train') as log:
						# log('fps', episode_frame / elapsed_time)
						# log('total_time', total_time)
						log('episode_reward', episode_reward)
						# log('episode_length', episode_frame)
						# log('episode', self.global_episode)
						# log('buffer_size', len(self.replay_storage))
						# log('step', self.global_step)


				# reset env
				time_steps = list()
				observations = list()
				actions = list()

				time_step = self.train_env.reset()
				time_steps.append(time_step)
				observations.append(time_step.observation[self.cfg.obs_type])
				actions.append(time_step.action)
				self.train_video_recorder.init(time_step.observation[self.cfg.obs_type])
				# try to save snapshot
				if self.cfg.suite.save_snapshot:
					self.save_snapshot()
				episode_step = 0
				episode_reward = 0

			action_time += 1
			# try to evaluate
			if eval_every_step(self.global_step):
				self.logger.log('eval_total_time', self.timer.total_time(),
								self.global_frame)
				self.eval()
				
			# sample action
			with torch.no_grad(), utils.eval_mode(self.agent):
				action = self.agent.act(time_step.observation,
										self.global_step,
										eval_mode=False)
				# print("the action is:", action," the global step is:", self.global_step)
			# try to update the agent
			if not seed_until_step(self.global_step):
				metrics = self.agent.update(self.replay_iter, self.expert_replay_iter, 
											self.global_step, self.cfg.bc_regularize)
				self.logger.log_metrics(metrics, self.global_frame, ty='train')

			# take env step
			time_step = self.train_env.step(action)
			episode_reward += time_step.reward

			time_steps.append(time_step)
			observations.append(time_step.observation[self.cfg.obs_type])
			actions.append(time_step.action)

			self.train_video_recorder.record(time_step.observation[self.cfg.obs_type])
			episode_step += 1
			self._global_step += 1

		print("I finish train")

	def save_snapshot(self):
		snapshot = self.work_dir / 'snapshot.pt'
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
