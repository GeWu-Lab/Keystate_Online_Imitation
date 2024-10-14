from collections import deque
from typing import Any, NamedTuple

import gym

from gym import Wrapper, spaces
from gym.wrappers import FrameStack
import gymnasium as gym
import dm_env
import numpy as np
from dm_env import StepType, specs, TimeStep
from dm_control.utils import rewards
import pdb

import cv2

class RGBArrayAsObservationWrapper(dm_env.Environment):
	"""
	Use env.render(rgb_array) as observation
	rather than the observation environment provides

	From: https://github.com/hill-a/stable-baselines/issues/915
	"""
	def __init__(self, env, width=128, height=128):
		self._env = env
		self._width = width
		self._height = height
		self._env.reset()
		self.observation_space  = spaces.Box(low=0, high=255, shape=(3, width, height), dtype=np.float32)
		self.action_space = spaces.Box(np.ones(7) * -1., np.ones(7) * 1.)
		
		# Action spec
		wrapped_action_spec = self.action_space
		if not hasattr(wrapped_action_spec, 'minimum'):
			wrapped_action_spec.minimum = -np.ones(wrapped_action_spec.shape)
		if not hasattr(wrapped_action_spec, 'maximum'):
			wrapped_action_spec.maximum = np.ones(wrapped_action_spec.shape)
		self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
										np.float32,
										wrapped_action_spec.minimum,
										wrapped_action_spec.maximum,
										'action')
		#Observation spec
		self._obs_spec = {}
		self._obs_spec['pixels'] = specs.BoundedArray(shape=self.observation_space.shape,
													  dtype=np.uint8,
													  minimum=0,
													  maximum=255,
													  name='observation')
		self.img = None

	def reset(self, **kwargs):
		obs = {}
		data = self._env.reset(**kwargs)
		# print("data is:", data[0]['observation'])
		img = data['agentview_image'].astype(np.uint8)
		obs['pixels'] = np.flip(img, axis=0) 
		# obs['pixels'] = obs['pixels'].astype(np.uint8)
		obs['goal_achieved'] = False
		obs['state'] = data['robot0_joint_pos']

		self.img = obs['pixels']
		return obs

	def step(self, action):
		eval_obs, reward, done, _  = self._env.step(action)
		# print("the terminate is:", terminate) 
		# print("the truncated is:", truncated)
		obs = {}
		# print("the observation is:", observation)
		# print("the truncated is:", truncated)
		
		# print("the observation is:", observation)
		img = eval_obs['agentview_image'].astype(np.uint8)
		obs['pixels'] = np.flip(img, axis=0)
		obs["state"] = eval_obs['robot0_joint_pos']
		obs['goal_achieved'] = done
		self.img = obs['pixels']
		return obs, reward, done

	def observation_spec(self):
		return self._obs_spec

	def action_spec(self):
		return self._action_spec

	def render(self,  width=256, height=256):
		return self.img

	def __getattr__(self, name):
		return getattr(self._env, name)

class ExtendedTimeStep(NamedTuple):
	step_type: Any
	reward: Any
	discount: Any
	observation: Any
	action: Any
	state: Any

	def first(self):
		return self.step_type == StepType.FIRST

	def mid(self):
		return self.step_type == StepType.MID

	def last(self):
		return self.step_type == StepType.LAST

	def __getitem__(self, attr):
		return getattr(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
	def __init__(self, env, num_repeats):
		self._env = env
		self._num_repeats = num_repeats
		
	def step(self, action):
		reward = 0.0
		discount = 1.0
		for i in range(self._num_repeats):
			time_step = self._env.step(action)
			reward += (time_step.reward or 0.0) * discount
			discount *= time_step.discount
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

class FrameStackWrapper(dm_env.Environment):
	def __init__(self, env, num_frames):
		self._env = env
		self._num_frames = num_frames
		self._frames = deque([], maxlen=num_frames)

		wrapped_obs_spec = env.observation_spec()['pixels']

		pixels_shape = wrapped_obs_spec.shape
		if len(pixels_shape) == 4:
			pixels_shape = pixels_shape[1:]
		# self._obs_spec = {}
		# self._obs_spec['pixels'] = specs.BoundedArray(shape=np.concatenate(
		# 	[[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
		# 									dtype=np.uint8,
		# 									minimum=0,
		# 									maximum=255,
		# 									name='observation')

	def _transform_observation(self, time_step):
		assert len(self._frames) == self._num_frames
		obs = {}
		obs['pixels'] = np.concatenate(list(self._frames), axis=0)
		obs['goal_achieved'] = time_step.observation['goal_achieved']
		obs['state'] = time_step.observation['state']
		return time_step._replace(observation=obs)

	def _extract_pixels(self, time_step):
		pixels = time_step.observation['pixels']
		# remove batch dim
		if len(pixels.shape) == 4:
			pixels = pixels[0]
		return pixels.transpose(2, 0, 1).copy()

	def reset(self):
		time_step = self._env.reset()
		pixels = self._extract_pixels(time_step)
		for _ in range(self._num_frames):
			self._frames.append(pixels)
		return self._transform_observation(time_step)

	def step(self, action):
		time_step = self._env.step(action)
		pixels = self._extract_pixels(time_step)
		self._frames.append(pixels)
		return self._transform_observation(time_step)

	def observation_spec(self):
		return self._obs_spec

	def action_spec(self):
		return self._env.action_spec()

	def __getattr__(self, name):
		return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
	def __init__(self, env, dtype):
		self._env = env
		self._discount = 1.0

		# Action spec
		wrapped_action_spec = env.action_space
		if not hasattr(wrapped_action_spec, 'minimum'):
			wrapped_action_spec.minimum = -np.ones(wrapped_action_spec.shape)
		if not hasattr(wrapped_action_spec, 'maximum'):
			wrapped_action_spec.maximum = np.ones(wrapped_action_spec.shape)
		self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
										np.float32,
										wrapped_action_spec.minimum,
										wrapped_action_spec.maximum,
										'action')
		#Observation spec
		self._obs_spec = {}
		self._obs_spec['pixels'] = specs.BoundedArray(shape=env.observation_space.shape,
										dtype=np.uint8,
										minimum=0,
										maximum=255,
										name='observation')

	def step(self, action):
		action = action.astype(self._env.action_space.dtype)
		# Make time step for action space
		# observation, reward, done, info = self._env.step(action)
		observation, reward, done = self._env.step(action)
		# print("the info is:", info)
		reward = reward + 1
		step_type = StepType.LAST if done else StepType.MID
		return TimeStep(
					step_type=step_type,
					reward=reward,
					discount=self._discount,
					observation=observation
				)

	def observation_spec(self):
		return self._obs_spec

	def action_spec(self):
		return self._action_spec

	def reset(self):
		obs = self._env.reset()
		return TimeStep(
					step_type=StepType.FIRST,
					reward=0,
					discount=self._discount,
					observation=obs
				)

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
		return ExtendedTimeStep(observation=time_step.observation,
								step_type=time_step.step_type,
								state = None,
								action=action,
								reward=time_step.reward or 0.0,
								discount=time_step.discount or 1.0)

	def _replace(self, time_step, step_type = None, observation=None, action=None, reward=None, discount=None,state=None):
		if observation is None:
			observation = time_step.observation
		if action is None:
			action = time_step.action
		if reward is None:
			reward = time_step.reward
		if discount is None:
			discount = time_step.discount
		if state is None:
			state = time_step.state
		if step_type is None:
			step_type = time_step.step_type
		return ExtendedTimeStep(observation=observation,
								step_type=step_type,
								action=action,
								state=state,
								reward=reward,
								discount=discount)


	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._env.action_spec()

	def __getattr__(self, name):
		return getattr(self._env, name)


from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os


def make(name, action_repeat, frame_stack ):
	benchmark_dict = benchmark.get_benchmark_dict()
	task_suite_name = "libero_goal" # can also choose libero_spatial, libero_object, etc.
	task_suite = benchmark_dict[task_suite_name]()

	task_names = task_suite.get_task_names()

	print("the task names are", task_names)
	task_id = 0
 
	for id, task_name in enumerate(task_names):
		if name in task_name:
			task_id = id
   
	task = task_suite.get_task(task_id)
	task_name = task.name
	task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

	camera_heights = 128
	camera_widths = 128
 
	env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": camera_heights,
        "camera_widths": camera_widths,
        "has_renderer": True,
        "has_offscreen_renderer": True,
        "control_freq": 20
    }
	env = OffScreenRenderEnv(**env_args)

	# if tasks_to_complete == "sdoor_open-v3":
	# 	tasks_to_complete = ["slide cabinet"]
	
	# env = gym.make(name, max_episode_steps = max_episode_steps, render_mode=render_mode, tasks_to_complete=tasks_to_complete)
	# env.seed(seed)
	
	# add wrappers
 
 
	env = RGBArrayAsObservationWrapper(env, width=camera_widths, height=camera_heights)
	env = ActionDTypeWrapper(env, np.float32)
	env = ActionRepeatWrapper(env, action_repeat)
	env = FrameStackWrapper(env, frame_stack)
	env = ExtendedTimeStepWrapper(env)
	return env


if __name__ == "__main__":
    env = make("none", 1, 1)
    env.reset()
    for _ in range(200):
        timestep = env.step(env.action_space.sample())
        obs = timestep.observation['pixels']
        print("the obs is:", obs.shape)
        obs = obs.transpose(1,2,0)
        # obs = np.flip(obs, 0)
        cv2.imshow("img", obs)
        cv2.waitKey(1)
  
