# import utils
import cv2
import torch
import pickle
import numpy as np
from torchvision import transforms

import utils

class KeystateDistribution():
    def __init__(self, device):
        self.device = device

    def __call__(self, skey, exp_demo):
        '''
        exp_demos: numpy, from enumerate(exp_buffer)
        '''
        # extract keystates
        seq_len = exp_demo.shape[0]

        semantic_keystate = skey

        semantic_keystate = [0] + [idx-1 for idx in semantic_keystate]

        # employ optical flow to get motion key state
        # rgb to gray
        transGray = np.array([0.299,  0.587, 0.114])
        gray_frames = [exp_demo[i].transpose(1, 2, 0)@transGray for i in range(seq_len)]
        optical_flow = np.array([np.linalg.norm(self._optical_flow(gray_frames[i], gray_frames[i+1])) for i in range(seq_len-1)])
        
        motion_keystate = []
        for i in range(len(semantic_keystate)-1):
            cur_shot = optical_flow[semantic_keystate[i]: semantic_keystate[i+1]]
            if len(cur_shot) > 0:
                motion_keystate.append(semantic_keystate[i]+np.argmax(cur_shot)+1)

        semantic_keystate = semantic_keystate[1: ]

        # index to distribution
        distribution = self._generate_gaussian(seq_len, semantic_keystate, motion_keystate)

        return distribution, semantic_keystate


    def _optical_flow(self, prev_gray, next_gray):
            return cv2.calcOpticalFlowFarneback(prev=prev_gray, next=next_gray, flow=None, pyr_scale=0, levels=0,
                                                winsize=5,
                                                iterations=5, poly_n=7, poly_sigma=1.2,
                                                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    
    def _generate_gaussian(self, seq_len, semantic_keystate, motion_keystate=None):
        # distribution = np.zeros(seq_len)
        distribution = np.ones(seq_len) / seq_len
        for key in semantic_keystate[:-1]:
            distribution += 0.15*utils.gaussian_distribution(seq_len, key, 10)
        distribution += 0.35*utils.gaussian_distribution(seq_len, semantic_keystate[-1], 10)

        if motion_keystate is not None:
            for sub_key in motion_keystate:
                distribution += 0.05*utils.gaussian_distribution(seq_len, sub_key, 25)

        distribution = distribution / distribution.sum()

        return distribution
