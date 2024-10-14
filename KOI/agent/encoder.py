import torch.nn as nn
import utils
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
import torch


class MultiModalEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        
        if obs_shape[1] == 84:
            self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                        nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                        nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                        nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                        nn.ReLU())
            self.repr_dim = 32 * 35 * 35
        
        elif obs_shape[1] == 128:
            self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 5, stride=2),
                                        nn.ReLU(), nn.Conv2d(32, 32, 5, stride=2),
                                        nn.ReLU(), nn.Conv2d(32, 64, 3, stride=1),
                                        nn.ReLU(), nn.Conv2d(64, 32, 3, stride=1),
                                        nn.ReLU())
            repr_dim = 20000

            self.repr_dim = 512
            self.linear = nn.Sequential(
                nn.Linear(repr_dim, self.repr_dim),
                nn.ReLU(),
                nn.LayerNorm(self.repr_dim),
            )
            
            self.state_encoder = nn.Sequential(
                nn.Linear(7, self.repr_dim),
                nn.ReLU(),
                nn.Linear(self.repr_dim, self.repr_dim),
                nn.ReLU(),
                nn.Linear(self.repr_dim, self.repr_dim)
            )

        else:
            self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 5, stride=2),
                                        nn.ReLU(), nn.Conv2d(32, 32, 5, stride=2),
                                        nn.ReLU(), nn.Conv2d(32, 64, 5, stride=2),
                                        nn.ReLU(), nn.Conv2d(64, 32, 3, stride=1),
                                        nn.ReLU())
            repr_dim = 23328
            self.repr_dim = 512
            self.linear = nn.Sequential(
                nn.Linear(repr_dim, self.repr_dim),
                nn.ReLU(),
                nn.LayerNorm(self.repr_dim),
            )
            
            self.state_encoder = nn.Sequential(
                nn.Linear(7, self.repr_dim),
                nn.ReLU(),
                nn.Linear(self.repr_dim, self.repr_dim),
                nn.ReLU(),
                nn.Linear(self.repr_dim, self.repr_dim)
            )
            
        self.obs_shape = obs_shape



        self.apply(utils.weight_init)


    def get_image_feature(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.reshape(h.shape[0], -1)
        if self.repr_dim == 512:
            h = self.linear(h)
        return h


    def forward(self, obs, state):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        if self.repr_dim == 512:
            h = self.linear(h)
        state_feature = self.state_encoder(state)  
        h = torch.cat((h, state_feature), dim=1)
        return h


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        
        if obs_shape[1] == 84:
            self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                        nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                        nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                        nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                        nn.ReLU())
            self.repr_dim = 32 * 35 * 35
        elif obs_shape[1] == 128:
            self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 5, stride=2),
                                        nn.ReLU(), nn.Conv2d(32, 32, 5, stride=2),
                                        nn.ReLU(), nn.Conv2d(32, 64, 3, stride=1),
                                        nn.ReLU(), nn.Conv2d(64, 32, 3, stride=1),
                                        nn.ReLU())
            self.repr_dim = 20000      
        self.obs_shape = obs_shape
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h
