import torch
import torch.nn as nn
import torch.nn.functional as F


class ACNetCentralized(nn.Module):
    def __init__(self, size_world, n_agents):
        super().__init__()
        self.size_world = size_world
        self.n_agents = n_agents
        self.dim_map = size_world[0] * size_world[1]
        self.conv = nn.Conv2d(2, 1, 3)
        # self.pool = nn.MaxPool2d(3)
        h_out = size_world[0] - 2
        w_out = size_world[1] - 2
        self.pos_embed = nn.Linear(2 * n_agents, 256)
        self.linear1 = nn.Linear(256+h_out*w_out, 1024)
        # self.pos_linear1 = nn.Linear(3, 128)
        # self.pos_linear2 = nn.Linear(128, 256)
        self.actor = nn.Sequential(nn.Linear(1024, 256),
                                   nn.Linear(256, self.dim_map * n_agents))
        self.critic = nn.Sequential(nn.Linear(1024, 256),
                                   nn.Linear(256, 1))

    def forward(self, heatmap, cov_lvl, pos):
        maps = torch.cat([heatmap, cov_lvl], dim=1)  # (batch, channel, height, width)
        batch_size = maps.size(0)
        maps = self.conv(maps)
        # maps = self.pool(maps)
        x = self.pos_embed(pos)
        x = F.leaky_relu(x) # (batch, n_agents, 256)
        x = torch.cat([x.view(batch_size, -1), maps.view(batch_size, -1)], dim=-1)
        x = self.linear1(x)
        x = F.leaky_relu(x)
        # x = torch.cat([x, pos], dim=-1)
        logits = self.actor(x).view(batch_size, self.n_agents, self.size_world[0]*self.size_world[1])
        value = torch.squeeze(self.critic(x))
        # probs = F.softmax(logits, dim=1)
        return logits, value
    
    def embed_observe(self, nothing):
        return
    

class ACNetDistributed(nn.Module):
    def __init__(self, size_world, n_agents):
        super().__init__()
        self.size_world = size_world
        self.n_agents = n_agents
        self.dim_map = size_world[0] * size_world[1]
        self.conv = nn.Conv2d(2, 1, 3)
        # self.pool = nn.MaxPool2d(3)
        h_out = size_world[0] - 2
        w_out = size_world[1] - 2
        self.pos_embed = nn.Linear(3, 128) # Only taking agent's own position
        self.linear1 = nn.Linear(128+h_out*w_out, 1024)
        # self.pos_linear1 = nn.Linear(3, 128)
        # self.pos_linear2 = nn.Linear(128, 256)
        self.actor = nn.Sequential(nn.Linear(1024, 256),
                                   nn.Linear(256, self.dim_map))
        self.critic = nn.Sequential(nn.Linear(1024, 256),
                                   nn.Linear(256, 1))

    def forward(self, heatmap, cov_lvl, pos):
        maps = torch.cat([heatmap, cov_lvl], dim=1)  # (batch, channel, height, width)
        batch_size = maps.size(0)
        maps = self.conv(maps)
        # maps = self.pool(maps)
        x = self.pos_embed(pos)
        x = F.leaky_relu(x) # (batch, n_agents, 256)
        x = torch.cat([x.view(batch_size, -1), maps.view(batch_size, -1)], dim=-1)
        x = self.linear1(x)
        x = F.leaky_relu(x)
        # x = torch.cat([x, pos], dim=-1)
        logits = self.actor(x).view(batch_size, self.size_world[0]*self.size_world[1])
        value = torch.squeeze(self.critic(x))
        # probs = F.softmax(logits, dim=1)
        return logits, value
    
    def embed_observe(self, nothing):
        return