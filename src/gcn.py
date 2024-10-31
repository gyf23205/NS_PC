import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvNet(nn.Module):
    def __init__(self, n_embed_channel, size_kernal, dim_map, n_rel, n_head):
        super().__init__()
        self.n_rel = n_rel
        self.dim_map = dim_map
        self.dim_embed = ((dim_map - size_kernal) + 1)**2 # dim_out of CNN = (dim_in - size_kernal + padding) / stride + 1
        self.encoder = nn.Conv2d(1, n_embed_channel, size_kernal)
        self.relations = nn.ModuleList([nn.MultiheadAttention(self.dim_embed, n_head) for _ in range(n_rel)])
        self.ffs = nn.ModuleList([nn.Linear(self.dim_embed, self.dim_embed) for _ in range(n_rel)])
        self.out = nn.Linear(self.dim_embed, dim_map**2)

    def forward(self, x, neighbors):
        '''
        x.shape = (dim_map, dim_map)
        neighbors.shape = (number of neighbors of current agent at this time step, dim_embed)
        Using unbatch data because the number of neighbors changes with agent and time.
        '''
        x = self.encoder(x)
        x = x.view(1, self.dim_embed)
        for i in range(self.n_rel):
            x = self.relations[i](x, neighbors, neighbors)
            x = F.relu(self.ffs[i](x))
            # Compared with transformer, no normalization here.
        prob = F.softmax(self.out(x))
        return prob.view(-1, self.dim_map, self.dim_map)
    
    def embed_obs(self, x):
        with torch.no_grad():
             x = self.encoder(x)
             x = x.view(-1, 1, self.dim_embed)
        return x
