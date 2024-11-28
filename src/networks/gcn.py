import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvNet(nn.Module):
    def __init__(self, n_embed_channel, size_kernal, dim_observe, size_world, n_rel, n_head=8):
        super().__init__()
        self.n_rel = n_rel
        self.dim_map = size_world[0] * size_world[1]
        self.dim_embed = n_embed_channel * ((dim_observe - size_kernal) + 1)**2 # dim_out of CNN = (dim_in - size_kernal + padding) / stride + 1
        self.encoder = nn.Conv2d(1, n_embed_channel, size_kernal)
        self.relations = nn.ModuleList([nn.MultiheadAttention(self.dim_embed, n_head) for _ in range(n_rel)])
        self.ffs = nn.ModuleList([nn.Linear(self.dim_embed, self.dim_embed) for _ in range(n_rel)])
        self.out = nn.Linear(self.dim_embed, self.dim_map)


    def forward(self, x, neighbors):
        '''
        x.shape = (1, dim_observe, dim_observe)
        neighbors.shape = (number of neighbors of current agent at this time step + 1(self), dim_embed)
        Using unbatch data because the number of neighbors changes with agent and time.
        '''
        x = self.encoder(x)
        x = x.view(1, self.dim_embed)
        for i in range(self.n_rel):
            x, _ = self.relations[i](x, neighbors, neighbors)
            x = F.relu(self.ffs[i](x))
            # Compared with transformer, no normalization here.
            out = torch.squeeze(self.out(x))
        return F.softmax(out, dim=0)
    
    def embed_observe(self, x):
        with torch.no_grad():
             x = self.encoder(x)
             x = x.view(1, self.dim_embed)
        return x
