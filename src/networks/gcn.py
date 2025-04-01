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


class GCNPos(nn.Module):
    def __init__(self, n_embed_channel, size_kernal, dim_observe, size_world, n_rel, n_head=8):
        super().__init__()
        n_hidden = 64
        self.n_rel = n_rel
        self.dim_map = size_world[0] * size_world[1]
        self.dim_embed = n_embed_channel * ((dim_observe - size_kernal) + 1)**2 # dim_out of CNN = (dim_in - size_kernal + padding) / stride + 1
        self.encoder = nn.Conv2d(1, n_embed_channel, size_kernal, bias=False)
        self.pos_embed = nn.Linear(3, 16, bias=False)
        self.linear = nn.Linear(self.dim_embed + 16, n_hidden, bias=False) # 3 extra extries represent the position and orientaion of the agent. 512 comes from "Attention is all you need".
        self.relations = nn.ModuleList([nn.MultiheadAttention(n_hidden, n_head, bias=False) for _ in range(n_rel)])
        self.layernorms = nn.ModuleList([nn.LayerNorm(n_hidden) for _ in range(2*n_rel)])
        self.ffs1 = nn.ModuleList([nn.Linear(n_hidden, n_hidden, bias=False) for _ in range(n_rel)])
        self.ffs2 = nn.ModuleList([nn.Linear(n_hidden, n_hidden, bias=False) for _ in range(n_rel)])
        self.out = nn.Linear(n_hidden, self.dim_map, bias=False)
        # self.out = nn.Linear(512, 36, bias=False)


    def forward(self, x, neighbors, pos):
        '''
        x.shape = (1, dim_observe, dim_observe)
        pos.shape = (1, 3)
        neighbors.shape = (number of neighbors of current agent at this time step + 1(self), dim_embed)
        Using unbatch data because the number of neighbors changes with agent and time.
        '''
        # if x.shape == (1, 6, 6):
        #     pass
        # else:
        #     print(x.shape)
        x = self.encoder(x)
        x = x.view(1, self.dim_embed)
        pos = self.pos_embed(pos)
        x = torch.cat([x, pos], dim=-1)
        x = F.relu(self.linear(x))
        for i in range(self.n_rel):
            res1 = x
            x, _ = self.relations[i](x, neighbors, neighbors)
            x = x + res1
            x = self.layernorms[i](x)
            res2 = x
            x = self.ffs2[i](F.relu(self.ffs1[i](x)))
            x = x + res2
            x = self.layernorms[i+self.n_rel](x)
        out = torch.squeeze(self.out(x))
        return F.softmax(out, dim=0)
    
    def embed_observe(self, x, pos):
        with torch.no_grad():
            # if x.shape == (1, 6, 6):
            #     pass
            # else:
            #     print(x.shape)
            x = self.encoder(x)
            x = x.view(1, self.dim_embed)
            pos = self.pos_embed(pos)
            x = torch.cat([x, pos], dim=-1)
            x = F.relu(self.linear(x))
        return x
    


class GCNPosOnly(nn.Module):
    def __init__(self, n_embed_channel, size_kernal, dim_observe, size_world, n_rel, n_head=8):
        super().__init__()
        n_hidden = 64
        self.n_rel = n_rel
        self.dim_map = size_world[0] * size_world[1]
        # self.dim_embed = n_embed_channel * ((dim_observe - size_kernal) + 1)**2 # dim_out of CNN = (dim_in - size_kernal + padding) / stride + 1
        # self.encoder = nn.Conv2d(1, n_embed_channel, size_kernal, bias=False)
        self.pos_embed = nn.Linear(3, n_hidden, bias=False)
        self.layernorm0 = nn.LayerNorm(n_hidden)
        # self.linear = nn.Linear(32, n_hidden, bias=False)
        self.relations = nn.ModuleList([nn.MultiheadAttention(n_hidden, n_head, bias=False) for _ in range(n_rel)])
        self.layernorms = nn.ModuleList([nn.LayerNorm(n_hidden) for _ in range(2*n_rel)])
        self.ffs1 = nn.ModuleList([nn.Linear(n_hidden, n_hidden, bias=False) for _ in range(n_rel)])
        self.ffs2 = nn.ModuleList([nn.Linear(n_hidden, n_hidden, bias=False) for _ in range(n_rel)])
        # self.out = nn.Linear(n_hidden, self.dim_map, bias=False)
        self.out = nn.Linear(n_hidden, 36, bias=False)


    def forward(self, nothing, neighbors, pos):
        '''
        pos.shape = (1, 3)
        neighbors.shape = (number of neighbors of current agent at this time step + 1(self), dim_embed)
        Using unbatch data because the number of neighbors changes with agent and time.
        '''
        x = F.leaky_relu(self.pos_embed(pos))
        x = self.layernorm0(x)
        for i in range(self.n_rel):
            res1 = x
            x, _ = self.relations[i](x, neighbors, neighbors)
            x = x + res1
            x = self.layernorms[i](x)
            res2 = x
            x = self.ffs2[i](F.leaky_relu(self.ffs1[i](x)))
            x = x + res2
            x = self.layernorms[i+self.n_rel](x)
        out = torch.squeeze(self.out(x))
        return F.softmax(out, dim=0)
    
    def embed_observe(self, nothing, pos):
        with torch.no_grad():
            x = F.leaky_relu(self.pos_embed(pos))
        return x    

class NetTest(nn.Module):
    def __init__(self, n_embed_channel, size_kernal, dim_observe, size_world, n_rel, n_head=8):
        super().__init__()
        self.dim_map = size_world[0] * size_world[1]
        self.linear1 = nn.Linear(3, 256)

        self.linear2 = nn.Linear(256, 512)
        # self.pos_linear1 = nn.Linear(3, 128)
        # self.pos_linear2 = nn.Linear(128, 256)
        self.out = nn.Linear(512, self.dim_map)

    def forward(self, nothing, neighbors, pos):
        # pos = self.pos_linear1(pos)
        # pos = F.relu(pos)
        # pos = self.pos_linear2(pos)
        # pos = F.relu(pos)
        x = self.linear1(pos)
        x = F.leaky_relu(x)

        x = self.linear2(x)
        x = F.leaky_relu(x)
        # x = torch.cat([x, pos], dim=-1)
        out = torch.squeeze(self.out(x))
        prob = F.softmax(out, dim=0)
        # idx = torch.argmax(prob)
        # eps = (prob[idx] * 0.5)/2499
        # prob[idx] = prob[idx] * 0.5
        # prob[0:idx] = prob[0:idx] + eps
        # prob[idx+1:] = prob[idx+1:] + eps
        # print(torch.max(prob))
        return prob
    
    def embed_observe(self, nothing, pos):
        with torch.autograd.no_grad():
            x = self.linear1(pos)
            x = F.leaky_relu(x)
            x = self.linear2(x)
            x = F.leaky_relu(x)
        return x
    

class NetTest1(nn.Module):
    def __init__(self, n_embed_channel, size_kernal, dim_observe, size_world, n_rel, n_head=8):
        super().__init__()
        self.dim_map = size_world[0] * size_world[1]
        self.conv = nn.Conv2d(2, 1, 3)
        self.pool = nn.MaxPool2d(3)
        # self.dim_embed = 784 # This is for 30 * 30 world size
        self.linear1 = nn.Linear(3, 256)
        self.linear2 = nn.Linear(337, 512)
        # self.pos_linear1 = nn.Linear(3, 128)
        # self.pos_linear2 = nn.Linear(128, 256)
        self.out = nn.Linear(512, self.dim_map)

    def forward(self, heatmap, cov_lvl, pos):
        maps = torch.stack([heatmap, cov_lvl])
        maps = self.conv(maps)
        maps = self.pool(maps)
        x = self.linear1(pos)
        x = F.leaky_relu(x)
        x = torch.cat([x, maps.view(1, -1)], dim=-1)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        # x = torch.cat([x, pos], dim=-1)
        out = torch.squeeze(self.out(x))
        prob = F.softmax(out, dim=0)
        # idx = torch.argmax(prob)
        # eps = (prob[idx] * 0.5)/2499
        # prob[idx] = prob[idx] * 0.5
        # prob[0:idx] = prob[0:idx] + eps
        # prob[idx+1:] = prob[idx+1:] + eps
        # print(torch.max(prob))
        return prob
    
    def embed_observe(self, nothing):
        return