import torch
import hypers
import numpy as np
import pypose as pp
import matplotlib.pyplot as plt


class Unicycle(pp.module.NLS):
    def __init__(self, id, dt, init_states, v_lim=1, omega_lim=torch.pi/4, r_d = 0.5, r_c=2.0, r_s=5, alpha=0.005, dev='cuda'):
        super().__init__()
        # Dynamics
        self.id = id
        self.dt = dt
        self.states = init_states
        self.v_range = [-v_lim, v_lim]
        self.omega_range = [-omega_lim, omega_lim]
        # Sensening and collision avoidance
        self.r_d = r_d
        self.r_c = r_c
        self.r_s = r_s
        self.alpha = alpha
        # Communication
        self.neighbors = None
        self.local_embed = None

        self.dev = dev

    def state_transition(self, state, input, t=None):
         x, y, theta = state[0]
         v, omega = input[0]
         d_state = torch.stack([v*torch.cos(theta), v*torch.sin(theta), omega], dim=0)
         return (state.squeeze() + d_state * self.dt).unsqueeze(0)
    
    def observation(self, state, input, t=None):
         return state

    def get_discount_cost(self, costs):
            discounts = torch.tensor([hypers.discount**(i+1) for i in range(len(costs))])
            return torch.sum(costs * discounts)

    def update_neighbors(self, dist):
        self.neighbors = np.where(dist <= self.r_c)[0]



if __name__=='__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_batch = 1
    dt = 0.1
    T = 7
    Q_x = 10
    Q_y = 10
    Q_theta = 10
    R_v = 0.5
    R_omega = 0.005
    n_states = 3
    n_control = 2
    n_sc = n_states + n_control
    Q = torch.diag(torch.tensor([Q_x, Q_y, Q_theta, R_v, R_omega]))
    Q = torch.tile(Q, (n_batch, T, 1, 1)).to(dev)
    p = torch.zeros((n_batch, T, n_sc), device=dev)
    # time  = torch.arange(0, T, device=dev) * dt

    target = torch.tensor([5., 8., 0.], device=dev)
    init_states = torch.tensor([1., 2., 0.], device=dev).unsqueeze(0)
    init_control = torch.tile(torch.tensor([0., 0.], device=dev), (n_batch, T, 1))

    unicycle = Unicycle(0, dt, init_states.squeeze())
    MPC = pp.module.MPC(unicycle, Q, p, T).to(dev)
    states = []
    for i in range(40):
        x, u, cost = MPC(dt, (unicycle.states - target).unsqueeze(0))
        unicycle.states = x[0, 1, :] + target
        states.append(unicycle.states)

    states = torch.stack(states, dim=0).cpu().numpy()
    plt.plot(states[:, 0])
    plt.plot(states[:, 1])
    plt.plot(states[:, 2])
    plt.show()
    # print(u)
    # print(cost)
    