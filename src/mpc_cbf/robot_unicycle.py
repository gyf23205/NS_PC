import numpy as np
import casadi
import copy
import matplotlib
import torch
from torch.distributions import Categorical
matplotlib.use('tkagg')
import hypers
from mpc_cbf.plan_dubins import plan_dubins_path
# from visualization import simulate
from utils import dm_to_array, normalize_pos
from networks.gcn import GraphConvNet

class MPC_CBF_Unicycle:
    def __init__(self, id, dt ,N, v_lim, omega_lim,  
                 Q, R, flag_cbf, init_state, 
                 obstacles= None,  obs_diam = 3, r_d = 0.5, r_c=2.0, r_s=5, alpha=0.005, dev='cuda'):
        '''
        Inputs:
        dt: Scalar, computing period of the MPC solver.
        N: Time horizen of the MPC solver.
        Q: np array with size (3,), containing the diagnal elements of the PSD matrix which is used to penalize the states in the quadratic MPC cost.
        R: np array with size (3,) vector, penalize inputs in the MPC cost.
        v_lim: Scalar. Maximum velocity input.
        omega_lim: Scalar. Maximum angular velocity input.
        flag_cbf: Bool. Flag to enable obstacle avoidance
        init_state: np array. [init_x, init_y, init_omega]
        r_d: Scalar. Radius of danger, i.e., collision radius.
        r_c: Scalar. Communication radius.
        r_s: Integer for square shape sensing regions. Sensing radius. Grids whose center locates in this range is considered as been "checked" by the robot. Heat of that grid is set to 0.
        alpha: Positive scalar for class-K function.
        '''
        self.id = id # 0 ~ N-1
        self.dt = dt # Period
        self.N = N  # Horizon Length 
        self.Q_x = Q[0]
        self.Q_y = Q[1]
        self.Q_theta = Q[2]
        self.R_v = R[0]
        self.R_omega = R[1]
        self.n_states = 0
        self.n_controls = 0
        self.states = init_state

        self.v_lim = v_lim
        self.omega_lim = omega_lim

        # Initialized in mpc_setup
        self.solver = None
        self.f = None

        self.r_d = r_d
        self.r_c = r_c
        self.r_s = r_s
        self.obs_diam = obs_diam
        self.flag_cbf = flag_cbf # Bool flag to enable obstacle avoidance
        self.alpha= alpha # Parameter for scalar class-K function, must be positive
        self.obstacles = obstacles
        self.neighbors = None
        self.local_embed = None
        # self.running = False # Indicate if current waypoints have all been visited.
        # self.buffer_states = [] # Store all the future MPC-generated states for current set of waypoints.
        # self.pointer_buffer_state = 0
        # self.buffer_cost = []
        # self.log_prob = None
        self.dev = dev
        # Setup with initialization params
        self.setup()

    def get_discount_cost(self, costs):
        discounts = torch.tensor([hypers.discount**(i+1) for i in range(len(costs))])
        return torch.sum(costs * discounts)

    def update_neighbors(self, dist):
        self.neighbors = np.where(dist <= self.r_c)[0]

    # def set_decision_NN(self, dim_observe, size_world):
    #     self.decisionNN = GraphConvNet(hypers.n_embed_channel, size_kernal=3, dim_observe=dim_observe, size_world=size_world, n_rel=hypers.n_rel)

    def embed_local(self, observe, size_world, len_grid):
        observe = observe.to(self.dev)
        pos = normalize_pos(self.states, size_world, len_grid)
        self.local_embed = self.decisionNN.embed_observe(observe, torch.tensor(pos, dtype=torch.float32, device=self.dev).view(1, -1))

    def generate_waypoints(self, observe, neighbors_observe, size_world, len_grid):
        '''
        Generate waypoints given local observation and neighbors' embeded information
        observe: (1, dim_map, dim_map)
        neighbors_observe: (n_neighbors, dim_embed)
        '''
        observe, neighbors_observe = torch.tensor(observe, dtype=torch.float32, device=self.dev), neighbors_observe.to(self.dev)
        pos = normalize_pos(self.states, size_world, len_grid)
        logits = self.decisionNN(observe, neighbors_observe, torch.tensor(pos, dtype=torch.float32, device=self.dev).view(1, -1))
        probs = torch.nn.functional.softmax(logits, dim=0)
        # actions = torch.multinomial(prob, 1).cpu().numpy() # Sample one destination for current time step
        # log_prob = torch.log(prob)
        # print(torch.argmax(probs))
        # print(torch.max(probs))
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, logits

    def generate_waypoints_centralized(self, heatmap, cov_lvl):
        '''
        Generate waypoints given local observation and neighbors' embeded information
        observe: (1, dim_map, dim_map)
        neighbors_observe: (n_neighbors, dim_embed)
        '''
        # observe, neighbors_observe = torch.tensor(observe, dtype=torch.float32, device=self.dev), neighbors_observe.to(self.dev)
        heatmap = torch.tensor(heatmap, dtype=torch.float32, device=self.dev)
        cov_lvl = torch.tensor(cov_lvl, dtype=torch.float32, device=self.dev)
        prob = self.decisionNN(heatmap, cov_lvl, torch.tensor(self.states, dtype=torch.float32, device=self.dev).view(1, -1))
        # actions = torch.multinomial(prob, 1).cpu().numpy() # Sample one destination for current time step
        # log_prob = torch.log(prob)
        print(torch.argmax(prob))
        dist = Categorical(prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.cpu().numpy(), log_prob

    ## Utilies used in MPC optimization
    # CBF Implementation
    def h_obs(self, state, obstacle, r):
            ox, oy, _ = obstacle
            return ((ox - state[0])**2 + (oy - state[1])**2 - r**2)


    def shift_timestep(self, h, time, state, control):
        # delta_state = self.f(state, control[:, 0])
        # next_state = casadi.DM.full(state + h * delta_state)
        # next_time = time + h
        # next_control = casadi.horzcat(control[:, 1:],
        #                             casadi.reshape(control[:, -1], -1, 1))
        # return next_time, next_state, next_control
        delta_state = self.f(state, control[:, 0])
        next_state = dm_to_array(state + h * delta_state)
        next_state[0:2][next_state[0:2, :] < 0] = 1e-7
        next_state = casadi.DM(next_state)
        next_state = casadi.DM.full(next_state)
        next_time = time + h
        next_control = casadi.horzcat(control[:, 1:],
                                    casadi.reshape(control[:, -1], -1, 1))
        return next_time, next_state, next_control

    def update_param(self, x0, ref, k, N):
        p = casadi.vertcat(x0)
        for l in range(N):
            if k+l < ref.shape[0]:
                ref_state = ref[k+l, :]
            else:
                ref_state = ref[-1, :]
            xt = casadi.DM([ref_state[0], ref_state[1], ref_state[2]])
            p = casadi.vertcat(p, xt)
        return p
    
    def setup(self):
        x = casadi.SX.sym('x')
        y = casadi.SX.sym('y')
        theta = casadi.SX.sym('theta')
        states = casadi.vertcat(x, y, theta)
        self.n_states = states.numel()

        v = casadi.SX.sym('v')
        omega = casadi.SX.sym('omega')
        controls = casadi.vertcat(v, omega)
        self.n_controls = controls.numel()

        X = casadi.SX.sym('X', self.n_states, self.N + 1)
        U = casadi.SX.sym('U', self.n_controls, self.N)
        P = casadi.SX.sym('P', (self.N + 1) * self.n_states)
        Q = casadi.diagcat(self.Q_x, self.Q_y, self.Q_theta)
        R = casadi.diagcat(self.R_v, self.R_omega)

        rhs = casadi.vertcat(v * casadi.cos(theta), v * casadi.sin(theta), omega)
        self.f = casadi.Function('f', [states, controls], [rhs])

        cost = 0
        g = X[:, 0] - P[:self.n_states]

        for k in range(self.N):
            state = X[:, k]
            control = U[:, k]
            cost = cost + (state - P[(k+1)*self.n_states:(k+2)*self.n_states]).T @ Q @ (state - P[(k+1)*self.n_states:(k+2)*self.n_states]) + \
                    control.T @ R @ control
            next_state = X[:, k + 1]
            k_1 = self.f(state, control)
            k_2 = self.f(state + self.dt/2 * k_1, control)
            k_3 = self.f(state + self.dt/2 * k_2, control)
            k_4 = self.f(state + self.dt * k_3, control)
            predicted_state = state + self.dt/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
            g = casadi.vertcat(g, next_state - predicted_state)
        
        if self.flag_cbf:
            for k in range(self.N):
                state = X[:, k]
                next_state = X[:, k+1]
                for obs in self.obstacles:    
                    h = self.h_obs(state, obs, (self.r_d / 2 + self.obs_diam / 2))
                    h_next = self.h_obs(next_state, obs, (self.r_d / 2 + self.obs_diam / 2))
                    g = casadi.vertcat(g,-(h_next-h + self.alpha*h))

        opt_variables = casadi.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))

        nlp_prob = {
            'f': cost,
            'x': opt_variables,
            'g': g,
            'p': P
        }

        opts = {
            'ipopt': {
                'sb': 'yes',
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0,
        }
        self.solver = casadi.nlpsol('solver', 'ipopt', nlp_prob, opts)


    def solve(self, X0, u0,  ref, idx, ub, lb):   

        lbx = casadi.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N, 1))
        ubx = casadi.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N, 1))

        lbx[0:self.n_states * (self.N + 1):self.n_states] = lb[0]
        lbx[1:self.n_states * (self.N + 1):self.n_states] = lb[1]
        lbx[2:self.n_states * (self.N + 1):self.n_states] = lb[2]

        ubx[0:self.n_states * (self.N + 1):self.n_states] = ub[0]
        ubx[1:self.n_states * (self.N + 1):self.n_states] = ub[1]
        ubx[2:self.n_states * (self.N + 1):self.n_states] = ub[2]

        lbx[self.n_states * (self.N + 1):self.n_states * (self.N + 1) + self.n_controls * self.N:self.n_controls] = self.v_lim[0]
        ubx[self.n_states * (self.N + 1):self.n_states * (self.N + 1) + self.n_controls * self.N:self.n_controls] = self.v_lim[1]
        lbx[self.n_states * (self.N + 1) + 1:self.n_states * (self.N + 1) + self.n_controls *self.N:self.n_controls] = self.omega_lim[0]
        ubx[self.n_states * (self.N + 1) + 1:self.n_states * (self.N + 1) + self.n_controls * self.N:self.n_controls] = self.omega_lim[1]
        
        if self.flag_cbf:
            lbg = casadi.DM.zeros((self.n_states * (self.N + 1) + len(self.obstacles)*(self.N), 1))
            ubg = casadi.DM.zeros((self.n_states * (self.N + 1) + len(self.obstacles)*(self.N), 1))

            lbg[self.n_states * (self.N + 1):] = -casadi.inf
            ubg[self.n_states * (self.N + 1):] = 0
        else:
            lbg = casadi.DM.zeros((self.n_states * (self.N+1)))
            ubg = -casadi.DM.zeros((self.n_states * (self.N+1)))

        args = {
            'lbg': lbg,
            'ubg': ubg,
            'lbx': lbx,
            'ubx': ubx
        }
        args['p'] = self.update_param(X0[:,0], ref, idx, self.N)
        args['x0'] = casadi.vertcat(casadi.reshape(X0, self.n_states * (self.N + 1), 1),
                                        casadi.reshape(u0, self.n_controls * self.N, 1))

        sol = self.solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'],
                        lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

        u = casadi.reshape(sol['x'][self.n_states * (self.N + 1):], self.n_controls, self.N)
        X = casadi.reshape(sol['x'][:self.n_states * (self.N + 1)], self.n_states, self.N + 1)
        return u, X 


# def main(args=None):
#     Q_x = 10
#     Q_y = 10
#     Q_theta = 10
#     R_v = 0.5
#     R_omega = 0.005

#     dt = 0.1
#     N = 20
#     idx = 0
#     t0 = 0

#     x_0 = 0
#     y_0 = 0
#     theta_0 = 0

#     x_goal = 6
#     y_goal = 3
#     theta_goal = np.pi/2

#     r = 1 
#     v = 1
#     path_x, path_y, path_yaw, _, _ = plan_dubins_path(x_0, y_0, theta_0, x_goal, y_goal, theta_goal, r, step_size=v*dt)

#     ref_states = np.array([path_x, path_y, path_yaw]).T

#     v_lim = [-1, 1]
#     omega_lim = [-casadi.pi/4, casadi.pi/4]
#     Q = [Q_x, Q_y, Q_theta]
#     R = [R_v, R_omega]
#     obs_list = [(4,0), (8,5), (6,9), (2, -4), (8,-5), (6,-9), (5, -6)]

#     mpc_cbf = MPC_CBF_Unicycle(dt,N, v_lim, omega_lim, Q, R, obstacles= obs_list, flag_cbf=True)
#     state_0 = casadi.DM([x_0, y_0, theta_0])
#     u0 = casadi.DM.zeros((mpc_cbf.n_controls, N))
#     X0 = casadi.repmat(state_0, 1, N + 1)
#     cat_states = dm_to_array(X0)
#     cat_controls = dm_to_array(u0[:, 0])

#     x_arr = [x_0]
#     y_arr = [y_0]
#     for i in range(len(ref_states)):    
#         u, X_pred = mpc_cbf.solve(X0, u0, ref_states, i)

#         cat_states = np.dstack((cat_states, dm_to_array(X_pred)))
#         cat_controls = np.dstack((cat_controls, dm_to_array(u[:, 0])))
        
#         t0, X0, u0 = mpc_cbf.shift_timestep(dt, t0, X_pred, u)
        
#         x_arr.append(X0[0,1])
#         y_arr.append(X0[1,1])
#         idx += 1
    
#     num_frames = len(ref_states)
#     simulate(ref_states, cat_states, cat_controls, num_frames, dt, N,
#          np.array([x_0, y_0, theta_0, x_goal, y_goal, theta_goal]), save=False)


# if __name__ == '__main__':
#     main()