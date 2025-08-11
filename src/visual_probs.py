import torch
import numpy as np
import matplotlib.pyplot as plt
from networks.gcn import NetCentralized
from networks.ACnet import ACNetCentralized
import hypers
import torch.nn.functional as F

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size_world = (30, 30)
    len_grid = 1
    n_agents = 4
    hypers.init([5, 5, 0.1])
    epochs = 20
    # Set random seed for reproducibility
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    affix = f'agent{n_agents}_epoch{epochs}_centralized_ppo_shuffled'
    decisionNN = ACNetCentralized(size_world, n_agents).to(dev)
    state_dict = torch.load('results/saved_models/model_' + affix+ '.tar', weights_only=True)['net_dict']
    decisionNN.load_state_dict(state_dict)
    decisionNN.eval()

    # Create dummy inputs to test the model
    heatmap = torch.ones(size_world, dtype=torch.float32, device=dev)
    for i in range(6):
        heatmap[:, int(i*5):int((i+1)*5)] = 0.2 * i
    heatmap_flip = torch.flip(heatmap, dims=[1])

    cov_lvl = torch.zeros(size_world, dtype=torch.float32, device=dev)
    # cov_lvl[10:20, 10:20] = 0.5
    cov_lvl1 = torch.zeros(size_world, dtype=torch.float32, device=dev)

    x_init = torch.rand(n_agents, 1, device=dev) * size_world[0]
    y_init = torch.rand(n_agents, 1, device=dev) * size_world[1]
    # xy_init[0, :] = np.array([25, 25])
    # theta_init= torch.rand(n_agents, 1, device=dev)
    state_init = torch.cat((x_init, y_init), dim=-1)
    pos = state_init.view(1, n_agents * 2)

    logits, _ = decisionNN(heatmap.view(-1, 1, size_world[0], size_world[1]), cov_lvl.view(-1, 1, size_world[0], size_world[1]), pos)
    probs = F.softmax(torch.squeeze(logits), dim=-1).view(n_agents, size_world[0], size_world[1]).cpu().detach().numpy()

    logits_flip, _ = decisionNN(heatmap_flip.view(-1, 1, size_world[0], size_world[1]), cov_lvl1.view(-1, 1, size_world[0], size_world[1]), pos)
    probs_flip = F.softmax(torch.squeeze(logits_flip), dim=-1).view(n_agents, size_world[0], size_world[1]).cpu().detach().numpy()

    # plot the results
    fig, axs = plt.subplots(3, 4, figsize=(12, 6))

    vmax_prob = np.max(probs)
    vmax_prob_flip = np.max(probs_flip)
    for i in range(n_agents):
        axs[0, i].imshow(probs[i], origin='lower', cmap='viridis', vmin=0, vmax=vmax_prob)
        axs[0, i].set_title(f'Agent {i} Original')
        axs[1, i].imshow(probs_flip[i], origin='lower', cmap='viridis', vmin=0, vmax=vmax_prob_flip)
        axs[1, i].set_title(f'Agent {i} Flipped')
        axs[2, i].imshow(probs[i] - probs_flip[i], origin='lower', cmap='viridis', vmin=0, vmax=0.01)
        axs[2, i].set_title(f'Agent {i} Difference')


    plt.tight_layout()
    plt.show()