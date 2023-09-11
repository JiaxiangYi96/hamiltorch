# this is tutorial for HMC from their website
import matplotlib.pyplot as plt
import numpy as np
import torch

import hamiltorch

hamiltorch.set_random_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# for neural network 
def cubic_sin(x: torch.Tensor,
                 noise_std: float = 0.05) -> torch.Tensor:
    """forrestor function with noise"""

    obj =torch.sin(6*x)**3 + torch.randn_like(x) * noise_std

    return obj.reshape((-1, 1))

# generate samples 
# generate data
sample_x1 = torch.linspace(-0.8, -0.2, 17).reshape((-1, 1))
sample_x2 = torch.linspace(0.2, 0.8, 17).reshape((-1, 1))
sample_x = torch.cat([sample_x1, sample_x2], dim=0)
# sample_x = torch.linspace(0, 1, 8).reshape((-1, 1))
# print(sample_x)
sample_y = cubic_sin(sample_x, noise_std=0.1)

# test data
test_x = torch.linspace(-1, 1, 1000).reshape((-1, 1))
test_y = cubic_sin(test_x, noise_std=0.0)

# define a net
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(1, 50)
        self.fc2 = torch.nn.Linear(50, 50)
        self.fc3 = torch.nn.Linear(50, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x= torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()


# prepare the for sampling
step_size = 0.001
num_samples = 1000
L = 30
burn = -1
store_on_GPU = False
debug = False
model_loss = 'regression'
mass = 1.0

# Effect of tau
# Set to tau = 1000. to see a function that is less bendy (weights restricted to small bends)
# Set to tau = 1. for more flexible

tau = 1.0 # Prior Precision
tau_out = 100 # Output Precision (Likelihood Precision), known noise 
r = 0 # Random seed


tau_list = []
for w in net.parameters():
    tau_list.append(tau) # set the prior precision to be the same for each set of weights
    
tau_list = torch.tensor(tau_list).to(device)

# Set initial weights
params_init = hamiltorch.util.flatten(net).to(device).clone()
# Set the Inverse of the Mass matrix
inv_mass = torch.ones(params_init.shape) / mass

integrator = hamiltorch.Integrator.EXPLICIT
sampler = hamiltorch.Sampler.HMC

hamiltorch.set_random_seed(r)
params_hmc_f = hamiltorch.sample_model(net,sample_x.to(device), sample_y.to(device), params_init=params_init,
                                       model_loss=model_loss, num_samples=num_samples,
                                       burn = burn, inv_mass=inv_mass.to(device),step_size=step_size,
                                       num_steps_per_sample=L,tau_out=tau_out, tau_list=tau_list,
                                       debug=debug, store_on_GPU=store_on_GPU,
                                       sampler = sampler)

# At the moment, params_hmc_f is on the CPU so we move to GPU

params_hmc_gpu = [ll.to(device) for ll in params_hmc_f[1:]]



# Let's predict over the entire test range [-2,2]
pred_list, log_probs_f = hamiltorch.predict_model(net, x = test_x.to(device),
                                                  y = test_y.to(device), samples=params_hmc_gpu,
                                                  model_loss=model_loss, tau_out=tau_out,
                                                  tau_list=tau_list)
# Let's evaluate the performance over the training data
pred_list_tr, log_probs_split_tr = hamiltorch.predict_model(net, x = sample_x.to(device), y=sample_y.to(device),
                                                            samples=params_hmc_gpu, model_loss=model_loss,
                                                            tau_out=tau_out, tau_list=tau_list)
ll_full = torch.zeros(pred_list_tr.shape[0])
ll_full[0] = - 0.5 * tau_out * ((pred_list_tr[0].cpu() - sample_y) ** 2).sum(0)
for i in range(pred_list_tr.shape[0]):
    ll_full[i] = - 0.5 * tau_out * ((pred_list_tr[:i].mean(0).cpu() - sample_y) ** 2).sum(0)




fs = 16

m = pred_list[200:].mean(0).to('cpu')
s = pred_list[200:].std(0).to('cpu')
s_al = (pred_list[200:].var(0).to('cpu') + tau_out ** -1) ** 0.5

f, ax = plt.subplots(1, 1, figsize=(5, 4))

# Get upper and lower confidence bounds
lower, upper = (m - s*2).flatten(), (m + s*2).flatten()
# + aleotoric
lower_al, upper_al = (m - s_al*2).flatten(), (m + s_al*2).flatten()

# Plot training data as black stars
ax.plot(sample_x.numpy(), sample_y.numpy(), 'k*', rasterized=True)
# Plot predictive means as blue line
ax.plot(test_x.numpy(), m.numpy(), 'b', rasterized=True)
# Shade between the lower and upper confidence bounds
ax.fill_between(test_x.flatten().numpy(), lower.numpy(), upper.numpy(), alpha=0.5, rasterized=True)
ax.fill_between(test_x.flatten().numpy(), lower_al.numpy(), upper_al.numpy(), alpha=0.2, rasterized=True)
ax.set_ylim([-3, 3])
ax.set_xlim([-1, 1])
plt.grid()
ax.legend(['Observed Data', 'Mean', 'Epistemic', 'Aleatoric'], fontsize = fs)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)

bbox = {'facecolor': 'white', 'alpha': 0.8, 'pad': 1, 'boxstyle': 'round', 'edgecolor':'black'}
# plt.text(1., -1.5, 'Acceptance Rate: 58 %', bbox=bbox, fontsize=16, horizontalalignment='center')


plt.tight_layout()
# plt.savefig('plots/full_hmc.pdf', rasterized=True)
    
plt.show()