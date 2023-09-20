# this notebook is used explore the performance of hmc on forrester function
# the forrester function is a 1D function
# the function is defined as:
# f(x) = (6x - 2)^2 sin(12x - 4) + \epsilon
import autograd.numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist

import hamiltorch

hamiltorch.set_random_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def forrester(x, noise=0.1):
    # x is a torch tensor
    obj = (6 * x - 2) ** 2 * torch.sin(12 * x - 4) + \
        noise * torch.randn_like(x)
    return obj.reshape(-1, 1)


# get samples for training
sample_x = torch.linspace(0, 1, 100).reshape(-1, 1)
# get samples for testing
test_x = torch.linspace(-1, 2, 1000).reshape(-1, 1)
# get the corresponding objective values
sample_y = forrester(sample_x)
# get the corresponding objective values for testing
test_y = forrester(test_x, noise=0.0)


# plot the forrester function
plt.figure(figsize=(8, 6))
plt.plot(sample_x.numpy(), sample_y.numpy(), 'r.')
plt.plot(test_x.numpy(), test_y.numpy(), 'b-')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Forrester function')
plt.show()


# define the model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(1, 50)
        self.fc2 = torch.nn.Linear(50, 50)
        self.fc3 = torch.nn.Linear(50, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# prepare the for sampling
step_size = 0.001
num_samples = 10000
L = 50
burn = -1
store_on_GPU = False
debug = False
model_loss = 'regression'
mass = 1.0

tau = 1.0  # Prior Precision, inverse of sigma2
# Output Precision (Likelihood Precision), known noise, inverse of sigma_a**2
tau_out = 100

tau_list = []
for w in net.parameters():
    # set the prior precision to be the same for each set of weights
    tau_list.append(tau)

tau_list = torch.tensor(tau_list).to(device)

# Set initial weights
params_init = hamiltorch.util.flatten(net).to(device).clone()
# Set the Inverse of the Mass matrix
inv_mass = torch.ones(params_init.shape) / mass
print(f'number of parameters: {params_init.shape}')

integrator = hamiltorch.Integrator.EXPLICIT
sampler = hamiltorch.Sampler.HMC

#
params_hmc_f = hamiltorch.sample_model(net, sample_x.to(device), sample_y.to(device), params_init=params_init,
                                       model_loss=model_loss, num_samples=num_samples,
                                       burn=burn, inv_mass=inv_mass.to(device), step_size=step_size,
                                       num_steps_per_sample=L, tau_out=tau_out, tau_list=tau_list,
                                       debug=debug, store_on_GPU=store_on_GPU,
                                       sampler=sampler)

# get prediction
pred_list, log_probs_f = hamiltorch.predict_model(net, x=test_x.to(device),
                                                  y=test_y.to(device), samples=params_hmc_f,
                                                  model_loss=model_loss, tau_out=tau_out,
                                                  tau_list=tau_list)

# get the log likehood function value for training data
pred_list_tr, log_probs_split_tr = hamiltorch.predict_model(net, x=sample_x.to(device), y=sample_y.to(device),
                                                            samples=params_hmc_f, model_loss=model_loss,
                                                            tau_out=tau_out, tau_list=tau_list)
ll_full = torch.zeros(pred_list_tr.shape[0])
ll_full[0] = - 0.5 * tau_out * ((pred_list_tr[0].cpu() - sample_y) ** 2).sum(0)
for i in range(pred_list_tr.shape[0]):
    ll_full[i] = - 0.5 * tau_out * \
        ((pred_list_tr[:i].mean(0).cpu() - sample_y) ** 2).sum(0)

f, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.set_title('Training Log-Likeilihood')
ax.plot(ll_full)
plt.xlabel("Iteration")
plt.ylabel("log likelihood")
plt.grid()
plt.savefig('forrester_training_log_likelihood.png', dpi=300)
plt.show()


num_burn_in = 100
m = pred_list[num_burn_in:].mean(0).to('cpu')
s = pred_list[num_burn_in:].std(0).to('cpu')
s_al = (pred_list[num_burn_in:].var(0).to('cpu') + tau_out ** -1) ** 0.5

f, ax = plt.subplots(1, 1, figsize=(10, 4))
# Get upper and lower confidence bounds
lower, upper = (m - s*2).flatten(), (m + s*2).flatten()
lower_al, upper_al = (m - s_al*2).flatten(), (m + s_al*2).flatten()

# Plot training data as black stars
ax.plot(sample_x.numpy(), sample_y.numpy(), 'k*', rasterized=True)
# Plot predictive means as blue line
ax.plot(test_x.numpy(), m.numpy(), 'b', rasterized=True)
# Shade between the lower and upper confidence bounds
ax.fill_between(test_x.flatten().numpy(), lower.numpy(),
                upper.numpy(), alpha=0.5, rasterized=True)
ax.fill_between(test_x.flatten().numpy(), lower_al.numpy(),
                upper_al.numpy(), alpha=0.2, rasterized=True)
plt.grid()
ax.legend(['Observed Data', 'Mean', 'Epistemic', 'Total'], fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)

bbox = {'facecolor': 'white', 'alpha': 0.8, 'pad': 1,
        'boxstyle': 'round', 'edgecolor': 'black'}
plt.tight_layout()
plt.savefig('forrester_hmc.png', dpi=300)
plt.show()
