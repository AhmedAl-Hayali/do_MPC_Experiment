import numpy as np
from matplotlib import pyplot as plt, rcParams
import do_mpc
from casadi import vertcat, dot

# Requirements: model in terms of a continuous ODE, a differential algebraic equation (DAE), or discrete eq'n
# States $x$, inputs $u$, "algebraic states" $z$, and parameters $p$

# TODO: Some form of class definition I'd imagine
# TODO: Revisit variable naming - naming things phi and tau is very dependent on formulation, try to decouple
model_type = 'continuous' # Can also be 'discrete'
model = do_mpc.model.Model(model_type=model_type)

n_states = 3    # The $n_x$ for $x \in \mathbb{R}^{n_{x}}$
n_inputs = 2    # The $n_u$ for $u \in \mathbb{R}^{n_{u}}$
n_params = 3    # The $n_p$ for $p \in \mathbb{R}^{n_{p}}$

# Model variables - states, inputs, and parameters; "algebraic states" too?
# States
phi = model.set_variable(var_type='states', var_name='phi', shape=(n_states, 1))     # TODO: Can I change the `1` in shape?
dphi = model.set_variable(var_type='states', var_name='dphi', shape=(n_states, 1))

# *Desired* input mechanism values - treated as system inputs
phi_m_set = model.set_variable(var_type='inputs', var_name='phi_m_set', shape=(n_inputs, 1))

# *True* input mechanism values - treated as system states
phi_m = model.set_variable(var_type='states', var_name='phi_m', shape=(n_inputs, 1))

# Parameters
theta = model.set_variable(var_type='parameter', var_name='theta', shape=(n_params, 1))

# Note: At this point, the states, inputs, and parameters; "algebraic states" too are all symbolic variables

# Constants
c = np.array([2.697, 2.66, 3.05, 2.86])*1e-3
d = np.array([6.78, 8.01, 8.82])*1e-5

# Model update rules
# $x_{1, \dots, 3}$ updates
phi_next = dphi
model.set_rhs(var_name='phi', expr=phi_next)

# `vertcat` is how CasADI, which works in the background, concatenates symbolic expressions
dphi_next = vertcat(
    1/theta[0] * (-c[0]*(phi[0] - phi_m[0]) - c[1]*(phi[0] - phi[1]) - d[0]*dphi[0]),
    1/theta[1] * (-c[1]*(phi[1] - phi[0]) - c[2]*(phi[1] - phi[2]) - d[1]*dphi[1]),
    1/theta[2] * (-c[2]*(phi[2] - phi[1]) - c[3]*(phi[2] - phi_m[1]) - d[2]*dphi[2])
)
model.set_rhs(var_name='dphi', expr=dphi_next)

tau = 1e-2
phi_m_next = vertcat(
    1/tau*(phi_m_set[0] - phi_m[0]),
    1/tau*(phi_m_set[1] - phi_m[1])
)
model.set_rhs(var_name='phi_m', expr=phi_m_next)

# *After* defining all variables symbolically, we setup the model before beginning MPC configuration
model.setup()

mpc = do_mpc.controller.MPC(model)

# Optimizer parameterization
# For set_param settings: https://www.do-mpc.com/en/latest/api/do_mpc.controller.MPC.html#set-param
# L of parameters and MPC settings: https://www.do-mpc.com/en/latest/api/do_mpc.controller.MPCSettings.html#mpcsettings
# By default (and is currently the only option), continuous models are discretized using collocation
mpc_kwargs = {
    'n_horizon': 20,
    't_step': 0.1,
    'n_robust': 1,      # Robust horizon for robust scenario-tree MPC; Optimization problem grows ~ e^{n_robust}
    'store_full_solution': True,
}
mpc.set_param(**mpc_kwargs)

# Objective (really a penalty) function given by
# C = \sum_{k=0}^{n-1}{l(x_k, u_k, z_k, p) + \Delta u_k^{\top} R \Delta u_k} + m(x_n)
# TODO: Search, find, define, and document "Lagrange term", "r-term", and "meyer term"
# Seems like lagrange is function of everything - states, inputs, and parameters; "algebraic states" too
# r-term is related to change in inputs vs. state of inputs, and a square $\mathbb{R}^{n_u \times n_u}$ $R$ matrix
# meyer term looks like a function of next state

# In this problem, lagrange term and m-term are equal - sum of squares of phi's
mterm = lterm = dot(phi, phi)
mpc.set_objective(lterm=lterm, mterm=mterm)

# For some reason, the r-term can't be set in the objective itself, so we set it separately
mpc.set_rterm(phi_m_set=1e-2)
# The call above sets only the diagonal elements of the $R$ matrix

# Setting input and state constraints // Have to do `_x` and `_u` instead of `states` and `inputs`
mpc.bounds['lower', '_x', 'phi'] = -2*np.pi
mpc.bounds['upper', '_x', 'phi'] = 2*np.pi
mpc.bounds['lower', '_u', 'phi_m_set'] = -2*np.pi
mpc.bounds['upper', '_u', 'phi_m_set'] = 2*np.pi

# Scaling so a poorly-conditioned OCP (optimal control problem) has states, inputs, and "algebraic variables" in the
# same magnitude, improving convergence of optimization problem
mpc.scaling['_x', 'phi'] = 2

# Handling uncertainty by defining uncertain (optimizer) parameters - ones whose values can vary
# The MPC then predicts and controls various future trajectories based on combinations of the uncertain parameters # TODO: Confirm
# TODO: Parameterize the 2.25 and 1e-4
inertia_masses = 2.25*1e-4*np.array([
    [1., .9, 1.1],
    [1., .9, 1.1],
    [1., 1., 1.]
])

mpc.set_uncertainty_values(theta=inertia_masses)

# Conclude optimizer setup and create optimization problem
mpc.setup()
# We can now use the optimizer to obtain control input

# Configuring simulator
simulator = do_mpc.simulator.Simulator(model)
# Simulator can use the same model as the one provided to the optimizer,
# but could also use a different (more complex) one instead

# Simulator parameterization
# `simulator.set_param(X = y)` is used in tutorial, but it'll be deprecated, so use `simulator.settings.X = y` instead
# Full list of simulator settings parameters (only `t_step` now):
# https://www.do-mpc.com/en/latest/api/do_mpc.simulator.SimulatorSettings.html#do_mpc.simulator.SimulatorSettings
simulator.settings.t_step = 0.1

# Handling uncertainty by defining uncertain (simulator) parameters - ones whose values can vary
# Model parameters can vary with time (but they don't for this example), so we must define a function that returns
# parameter values at each timestep. First, though, we need to get the parameter template/structure:
p_template = simulator.get_p_template()
# `print(p_template.keys())`
# The template keys are the parameters whose values we need to return from the parameter function (`p_fun`)


def p_fun(t_now):
    p_template['theta'] = 2.25e-4
    return p_template


simulator.set_p_fun(p_fun)

# Finalize simulator setup
simulator.setup()

# Control Loop
# TODO: Look into `estimator` object - control horizon vs prediction horizon (`make_step()` composition multiple times?)
# TODO: x_0 as initial-state vs. x_0 as current state?
x_0 = np.pi * np.array([1, 1, -1.5, 1, -1, 1, 0, 0]).reshape(-1, 1)

simulator.x0 = x_0
mpc.x0 = x_0
# Can access both variables using `mpc.x0` or `simulator.x0`

mpc.set_initial_guess()

# Controller performance & MPC predictions plotting and assessment
# Setup matplotlib plotting parameters (rc (runtime configuration) params)
rcParams['font.size'] = 18
rcParams['lines.linewidth'] = 3
rcParams['axes.grid'] = True

# TODO: Document `mpc.data` and `simulator.data`:
#  MPCData: https://www.do-mpc.com/en/latest/api/do_mpc.data.MPCData.html#mpcdata

# Initialize `graphics` module with appropriate data objects
# Setup graphics for MPC data (query current MPC trajectories) and simulator data (current system states)
mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)

# Use `%%capture` to avoid unnecessary outputs when working with notebook cells
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(16, 9))
fig.align_ylabels()

# `graphics.add_line(...)` mimics the API of `model.add_variable(...)`, but we also pass a matplotlib axis for plotting
for graphic in [sim_graphics, mpc_graphics]:
    # We configure states and inputs identically in this example, but this need not be the case

    # Plot states (angle positions in rotating discs example) on first axis (matplotlib axis not plot axis)
    graphic.add_line(var_type='_x', var_name='phi', axis=ax[0])

    # Plot inputs (desired/set motor positions in r.d. example) on second axis (mpl axis)
    graphic.add_line(var_type='_u', var_name='phi_m_set', axis=ax[1])

ax[0].set_ylabel('angle position [rad]')
ax[1].set_ylabel('motor angle [rad]')
ax[1].set_xlabel('time [s]')

# Running simulator
# State simulator if we didn't modify inputs (i.e., u = 0 for all u)
zerod_inputs = np.zeros((2, 1))
n_timesteps = 200

for _ in range(n_timesteps):
    simulator.make_step(zerod_inputs)

# Plot resulting simulator graphics and reset axes so the entire plot is showing
sim_graphics.plot_results()
sim_graphics.reset_axes()

# plt.show()

# Running optimizer
# Obtain current control input with current state (x_0)
u_0 = mpc.make_step(x_0)

sim_graphics.clear()
mpc_graphics.plot_predictions()
mpc_graphics.reset_axes()

plt.show()
