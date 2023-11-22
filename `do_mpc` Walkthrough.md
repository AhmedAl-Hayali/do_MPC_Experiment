>[!tip] Walkthrough source
>[`do-mpc` 4.6.2 documentation - "Getting started: MPC"](https://www.do-mpc.com/en/latest/getting_started.html)

This walkthrough aims to serve as an introductory reference for future MPC (model predictive control) development. I (Ahmed) would recommend looking through this walkthrough and the *[Getting started: MPC](https://www.do-mpc.com/en/latest/getting_started.html)* page simultaneously (split-screen) so you can compare and assess changes I make to the example's source code.

This walkthrough is not intended to explore MPCs' inner workings or the formulation of systems that are to be optimized for by an MPC - it is solely focused on the development component of implementing an MPC in python.

Ensure you follow the [`do-mpc` 4.6.2 documentation - Installation](https://www.do-mpc.com/en/latest/installation.html) page for library setup, but at a glance, running `pip install numpy matplotlib casadi do_mpc[full]` should suffice.
# Dependencies Setup
```python
# Convenient for working with n-dimensional arrays (`ndarray`s)
import numpy as np
# Plotting library; rcParams to modify, with more fidelity, simulator output
from matplotlib import  pyplot as plt, rcParams
# MPC API
import do_mpc
# Algorithmic differentiation engine - also handles symbolic variable representations
from casadi import vertcat, dot
```

# System Formulation
For the following collection of system *states*, *inputs*, and *parameters*,
$$\begin{aligned}
\phi_i & \hspace{1.5em} i^{\text{th}}\text{ disc angle, } i = 1,\, 2,\, 3 & \hspace{1.5em} \text{system states} \\
\phi_{m,i} & \hspace{1.5em} i^{\text{th}}\text{ stepper motor angle, } i = 1,\, 2 & \hspace{1.5em} \text{system inputs} \\
\Theta_i & \hspace{1.5em} i^{\text{th}}\text{ disc inertia, } i = 1,\, 2,\, 3 & \hspace{1.5em} \text{system parameters} \\
c_i & \hspace{1.5em} i^{\text{th}}\text{ spring constant, } i = 1,\, 2,\, 3,\, 4 & \hspace{1.5em} \text{system parameters} \\
d_i & \hspace{1.5em} i^{\text{th}}\text{ damping factor, } i = 1,\, 2,\, 3 & \hspace{1.5em} \text{system parameters,}
\end{aligned}$$
we will model the system given by the second-order ordinary differential equation (ODE)
$$\begin{aligned}
\Theta_1 \ddot{\phi}_1 & = - c_1 \left(\phi_1 - \phi_{m,\, 1}\right) - c_2 \left(\phi_1 - \phi_2\right) - d_1 \dot{\phi}_1 \\
\Theta_2 \ddot{\phi}_2 & = - c_2 \left(\phi_2 - \phi_1\right) - c_3 \left(\phi_2 - \phi_3\right) - d_2 \dot{\phi}_2 \\
\Theta_3 \ddot{\phi}_3 & = - c_3 \left(\phi_3 - \phi_2\right) - c_4 \left(\phi_3 - \phi_{m,\, 2}\right) - d_3 \dot{\phi}_3.
\end{aligned}$$

# Model Creation
### System Derivation
To model the system, begin by converting the second-order ODE above to a collection of first-order ODEs in the form[^1]
$$\frac{\partial \boldsymbol{x}}{\partial t} = f\left(\boldsymbol{x},\, \boldsymbol{u},\, \boldsymbol{p}\right),$$
where $f$ is a function of
$$\begin{aligned}
\boldsymbol{x} \in \mathbb{R}^{n_x}, & \hspace{1.5em} \text{the model states,} \\
\boldsymbol{u} \in \mathbb{R}^{n_u}, & \hspace{1.5em} \text{the model inputs, and} \\
\boldsymbol{p} \in \mathbb{R}^{n_p}, & \hspace{1.5em} \text{the model parameters.} \\
\end{aligned}$$
Then, the reformulated second-order ODE becomes the following first-order ODEs:
 $$\boldsymbol{x} = \begin{bmatrix}
 x_1 & x_2 & x_3 & x_4 & x_5 & x_6
 \end{bmatrix}^{\top} = \begin{bmatrix}
 \phi_1 & \phi_2 & \phi_3 & \dot{\phi}_1 & \dot{\phi}_2 & \dot{\phi}_3
 \end{bmatrix}^{\top}$$
Then,
 $$\begin{aligned}
 \frac{\partial \boldsymbol{x}}{\partial t} & = f\left(\boldsymbol{x},\, \boldsymbol{u},\, \boldsymbol{p}\right) =  \begin{bmatrix}
 \dot{x_1} & \dot{x_2} & \dot{x_3} & \dot{x_4} & \dot{x_5} & \dot{x_6}
 \end{bmatrix}^{\top} \\
 & = \begin{bmatrix}
 \dot{\phi_1} \\ \dot{\phi_2} \\ \dot{\phi_3} \\ \ddot{\phi_1} \\ \ddot{\phi_2} \\ \ddot{\phi_3}
 \end{bmatrix} = \begin{bmatrix}
 x_4 \\ x_5 \\ x_6 \\
\displaystyle - \frac{c_1}{\Theta_1} \left(\phi_1 - \phi_{m,\, 1}\right) - \frac{c_2}{\Theta_1} \left(\phi_1 - \phi_2\right) - \frac{d_1}{\Theta_1} \dot{\phi}_1 \\
\displaystyle - \frac{c_2}{\Theta_2} \left(\phi_2 - \phi_1\right) - \frac{c_3}{\Theta_2} \left(\phi_2 - \phi_3\right) - \frac{d_2}{\Theta_2} \dot{\phi}_2 \\
\displaystyle - \frac{c_3}{\Theta_3} \left(\phi_3 - \phi_2\right) - \frac{c_4}{\Theta_3} \left(\phi_3 - \phi_{m,\, 2}\right) - \frac{d_3}{\Theta_3} \dot{\phi}_3.
 \end{bmatrix}.
 \end{aligned}$$
This is the final model we will be implementing in the remainder of the walkthrough.

### Model Configuration
#### Model Initialization
To implement the system described by the first-order ODEs above, we begin by initializing a `model.Model` object.
> \[The `Model`\] class holds the full model description and is at the core of `do_mpc.simulator.Simulator`, `do_mpc.controller.MPC` and `do_mpc.estimator.Estimator`.[^2]
```python
model_type = 'continuous'  # Can also be 'discrete' - refer to `Model` object docs
model = do_mpc.model.Model(model_type=model_type)
```
Beyond initialization of the model, we must describe the system of interest programmatically.

#### Model Variables
We begin by parameterizing $n_x$, $n_u$, and $n_p$,
```python
n_states = 3
n_inputs = 2
n_params = 3
```
then symbolically express model variables - the states, inputs and parameters; alternatively referred to as `_x`, `_u`, and `_p`, respectively - using the `set_variable` method of a `Model` object.
```python
disc_angle = model.set_variable(var_type='states',
                                var_name='disc_angle',
                                shape=n_states)
d_disc_angle = model.set_variable(var_type='states',
                                  var_name='d_disc_angle',
                                  shape=n_states)

# *True* input mechanism values - treated as system states
motor_angle = model.set_variable(var_type='states',
                                 var_name='motor_angle',
                                 shape=n_inputs)

# *Desired* input mechanism values - treated as system inputs
motor_angle_set = model.set_variable(var_type='inputs',
                                     var_name='motor_angle_set',
                                     shape=n_inputs)

# Parameters
disc_inertia = model.set_variable(var_type='parameter',
								  var_name='disc_inertia',
								  shape=n_params)
```
Additional model constants can be stored as `numpy` arrays.
```python
spring_constant = np.array([2.697, 2.66, 3.05, 2.86]) * 1e-3  
damping_factor = np.array([6.78, 8.01, 8.82]) * 1e-5
```

#### Model Update Rules
The `set_rhs` method of a `Model` object defines the update rule, namely
$$\frac{\partial \boldsymbol{x}}{\partial t} = f\left(\boldsymbol{x},\, \boldsymbol{u},\, \boldsymbol{p}\right) =  \begin{bmatrix}
 \dot{x_1} & \dot{x_2} & \dot{x_3} & \dot{x_4} & \dot{x_5} & \dot{x_6}
 \end{bmatrix}^{\top}$$
```python
disc_angle_next = d_disc_angle
model.set_rhs(var_name='disc_angle', expr=disc_angle_next)
```
$\dot{x}_4$, $\dot{x}_5$, and $\dot{x}_6$ are more complex expressions, so we concatenate symbolic expressions using the `casadi.vertcat` method call[^3]
```python
d_disc_angle_next = vertcat(  
    1 / disc_inertia[0] * (-spring_constant[0] * (disc_angle[0] - motor_angle[0])  
                           - spring_constant[1] * (disc_angle[0] - disc_angle[1])  
                           - damping_factor[0] * d_disc_angle[0]),  
    1 / disc_inertia[1] * (-spring_constant[1] * (disc_angle[1] - disc_angle[0])  
                           - spring_constant[2] * (disc_angle[1] - disc_angle[2])  
                           - damping_factor[1] * d_disc_angle[1]),  
    1 / disc_inertia[2] * (-spring_constant[2] * (disc_angle[2] - disc_angle[1])  
                           - spring_constant[3] * (disc_angle[2] - motor_angle[1])  
                           - damping_factor[2] * d_disc_angle[2])  
)  
model.set_rhs(var_name='d_disc_angle', expr=d_disc_angle_next)
```
Finally, inputs' update rules are defined
```python
proportional_motor_angle_dampening = 1e-2  
motor_angle_next = vertcat(
    1 / proportional_motor_angle_dampening * (motor_angle_set[0] - motor_angle[0]),
    1 / proportional_motor_angle_dampening * (motor_angle_set[1] - motor_angle[1])
)
model.set_rhs(var_name='motor_angle', expr=motor_angle_next)
```
#### Model Setup
After declaring all model variables, we finalize the modelling process with a `model.setup` method call, locking the model and preventing further setting of variables, expressions, etc...[^4]
```python
model.setup()
```

# MPC Controller Configuration
### MPC Controller Instantiation
```python
mpc = do_mpc.controller.MPC(model)
```
### MPC Controller Parameter Setup
Setup controller parameters using an `MPC` object's `set_param` method[^5]
```python
mpc_kwargs = {  
    'n_horizon': 20,  
    't_step': 0.1,
    # Robust horizon for robust scenario-tree MPC;
    # Optimization problem grows ~ e^{n_robust}
    'n_robust': 1,
    'store_full_solution': True,  
}
# Very much anti-pythonic, but suggested by package devs  
for mpc_setting, mpc_setting_val in mpc_kwargs.items():  
    mpc._settings[mpc_setting] = mpc_setting_val
# Would prefer to use the line below, but it will be deprecated
# mpc.set_param(**mpc_kwargs)
```
L of parameters and MPC settings: https://www.do-mpc.com/en/latest/api/do_mpc.controller.MPCSettings.html#mpcsettings  
By default (and is currently the only option), continuous models are discretized using collocation  

[^1]: The *Getting started: MPC* page states that $f$ includes another argument, $\boldsymbol{z} \in \mathbb{R}^{n_z}$, the algebraic states, however it is unnecessary for the walkthrough (and most, in general) models, thus was omitted here. If it happens that the application does require algebraic states, it should be immediately obvious by the formulation, and hence is left to the discretion of the developer.
[^2]: [`do-mpc` 4.6.2 documentation - `Model` Objects](https://www.do-mpc.com/en/latest/api/do_mpc.model.Model.html#do_mpc.model.Model)
[^3]: [CasADi Python API documentation - `vertcat` and `horzcat`](https://web.casadi.org/docs/#:~:text=Vertical%20and%20horizontal%20concatenation%20is%20performed%20using%20the%20functions%20vertcat%20and%20horzcat)
[^4]: [`do-mpc` 4.6.2 documentation - `Model.setup`](https://www.do-mpc.com/en/latest/api/do_mpc.model.Model.html#do_mpc.model.Model.setup)
[^5]: [`do-mpc` 4.6.2 documentation - `MPC.set_param`](https://www.do-mpc.com/en/latest/api/do_mpc.controller.MPC.html#set-param)
