# A Project for ME 360 - Control Systems
I created a controller for the classic Cart-Pole system.
The controller was created programmatically by:
- Deriving the equations of motion using Lagrangian Mechanics
- Linearizing the equations of motion about the point where the cart is stationary with the pole balanced upright
- Creating a PD controller using the linear quadratic regulator algorithm
- Testing the controller on the original (non-linear) system under various initial conditions

You can read the full paper ![here](/mini_project/Final%20Report.pdf).

The following graphs show some of the results:

![Theta Perturbation](/mini_project/theta_perturbation.png)  
![x Perturbation](/mini_project/x_perturbation.png)  
![v Perturbation](/mini_project/v_perturbation.png)  
![Omega Perturbation](/mini_project/omega_perturbation.png)  
