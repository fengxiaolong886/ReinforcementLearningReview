# coding: UTF-8

"""
This code list implement the Monte Carlo Policy Gradient Alogrithm.
------------------------------------------------------------------
Input:
    differentiable policy function $\pi_{\theta}(a|s)$

Initalize:
    Parameter $\theta$ for policy function

Repeat  experience trajectory:
    Use $\pi_{\theta}(a|s)$ to generate one trajectory $(s_0,a_0,r_1....s_T)$
    Repeat each step in trajectory:
        G <--- cumlated reward at time step t
        Calculate the policy gradient  $\Delta\theta_t = \alpha \Delta_{\theta}log\pi_{\theta}(s_t, a_t)G_t$
------------------------------------------------------------------
"""


