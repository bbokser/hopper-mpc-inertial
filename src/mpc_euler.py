"""
Copyright (C) 2021-2022 Benjamin Bokser
"""

import numpy as np
import cvxpy as cp
from utils import rz, quat2euler
from scipy.linalg import expm


class Mpc:

    def __init__(self, t, N, Jinv, rhat, m, g, mu, **kwargs):
        n_x = 12  # number of states
        n_u = 6  # number of controls
        self.t = t  # sampling time (s)
        self.N = N  # prediction horizon
        self.Jinv = Jinv
        self.rhat = rhat
        self.m = m  # kg
        self.g = g
        self.mu = mu
        # phi = quat2euler(X_0[3:7])[2]  # extract z-axis euler angle
        A = np.zeros((n_x, n_x))
        A[0:3, 6:9] = np.eye(3)
        # A[3:6, 9:13] = rz(phi)
        B = np.zeros((n_x, n_u))
        B[6:9, 0:3] = np.eye(3) / self.m
        # J_w_inv = rz(phi) @ Jinv @ rz(phi).T
        # B[9:12, 0:3] = J_w_inv @ rhat
        # B[9:12, 3:6] = J_w_inv
        G = np.zeros((n_x, 1))
        G[8, :] = -self.g
        self.A = A
        self.B = B
        self.G = G
        self.n_x = n_x
        self.n_u = n_u

    def mpcontrol(self, x_in, x_ref_in, C):
        x_ref = x_ref_in
        N = self.N
        t = self.t
        m = self.m
        g = self.g
        mu = self.mu
        A = self.A
        B = self.B
        G = self.G
        Jinv = self.Jinv
        rhat = self.rhat
        n_x = self.n_x
        n_u = self.n_u
        x = cp.Variable((N+1, n_x))
        u = cp.Variable((N, n_u))
        Q = np.eye(n_x)
        R = np.eye(n_u)*0
        cost = 0
        constr = []
        u_ref = np.zeros(n_u)
        # TODO: Make Rz_phi a param instead of setting up the problem on every run
        rz_phi = rz(x_in[5])
        A[3:6, 9:] = rz_phi
        J_w_inv = rz_phi @ Jinv @ rz_phi.T  # world frame Jinv
        B[9:12, 0:3] = J_w_inv @ rhat
        B[9:12, 3:] = J_w_inv
        A_bar = np.vstack((np.hstack((A, B, G)), np.zeros((n_u + 1, n_x + n_u + 1))))
        # I_bar = np.eye(n_x + n_u + 1)
        # M = I_bar + A_bar * t + 0.5 * (t ** 2) * A_bar @ A_bar
        M = expm(A_bar * t)
        Ad = M[0:n_x, 0:n_x]
        Bd = M[0:n_x, n_x:n_x + n_u]
        Gd = M[0:n_x, -1]

        # --- calculate cost & constraints --- #
        np.fill_diagonal(Q, [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        np.fill_diagonal(R, [0., 0., 0., 0., 0., 0.])
        for k in range(0, N):
            kf = 10 if k == N - 1 else 1  # terminal cost
            kuf = 0 if k == N - 1 else 1  # terminal cost
            z = x[k, 2]
            fx = u[k, 0]
            fy = u[k, 1]
            fz = u[k, 2]
            if C[k] == 0:  # even
                u_ref[2] = 0
                cost += cp.quad_form(x[k + 1, :] - x_ref[k, :], Q * kf) + cp.quad_form(u[k, :] - u_ref, R * kuf)
                constr += [x[k + 1, :] == Ad @ x[k, :] + Bd @ u[k, :] + Gd,
                           0 == fx,
                           0 == fy,
                           0 == fz]
            else:  # odd
                u_ref[2] = m * g * 2
                cost += cp.quad_form(x[k + 1, :] - x_ref[k, :], Q * kf) + cp.quad_form(u[k, :] - u_ref, R * kuf)
                constr += [x[k + 1, :] == Ad @ x[k, :] + Bd @ u[k, :] + Gd,
                           0 >= fx - mu * fz,
                           0 >= -fx - mu * fz,
                           0 >= fy - mu * fz,
                           0 >= -fy - mu * fz,
                           fz >= 0,
                           z <= 3,
                           z >= 0.3]

        constr += [x[0, :] == x_in]  # initial condition
        # constr += [x[-1, :] == x_ref[-1, :]]  # final condition
        # --- set up solver --- #
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(solver=cp.ECOS)  # , verbose=True)
        if u.value is None:
            raise Exception("\n *** QP FAILED *** \n")

        u = u.value[0, :]
        # print(u)
        # breakpoint()
        return u
