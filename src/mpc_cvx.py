"""
Copyright (C) 2021-2022 Benjamin Bokser
"""

import numpy as np
import cvxpy as cp
from scipy.linalg import expm
from utils import rz


class Mpc:

    def __init__(self, t, N, A, B, J, rhat, m, g, mu, **kwargs):
        self.t = t  # sampling time (s)
        self.N = N  # prediction horizon
        self.A = A
        self.B = B
        self.J = J
        self.rhat = rhat
        self.m = m  # kg
        self.g = g
        self.mu = mu  # coefficient of friction

    def mpcontrol(self, x_in, x_ref, C):
        N = self.N
        t = self.t
        m = self.m
        g = self.g
        mu = self.mu
        A = self.A
        B = self.B
        J = self.J
        rhat = self.rhat
        n_x = np.shape(self.A)[1]
        n_u = np.shape(self.B)[1]
        x = cp.Variable((N+1, n_x-1))  # don't include gravity
        u = cp.Variable((N, n_u))
        Q = np.eye(n_x-1)
        R = np.eye(n_u)
        cost = 0
        constr = []
        u_ref = np.zeros(n_u)

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

            # A[0:3, 6:9] = np.eye(3)  # unnecessary; should already be this
            A[3:6, 9:13] = rz(x[5])
            # B[6:9, 0:3] = np.eye(3) / m  # unnecessary; should already be this
            J_w_inv = rz(x[5]) @ np.linalg.inv(J) @ rz(x[5]).T
            B[9:12, 0:3] = J_w_inv @ rhat
            B[9:12, 6:9] = J_w_inv
            AB = np.vstack((np.hstack((A, B)), np.zeros((n_u, n_x + n_u))))
            M = expm(AB * t)
            Ad = M[0:n_x - 1, 0:n_x - 1]
            Gd = M[0:n_x - 1, n_x - 1]  # gravity in the world frame
            Bd = M[0:n_x - 1, n_x:n_x + n_u]

            if C[k] == 0:  # even
                u_ref[-1] = 0
                cost += cp.quad_form(x[k + 1, :] - x_ref[k, :], Q * kf) + cp.quad_form(u[k, :] - u_ref, R * kuf)
                constr += [x[k + 1, :] == Ad @ x[k, :] + Bd @ u[k, :] + Gd,
                           0 == fx,
                           0 == fy,
                           0 == fz]
            else:  # odd
                u_ref[-1] = m * g * 2
                cost += cp.quad_form(x[k + 1, :] - x_ref[k, :], Q * kf) + cp.quad_form(u[k, :] - u_ref, R * kuf)
                constr += [x[k + 1, :] == Ad @ x[k, :] + Bd @ u[k, :] + Gd,
                           0 >= fx - mu * fz,
                           0 >= -fx - mu * fz,
                           0 >= fy - mu * fz,
                           0 >= -fy - mu * fz,
                           fz >= 0,
                           z <= 3,
                           z >= 0]
        constr += [x[0, :] == x_in, x[N, :] == x_ref[-1, :]]  # initial and final condition
        # --- set up solver --- #
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(solver=cp.ECOS)  #, verbose=True)
        u = u.value[0, :]
        if u is None:
            raise Exception("\n *** QP FAILED *** \n")
        # print(u)
        # breakpoint()
        return u
