"""
Copyright (C) 2021-2022 Benjamin Bokser
"""

import numpy as np
import cvxpy as cp
from utils import rz, hat
from scipy.linalg import expm


class Mpc:

    def __init__(self, n_x, n_u, t, N, m, g, mu, **kwargs):
        self.t = t  # sampling time (s)
        self.N = N  # prediction horizon
        self.m = m  # kg
        self.g = g
        self.mu = mu
        self.n_x = n_x
        self.n_u = n_u

    def mpcontrol(self, x_in, x_ref_in, Ad, Bd, Gk, C):
        x_ref = x_ref_in
        N = self.N
        m = self.m
        g = self.g
        mu = self.mu
        n_x = self.n_x
        n_u = self.n_u
        # print(x_in - x_ref_in[0, :])
        x = cp.Variable((N+1, n_x))
        u = cp.Variable((N, n_u))

        Q = np.eye(n_x)
        np.fill_diagonal(Q, [10., 10., 2., 1., 1., 2., 1., 1., 1., 1., 1., 2.])
        R = np.eye(n_u)
        np.fill_diagonal(R, [0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
        u_ref = np.zeros(n_u)
        # --- calculate cost & constraints --- #
        cost = 0
        constr = []
        for k in range(0, N):
            Ak = Ad[k, :, :]
            Bk = Bd[k, :, :]
            kf = 100 if k == N - 1 else 1  # terminal cost
            kuf = 0 if k == N - 1 else 1  # terminal cost
            z = x[k, 2]
            fx = u[k, 0]
            fy = u[k, 1]
            fz = u[k, 2]
            taux = u[k, 3]
            tauy = u[k, 4]
            tauz = u[k, 5]

            constr += [taux <= 20,
                       taux >= -20,
                       tauy <= 20,
                       tauy >= -20,
                       tauz <= 4,
                       tauz >= -4]

            if C[k] == 0:  # even
                u_ref[2] = 0
                cost += cp.quad_form(x[k + 1, :] - x_ref[k, :], Q * kf) + cp.quad_form(u[k, :] - u_ref, R * kuf)
                constr += [x[k + 1, :] == Ak @ x[k, :] + Bk @ u[k, :] + Gk,
                           0 == fx,
                           0 == fy,
                           0 == fz]
            else:  # odd
                u_ref[2] = m * g * 2
                cost += cp.quad_form(x[k + 1, :] - x_ref[k, :], Q * kf) + cp.quad_form(u[k, :] - u_ref, R * kuf)
                constr += [x[k + 1, :] == Ak @ x[k, :] + Bk @ u[k, :] + Gk,
                           0 >= fx - mu * fz,
                           0 >= -fx - mu * fz,
                           0 >= fy - mu * fz,
                           0 >= -fy - mu * fz,
                           fz >= 0,
                           fz <= m * g * 4,  # TODO: Calculate max vertical force
                           z >= 0.1] # ,  # body frame fy = 0
                           # np.array([0, 1, 0]) @ rz_phi @ x[k, 6:9].T == 0]  # body frame y velocity should be zero
                           # z <= 3]

        constr += [x[0, :] == x_in]  # initial condition
        # constr += [x[-1, :] == x_ref[-1, :]]  # final condition
        # --- set up solver --- #
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(solver=cp.OSQP)  # , verbose=True)
        if u.value is None:
            raise Exception("\n *** QP FAILED *** \n")

        u = u.value  # [0, :]
        # print(u)
        # breakpoint()
        return u
