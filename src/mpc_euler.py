"""
Copyright (C) 2021-2022 Benjamin Bokser
"""

import numpy as np
import cvxpy as cp
from utils import rz, hat
from scipy.linalg import expm


class Mpc:

    def __init__(self, t, N, Jinv, rh, m, g, mu, **kwargs):
        n_x = 12  # number of states
        n_u = 6  # number of controls
        self.t = t  # sampling time (s)
        self.N = N  # prediction horizon
        self.Jinv = Jinv
        self.rh = rh
        self.m = m  # kg
        self.g = g
        self.mu = mu
        A = np.zeros((n_x, n_x))
        A[0:3, 6:9] = np.eye(3)
        B = np.zeros((n_x, n_u))
        B[6:9, 0:3] = np.eye(3) / self.m
        G = np.zeros((n_x, 1))
        G[8, :] = -self.g
        self.A = A
        self.B = B
        self.G = G
        self.n_x = n_x
        self.n_u = n_u

    def mpcontrol(self, x_in, x_ref_in, rf, C):
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
        rh = self.rh
        n_x = self.n_x
        n_u = self.n_u

        x = cp.Variable((N+1, n_x))
        u = cp.Variable((N, n_u))
        # TODO: Make Rz_phi a param instead of setting up the problem on every run
        rz_phi = rz(x_in[5])
        rhat = hat(rh + rz_phi.T @ rf)
        A[3:6, 9:] = rz_phi
        J_w_inv = rz_phi @ Jinv @ rz_phi.T  # world frame Jinv
        # J_w_inv = np.linalg.inv(rz_phi @ J @ rz_phi.T)  # world frame Jinv
        B[9:12, 0:3] = J_w_inv @ rhat
        B[9:12, 3:] = J_w_inv @ rz_phi.T
        A_bar = np.vstack((np.hstack((A, B, G)), np.zeros((n_u + 1, n_x + n_u + 1))))
        I_bar = np.eye(n_x + n_u + 1)
        M = I_bar + A_bar * t  # + 0.5 * (t ** 2) * A_bar @ A_bar
        # M = expm(A_bar * t)
        Ad = M[0:n_x, 0:n_x]
        Bd = M[0:n_x, n_x:n_x + n_u]
        Gd = M[0:n_x, -1]

        Q = np.eye(n_x)
        np.fill_diagonal(Q, [1., 1., 0.5, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        R = np.eye(n_u)*0
        # np.fill_diagonal(R, [0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        u_ref = np.zeros(n_u)
        # --- calculate cost & constraints --- #
        cost = 0
        constr = []
        for k in range(0, N):
            kf = 10 if k == N - 1 else 1  # terminal cost
            kuf = 0 if k == N - 1 else 1  # terminal cost
            z = x[k, 2]
            fx = u[k, 0]
            fy = u[k, 1]
            fz = u[k, 2]
            taux = u[k, 3]
            tauy = u[k, 4]
            tauz = u[k, 5]
            '''
            constr += [taux <= 20,
                       taux >= -20,
                       tauy <= 20,
                       tauy >= -20,
                       tauz <= 4,
                       tauz >= -4]
            '''
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
                           fz >= 0,  # TODO: Calculate max vertical force
                           z >= 0.3]
                           #z <= 3]

        constr += [x[0, :] == x_in]  # initial condition
        # constr += [x[-1, :] == x_ref[-1, :]]  # final condition
        # --- set up solver --- #
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(solver=cp.OSQP)  # , verbose=True)
        if u.value is None:
            raise Exception("\n *** QP FAILED *** \n")

        u = u.value[0, :]
        # print(u)
        # breakpoint()
        return u
