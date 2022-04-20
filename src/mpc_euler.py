"""
Copyright (C) 2021-2022 Benjamin Bokser
"""

import numpy as np
import cvxpy as cp
from utils import rz, quat2euler


class Mpc:

    def __init__(self, X_0, t, N, J, rhat, m, g, mu, **kwargs):
        n_x = 12  # number of states
        n_u = 6  # number of controls
        self.t = t  # sampling time (s)
        self.N = N  # prediction horizon
        self.J = J
        self.rhat = rhat
        self.m = m  # kg
        self.g = g
        self.mu = mu
        phi = quat2euler(X_0[3:7])[2]  # extract z-axis euler angle
        A = np.zeros((n_x, n_x))
        A[0:3, 6:9] = np.eye(3)
        A[3:6, 9:13] = rz(phi)
        B = np.zeros((n_x, n_u))
        B[6:9, 0:3] = np.eye(3) / self.m
        J_w_inv = rz(phi) @ np.linalg.inv(self.J) @ rz(phi).T
        B[9:12, 0:3] = J_w_inv @ rhat
        B[9:12, 3:6] = J_w_inv
        G = np.zeros(n_x)
        G[8] = -self.g
        self.A = A
        self.B = B
        self.G = G
        self.n_x = n_x
        self.n_u = n_u

    def mpcontrol(self, x_in, x_ref, C):
        N = self.N
        t = self.t
        m = self.m
        g = self.g
        mu = self.mu
        A = self.A
        B = self.B
        G = self.G
        J = self.J
        rhat = self.rhat
        n_x = self.n_x
        n_u = self.n_u
        x = cp.Variable((N+1, n_x))
        u = cp.Variable((N, n_u))
        Q = np.eye(n_x)
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

            A[3:6, 9:13] = rz(x[k, 5])
            J_w_inv = rz(x[k, 5]) @ np.linalg.inv(J) @ rz(x[k, 5]).T
            B[9:12, 0:3] = J_w_inv @ rhat
            B[9:12, 6:9] = J_w_inv
            '''
            Ak = np.eye(n_x) + A * t  # first order Euler integration
            Bk = B * t
            Gk = G * t
            '''
            A_bar = np.hstack(A, B, G)
            I_bar = np.hstack(np.eye(n_x), np.zeros((n_x, n_u + 1)))
            print("A_bar = ", np.shape(A_bar), " I_bar = ", np.shape(I_bar))
            M = I_bar + A_bar * t + 0.5 * (t**2) * A_bar @ A_bar
            Ak = M[0:n_x, :]
            Bk = M[n_x:n_x+n_u, :]
            Gk = M[-1, :]

            if C[k] == 0:  # even
                u_ref[-1] = 0
                cost += cp.quad_form(x[k + 1, :] - x_ref[k, :], Q * kf) + cp.quad_form(u[k, :] - u_ref, R * kuf)
                constr += [x[k + 1, :] == Ak @ x[k, :] + Bk @ u[k, :] + Gk,
                           0 == fx,
                           0 == fy,
                           0 == fz]
            else:  # odd
                u_ref[-1] = m * g * 2
                cost += cp.quad_form(x[k + 1, :] - x_ref[k, :], Q * kf) + cp.quad_form(u[k, :] - u_ref, R * kuf)
                constr += [x[k + 1, :] == Ak @ x[k, :] + Bk @ u[k, :] + Gk,
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
