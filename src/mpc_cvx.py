"""
Copyright (C) 2021-2022 Benjamin Bokser
"""

import numpy as np
import cvxpy as cp
from scipy.linalg import expm


class Mpc:

    def __init__(self, t, A, B, N, m, g, mu, **kwargs):
        self.t = t  # sampling time (s)
        self.A = A
        self.B = B
        self.N = N  # prediction horizon
        self.m = m  # kg
        self.mu = mu  # coefficient of friction
        self.g = g

    def mpcontrol(self, X_in, X_ref):
        N = self.N
        t = self.t
        m = self.m
        mu = self.mu
        A = self.A
        B = self.B
        n_x = np.shape(self.A)[1]
        n_u = np.shape(self.B)[1]
        X = cp.Variable((N+1, n_x))
        U = cp.Variable((N, n_u))
        AB = np.vstack((np.hstack((A, B)), np.zeros((n_u, n_x+n_u))))
        M = expm(AB*t)
        Ad = M[0:n_x, 0:n_x]
        Bd = M[0:n_x, n_x:n_x+n_u]
        Q = np.eye(n_x)
        R = np.eye(n_u)
        cost = 0
        constr = []
        U_ref = np.zeros(n_u)
        # --- calculate cost & constraints --- #
        if n_x == 5:
            np.fill_diagonal(Q, [1, 1, 0.01, 0.01, 0])
            np.fill_diagonal(R, [0.01, 0.01])
            for k in range(0, N):
                kf = 3 if k == N - 1 else 1  # terminal cost
                kuf = 0 if k == N - 1 else 1  # terminal cost
                z = X[k, 1]
                fx = U[k, 0]
                fz = U[k, 1]
                if ((k + 1) % 2) == 0:  # even
                    U_ref[-1] = 0  # m * self.g * 2
                    cost += cp.quad_form(X[k + 1, :] - X_ref[k, :], Q * kf) + cp.quad_form(U[k, :] - U_ref, R * kuf)
                    constr += [X[k + 1, :] == Ad @ X[k, :] + Bd @ U[k, :],
                               0 == fx,
                               0 == fz]
                               #z >= 0]
                else:  # odd
                    U_ref[-1] = m * self.g * 2
                    cost += cp.quad_form(X[k + 1, :] - X_ref[k, :], Q * kf) + cp.quad_form(U[k, :] - U_ref, R * kuf)
                    constr += [X[k + 1, :] == Ad @ X[k, :] + Bd @ U[k, :],
                               0 >= fx - mu * fz,
                               0 >= -(fx + mu * fz),
                               0 >= fz,
                               z >= 0]
        elif n_x == 7:
            np.fill_diagonal(Q, [1., 1., 1., 0.01, 0.01, 0.01, 0.])
            np.fill_diagonal(R, [0., 0., 0.])
            for k in range(0, N):
                kf = 10 if k == N - 1 else 1  # terminal cost
                kuf = 0 if k == N - 1 else 1  # terminal cost
                z = X[k, 2]
                fx = U[k, 0]
                fy = U[k, 1]
                fz = U[k, 2]
                if ((k + 1) % 2) == 0:  # even
                    U_ref[-1] = 0
                    cost += cp.quad_form(X[k + 1, :] - X_ref[k, :], Q * kf) + cp.quad_form(U[k, :] - U_ref, R * kuf)
                    constr += [X[k + 1, :] == Ad @ X[k, :] + Bd @ U[k, :],
                               0 == fx,
                               0 == fy,
                               0 == fz]
                else:  # odd
                    U_ref[-1] = m * self.g * 2
                    cost += cp.quad_form(X[k + 1, :] - X_ref[k, :], Q * kf) + cp.quad_form(U[k, :] - U_ref, R * kuf)
                    constr += [X[k + 1, :] == Ad @ X[k, :] + Bd @ U[k, :],
                               0 >= fx - mu * fz,
                               0 >= -fx - mu * fz,
                               0 >= fy - mu * fz,
                               0 >= -fy - mu * fz,
                               fz >= 0,
                               z <= 3,
                               z >= 0]
        constr += [X[0, :] == X_in, X[N, :] == X_ref[-1, :]]  # initial and final condition
        # --- set up solver --- #
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(solver=cp.ECOS)  #, verbose=True)
        u = U.value
        x = X.value
        if u is None:
            raise Exception("\n *** QP FAILED *** \n")
        # print(u)
        # breakpoint()
        return u, x
