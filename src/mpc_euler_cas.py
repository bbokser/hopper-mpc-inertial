"""
Copyright (C) 2020-2021 Benjamin Bokser
"""

import numpy as np
import casadi as cs
import itertools

from utils import rz, quat2euler


def rz_c(phi):  # casadi version
    Rz = cs.SX.eye(3)
    Rz[0, 0] = cs.cos(phi)
    Rz[0, 1] = cs.sin(phi)
    Rz[1, 0] = -cs.sin(phi)
    Rz[1, 1] = cs.cos(phi)
    return Rz


class Mpc:

    def __init__(self, X_0, t, N, J, rhat, m, g, mu, **kwargs):
        n_x = 12  # number of states
        n_u = 6  # number of controls
        self.t = t  # sampling time (s)
        self.N = N  # prediction horizon
        self.Jinv = np.linalg.inv(J)
        self.rhat = rhat
        self.m = m  # kg
        self.g = g
        self.mu = mu
        phi = quat2euler(X_0[3:7])[2]  # extract z-axis euler angle
        A = cs.SX.zeros(n_x, n_x)
        A[0:3, 6:9] = np.eye(3)
        rz_phi = rz(phi)
        A[3:6, 9:] = rz_phi
        B = cs.SX.zeros(n_x, n_u)
        B[6:9, 0:3] = np.eye(3) / self.m
        J_w_inv = rz_phi @ self.Jinv @ rz_phi.T
        B[9:12, 0:3] = J_w_inv @ rhat
        B[9:12, 3:6] = J_w_inv
        G = cs.SX.zeros(n_x, 1)
        G[8] = -self.g
        self.A = A
        self.B = B
        self.G = G
        self.n_x = n_x
        self.n_u = n_u

    def mpcontrol(self, x_in, x_ref_in, C):
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

        x_p = cs.SX.sym('x_p', n_x, (N + 1))  # params: x_in and x_ref
        x = cs.SX.sym('x', n_x, (N + 1))  # represents the states over the opt problem.
        u = cs.SX.sym('u', n_u, N)  # decision variables, control action matrix

        obj = 0
        constr_dyn = []
        constr_fricx1 = []
        constr_fricx2 = []
        constr_fricy1 = []
        constr_fricy2 = []
        Q = np.eye(n_x)
        R = np.eye(n_u)

        constr_init = x[:, 0] - x_p[:, 0]  # initial condition constraints
        x_ref = x_p[:, 1:]  # extract x_ref from x_p
        # --- compute objective and constraints --- #
        for k in range(0, N):
            xk = x[:, k]  # state
            x_refk = x_ref[:, k]
            uk = u[:, k]  # control action
            u_refk = m * g * 2
            # calculate objective
            obj = obj + cs.mtimes(cs.mtimes((xk - x_refk).T, Q), xk - x_refk) \
                + cs.mtimes(cs.mtimes((uk - u_refk).T, R), uk - u_refk)

            rz_phi = rz_c(xk[5])
            A[3:6, 9:] = rz_phi
            J_w_inv = rz_phi @ Jinv @ rz_phi.T
            B[9:12, 0:3] = J_w_inv @ rhat
            B[9:12, 3:] = J_w_inv
            A_bar = cs.horzcat(cs.horzcat(A, B), G)
            A_bar = cs.vertcat(A_bar, np.zeros((n_u+1, n_x+n_u+1)))
            I_bar = np.eye(n_x+n_u+1)

            M = I_bar + A_bar * t + 0.5 * (t ** 2) * A_bar @ A_bar
            Ak = M[0:n_x, 0:n_x]
            Bk = M[0:n_x, n_x:n_x + n_u]
            Gk = M[0:n_x, -1]
            dyn = cs.mtimes(Ak, xk) + cs.mtimes(Bk, uk) + Gk
            constr_dyn = cs.vertcat(constr_dyn, x[:, k + 1] - dyn)  # compute constraints
            if C[k] == 1:
                constr_fricx1 = cs.vertcat(constr_fricx1, uk[0] - mu * uk[2])  # fx - mu*fz
                constr_fricx2 = cs.vertcat(constr_fricx2, -uk[0] - mu * uk[2])  # fx - mu*fz
                constr_fricy1 = cs.vertcat(constr_fricx1, uk[1] - mu * uk[2])  # fx - mu*fz
                constr_fricy2 = cs.vertcat(constr_fricx1, -uk[1] - mu * uk[2])  # fx - mu*fz

        # append them
        constr = []
        constr = cs.vertcat(constr, constr_init)
        constr = cs.vertcat(constr, constr_dyn)
        constr = cs.vertcat(constr, constr_fricx1)
        constr = cs.vertcat(constr, constr_fricx2)
        constr = cs.vertcat(constr, constr_fricy1)
        constr = cs.vertcat(constr, constr_fricy2)

        opt_variables = cs.vertcat(cs.reshape(x, n_x * (N + 1), 1), cs.reshape(u, n_u * N, 1))
        qp = {'x': opt_variables, 'f': obj, 'g': constr, 'p': x_p}
        opts = {'print_time': 0, 'error_on_fail': 0, 'printLevel': "none", 'boundTolerance': 1e-6,
                'terminationTolerance': 1e-6}
        solver = cs.qpsol('S', 'qpoases', qp, opts)

        c_length = np.shape(constr)[0]
        o_length = np.shape(opt_variables)[0]

        lbg = list(itertools.repeat(-1e10, c_length))  # inequality constraints: big enough to act like infinity
        lbg[0:(N + 1)] = itertools.repeat(0, N + 1)  # IC + dynamics equality constraint
        ubg = list(itertools.repeat(0, c_length))  # inequality constraints

        # constraints for optimization variables
        lbx = list(itertools.repeat(-1e10, o_length))  # input inequality constraints
        ubx = list(itertools.repeat(1e10, o_length))  # input inequality constraints

        ubfx, ubfy, ubfz, lbfx, lbfy, lbfz = ([0] * N,) * 6  # [0 for i in range(N)]
        for k in range(0, N):
            if C[k] == 1:  # only calculate leg forces if leg is in contact
                ubfx[k] = 200
                ubfy[k] = 200
                ubfz[k] = 200
                lbfx[k] = -200
                lbfy[k] = -200

        ubx[(n_x * (N + 1) + 0)::n_u] = ubfx  # upper bound on all f_x
        ubx[(n_x * (N + 1) + 1)::n_u] = ubfy  # upper bound on all f_y
        ubx[(n_x * (N + 1) + 2)::n_u] = ubfz  # upper bound on all f_z
        lbx[(n_x * (N + 1) + 0)::n_u] = lbfx  # lower bound on all f_x
        lbx[(n_x * (N + 1) + 1)::n_u] = lbfy  # lower bound on all f_y
        lbx[(n_x * (N + 1) + 2)::n_u] = lbfz  # lower bound on all f_z

        # --- setup is finished, now solve --- #
        u0 = np.zeros((N, n_u))  # six control inputs
        x0_init = np.tile(x_in, (1, N + 1)).T  # initialization of the state's decision variables

        # parameters and xin must be changed every timestep
        # parameters = cs.vertcat(np.reshape(x_in, (1, 12)), x_ref_in).T  # set values of parameters vector
        parameters = cs.horzcat(x_in, x_ref_in.T)  # set values of parameters vector
        # init value of optimization variables
        x0 = cs.vertcat(np.reshape(x0_init.T, (n_x * (N + 1), 1)), np.reshape(u0.T, (n_u * N, 1)))

        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=parameters)

        solu = np.array(sol['x'][n_x * (N + 1):])
        # u = np.reshape(solu.T, (n_u, N)).T  # get controls from the solution
        u = np.reshape(solu.T, (N, n_u)).T  # get controls from the solution
        # ss_error = np.linalg.norm(x0 - x_ref)  # defaults to Euclidean norm
        # print("ss_error = ", ss_error)

        return u[:, 0]
