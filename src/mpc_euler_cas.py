"""
Copyright (C) 2020-2021 Benjamin Bokser
"""

import numpy as np
import casadi as cs
import itertools

from utils import rz, quat2euler


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

        A = cs.SX.zeros(n_x, n_x)
        A[0:3, 6:9] = np.eye(3)
        # phi = quat2euler(X_0[3:7])[2]  # extract z-axis euler angle
        # rz_phi = rz(phi)
        # A[3:6, 9:] = rz_phi
        B = cs.SX.zeros(n_x, n_u)
        B[6:9, 0:3] = np.eye(3) / self.m
        # J_w_inv = rz_phi @ Jinv @ rz_phi.T
        # B[9:12, 0:3] = J_w_inv @ rhat
        # B[9:12, 3:6] = J_w_inv
        G = cs.SX.zeros(n_x, 1)
        G[8] = -self.g

        C_in = cs.SX.sym('C', n_x, N)
        rz_phi_in = cs.SX.sym('rz_phi', n_x, 3)
        x_p = cs.SX.sym('x_p', n_x, (N + 1))  # params: x_in, x_ref
        x = cs.SX.sym('x', n_x, (N + 1))  # represents the states over the opt problem.
        u = cs.SX.sym('u', n_u, N)  # decision variables, control action matrix

        C = C_in[0, :]  # awkwardly pull C out of larger array (rest should be zeros)

        rz_phi = rz_phi_in[0:3, :].T  # awkwardly pull rz_phi out of larger array (rest should be zeros)
        A[3:6, 9:] = rz_phi
        J_w_inv = rz_phi @ Jinv @ rz_phi.T
        B[9:12, 0:3] = J_w_inv @ rhat
        B[9:12, 3:] = J_w_inv
        A_bar = cs.horzcat(cs.horzcat(A, B), G)
        A_bar = cs.vertcat(A_bar, np.zeros((n_u + 1, n_x + n_u + 1)))
        I_bar = np.eye(n_x + n_u + 1)
        M = I_bar + A_bar * t + 0.5 * (t ** 2) * A_bar @ A_bar
        Ad = M[0:n_x, 0:n_x]
        Bd = M[0:n_x, n_x:n_x + n_u]
        Gd = M[0:n_x, -1]

        obj = 0
        constr_dyn = []
        constr_fricx1 = []
        constr_fricx2 = []
        constr_fricy1 = []
        constr_fricy2 = []
        Q = np.eye(n_x)
        R = np.eye(n_u)*0.01

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
            dyn = cs.mtimes(Ad, xk) + cs.mtimes(Bd, uk) + Gd
            constr_dyn = cs.vertcat(constr_dyn, x[:, k + 1] - dyn)  # compute constraints
            constr_fricx1 = cs.vertcat(constr_fricx1, (uk[0] - mu * uk[2]) * C[k])   # fx - mu*fz
            constr_fricx2 = cs.vertcat(constr_fricx2, (-uk[0] - mu * uk[2]) * C[k])  # fx - mu*fz
            constr_fricy1 = cs.vertcat(constr_fricx1, (uk[1] - mu * uk[2]) * C[k])  # fx - mu*fz
            constr_fricy2 = cs.vertcat(constr_fricx1, (-uk[1] - mu * uk[2]) * C[k])  # fx - mu*fz

        # append them
        constr = []
        constr = cs.vertcat(constr, constr_init)
        constr = cs.vertcat(constr, constr_dyn)
        constr = cs.vertcat(constr, constr_fricx1)
        constr = cs.vertcat(constr, constr_fricx2)
        constr = cs.vertcat(constr, constr_fricy1)
        constr = cs.vertcat(constr, constr_fricy2)

        opt_variables = cs.vertcat(cs.reshape(x, n_x * (N + 1), 1), cs.reshape(u, n_u * N, 1))
        params = cs.vertcat(x_p.T, rz_phi_in.T, C_in.T)
        qp = {'x': opt_variables, 'f': obj, 'g': constr, 'p': params}
        opts = {'print_time': 0, 'error_on_fail': 0, 'printLevel': "none", 'boundTolerance': 1e-6,
                'terminationTolerance': 1e-6}
        self.solver = cs.qpsol('S', 'qpoases', qp, opts)

        c_length = np.shape(constr)[0]
        o_length = np.shape(opt_variables)[0]

        lbg = list(itertools.repeat(-1e10, c_length))  # inequality constraints: big enough to act like infinity
        lbg[0:(N + 1)] = itertools.repeat(0, N + 1)  # IC + dynamics equality constraint
        ubg = list(itertools.repeat(0, c_length))  # inequality constraints

        # constraints for optimization variables
        lbx = list(itertools.repeat(-1e10, o_length))  # input inequality constraints
        ubx = list(itertools.repeat(1e10, o_length))  # input inequality constraints

        self.n_x = n_x
        self.n_u = n_u
        self.ubg = ubg
        self.lbg = lbg
        self.ubx = ubx
        self.lbx = lbx

    def mpcontrol(self, x_in, x_ref_in, C):
        N = self.N
        n_x = self.n_x
        n_u = self.n_u
        ubg = self.ubg
        lbg = self.lbg
        ubx = self.ubx
        lbx = self.lbx

        ubfx = ([200] * N) * C
        ubfy = ([200] * N) * C
        ubfz = ([200] * N) * C
        lbfx = ([-200] * N) * C
        lbfy = ([-200] * N) * C
        lbfz = ([0] * N) * C

        ubx[(n_x * (N + 1) + 0)::n_u] = ubfx  # upper bound on all f_x
        ubx[(n_x * (N + 1) + 1)::n_u] = ubfy  # upper bound on all f_y
        ubx[(n_x * (N + 1) + 2)::n_u] = ubfz  # upper bound on all f_z

        lbx[(n_x * (N + 1) + 0)::n_u] = lbfx  # lower bound on all f_x
        lbx[(n_x * (N + 1) + 1)::n_u] = lbfy  # lower bound on all f_y
        lbx[(n_x * (N + 1) + 2)::n_u] = lbfz  # lower bound on all f_z

        u0 = np.zeros((N, n_u))  # six control inputs
        x0_init = np.tile(x_in, (1, N + 1)).T  # initialization of the state's decision variables

        C_in = cs.vertcat(C.reshape(1, -1), np.zeros((n_x-1, N)))
        rz_phi_in = cs.horzcat(rz(x_in[5]), np.zeros((3, n_x-3)))
        parameters = cs.horzcat(x_in, x_ref_in.T, rz_phi_in.T, C_in).T  # set values of parameters vector
        # init value of optimization variables
        x0 = cs.vertcat(np.reshape(x0_init.T, (n_x * (N + 1), 1)), np.reshape(u0.T, (n_u * N, 1)))

        sol = self.solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=parameters)

        solu = np.array(sol['x'][n_x * (N + 1):])
        # u = np.reshape(solu.T, (n_u, N)).T  # get controls from the solution
        u = np.reshape(solu.T, (N, n_u)).T  # get controls from the solution
        # ss_error = np.linalg.norm(x0 - x_ref)  # defaults to Euclidean norm
        # print("ss_error = ", ss_error)

        return u[:, 0]
