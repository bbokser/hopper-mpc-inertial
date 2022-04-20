"""
Copyright (C) 2020-2021 Benjamin Bokser
"""

import numpy as np
import casadi as cs
import itertools

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
        '''
        p_x = cs.SX.sym('p_x')
        p_y = cs.SX.sym('p_y')
        p_z = cs.SX.sym('p_z')
        theta_x = cs.SX.sym('theta_x')
        theta_y = cs.SX.sym('theta_y')
        theta_z = cs.SX.sym('theta_z')
        dp_x = cs.SX.sym('dp_x')
        dp_y = cs.SX.sym('dp_y')
        dp_z = cs.SX.sym('dp_z')
        omega_x = cs.SX.sym('theta_x')
        omega_y = cs.SX.sym('theta_y')
        omega_z = cs.SX.sym('theta_z')
        states = [p_x, p_y, p_z,
                  theta_x, theta_y, theta_z,
                  dp_x, dp_y, dp_z,
                  omega_x, omega_y, omega_z]  # state vec x
        n_states = len(states)  # number of states

        f_x = cs.SX.sym('f_x')  # control force x
        f_y = cs.SX.sym('f_y')  # control force y
        f_z = cs.SX.sym('f_z')  # control force z
        tau_x = cs.SX.sym('tau_x')
        tau_y = cs.SX.sym('tau_y')
        tau_z = cs.SX.sym('tau_z')
        controls = [f_x, f_y, f_z, tau_x, tau_y, tau_z]
        n_controls = len(controls)  # number of controls
        '''
        x_p = cs.SX.sym('x_p', n_x, (N + 1))  # params: x_in and x_ref
        x = cs.SX.sym('x', n_x, (N + 1))  # represents the states over the opt problem.
        u = cs.SX.sym('u', n_u, N)  # decision variables, control action matrix

        obj = [None] * N  # preallocate objective function
        constr_dyn = [None] * N  # preallocate constraints
        constr_fricx1 = [None] * N  # preallocate constraints
        constr_fricx2 = [None] * N  # preallocate constraints
        constr_fricy1 = [None] * N  # preallocate constraints
        constr_fricy2 = [None] * N  # preallocate constraints
        Q = np.eye(n_x)
        R = np.eye(n_u)

        constr_init = x[:, 0] - x_p[:, 0]  # initial condition constraints
        x_ref = x_p[:, 1:]  # extract x_ref from x_p
        # compute objective and constraints
        for k in range(0, self.N):
            xk = x[:, k]  # state
            x_refk = x_ref[:, k]
            uk = u[:, k]  # control action
            u_refk = m * g * 2
            # calculate objective
            obj[k] = cs.mtimes(cs.mtimes((xk - x_refk).T, Q), xk - x_refk)\
                + cs.mtimes(cs.mtimes((uk - u_refk).T, R), uk - u_refk)

            A[3:6, 9:13] = rz(xk[5])
            J_w_inv = rz(xk[5]) @ np.linalg.inv(J) @ rz(xk[5]).T
            B[9:12, 0:3] = J_w_inv @ rhat
            B[9:12, 6:9] = J_w_inv
            A_bar = np.hstack(A, B, G)
            I_bar = np.hstack(np.eye(n_x), np.zeros((n_x, n_u + 1)))

            print("A_bar = ", np.shape(A_bar), " I_bar = ", np.shape(I_bar))

            M = I_bar + A_bar * t + 0.5 * (t ** 2) * A_bar @ A_bar
            Ak = M[0:n_x, :]
            Bk = M[n_x:n_x + n_u, :]
            Gk = M[-1, :]
            dyn = cs.mtimes(Ak, xk) + cs.mtimes(Bk, uk) + Gk
            constr_dyn[k] = x[:, k + 1] - dyn  # compute constraints
            constr_fricx1[k] = uk[0] - mu * uk[2]  # # fx - mu*fz
            constr_fricx2[k] = -uk[0] - mu * uk[2]  # # fx - mu*fz
            constr_fricy1[k] = uk[1] - mu * uk[2]  # # fx - mu*fz
            constr_fricy2[k] = -uk[1] - mu * uk[2]  # # fx - mu*fz

        # I guess this appends them??
        constr = constr_init + constr_dyn + constr_fricx1 + constr_fricx2 + constr_fricy1 + constr_fricy2
        opt_variables = cs.vertcat(cs.reshape(x, n_x * (N + 1), 1), cs.reshape(u, n_u * N, 1))
        qp = {'x': opt_variables, 'f': obj, 'g': constr, 'p': x_p}
        opts = {'print_time': 0, 'error_on_fail': 0, 'printLevel': "none", 'boundTolerance': 1e-6,
                'terminationTolerance': 1e-6}
        solver = cs.qpsol('S', 'qpoases', qp, opts)

        c_length = np.shape(constr)[0]
        o_length = np.shape(opt_variables)[0]

        lbg = list(itertools.repeat(-1e10, c_length))  # inequality constraints: big enough to act like infinity
        lbg[0:(self.N + 1)] = itertools.repeat(0, self.N + 1)  # IC + dynamics equality constraint
        ubg = list(itertools.repeat(0, c_length))  # inequality constraints

        # constraints for optimization variables
        lbx = list(itertools.repeat(-1e10, o_length))  # input inequality constraints
        ubx = list(itertools.repeat(1e10, o_length))  # input inequality constraints

        st_len = n_states * (self.N + 1)

        lbx[(st_len + 2)::3] = [0 for i in range(20)]  # lower bound on all f1z and f2z

        if c_l == 0:  # if left leg is not in contact... don't calculate output forces for that leg.
            ubx[(n_states * (self.N + 1))::6] = [0 for i in range(10)]  # upper bound on all f1x
            ubx[(n_states * (self.N + 1) + 1)::6] = [0 for i in range(10)]  # upper bound on all f1y
            lbx[(n_states * (self.N + 1))::6] = [0 for i in range(10)]  # lower bound on all f1x
            lbx[(n_states * (self.N + 1) + 1)::6] = [0 for i in range(10)]  # lower bound on all f1y
            ubx[(n_states * (self.N + 1) + 2)::6] = [0 for i in range(10)]  # upper bound on all f1z
        else:
            ubx[(n_states * (self.N + 1))::6] = [1 for i in range(10)]  # upper bound on all f1x
            ubx[(n_states * (self.N + 1) + 1)::6] = [1 for i in range(10)]  # upper bound on all f1y
            lbx[(n_states * (self.N + 1))::6] = [-1 for i in range(10)]  # lower bound on all f1x
            lbx[(n_states * (self.N + 1) + 1)::6] = [-1 for i in range(10)]  # lower bound on all f1y
            ubx[(n_states * (self.N + 1) + 2)::6] = [2.5 for i in range(10)]  # upper bound on all f1z

        # setup is finished, now solve-------------------------------------------------------------------------------- #

        u0 = np.zeros((self.N, n_controls))  # six control inputs
        X0 = np.matlib.repmat(x_in, 1, self.N + 1).T  # initialization of the state's decision variables

        # parameters and xin must be changed every timestep
        parameters = cs.vertcat(x_in, x_ref)  # set values of parameters vector
        # init value of optimization variables
        x0 = cs.vertcat(np.reshape(X0.T, (n_states * (self.N + 1), 1)),
                        np.reshape(u0.T, (n_controls * self.N, 1)))

        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=parameters)

        solu = np.array(sol['x'][n_states * (self.N + 1):])
        # u = np.reshape(solu.T, (n_controls, self.N)).T  # get controls from the solution
        u = np.reshape(solu.T, (self.N, n_controls)).T  # get controls from the solution
        # ss_error = np.linalg.norm(x0 - x_ref)  # defaults to Euclidean norm
        # print("ss_error = ", ss_error)

        return u[:, 0]
