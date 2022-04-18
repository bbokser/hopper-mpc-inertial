"""
Copyright (C) 2020-2022 Benjamin Bokser
"""
import plots
import mpc_cvx

import numpy as np
import copy
from scipy.linalg import expm
import itertools
np.set_printoptions(suppress=True, linewidth=np.nan)

H = np.zeros((4, 3))
H[1:4, 0:4] = np.eye(3)

T = np.zeros((4, 4))
np.fill_diagonal(T, [1.0, -1.0, -1.0, -1.0])


def projection(p0, v):
    # find point p projected onto ground plane from point p0 by vector v
    z = 0
    t = (z - p0[2]) / v[2]
    x = p0[0] + t * v[0]
    y = p0[1] + t * v[1]
    p = np.array([x, y, z])
    return p


def hat(w):
    # skew-symmetric
    return np.array([[0, - w[2], w[1]],
                     [w[2], 0, - w[0]],
                     [-w[1], w[0], 0]])


def L(Q):
    LQ = np.zeros((4, 4))
    LQ[0, 0] = Q[0]
    LQ[0, 1:4] = - np.transpose(Q[1:4])
    LQ[1:4, 0] = Q[1:4]
    LQ[1:4, 1:4] = Q[0] * np.eye(3) + hat(Q[1:4])
    return LQ


def R(Q):
    RQ = np.zeros((4, 4))
    RQ[0, 0] = Q[0]
    RQ[0, 1:4] = - np.transpose(Q[1:4])
    RQ[1:4, 0] = Q[1:4]
    RQ[1:4, 1:4] = Q[0] * np.eye(3) - hat(Q[1:4])
    return RQ


def rz_phi(Q_in):
    # linearized rotation matrix Rz(phi) using commanded yaw
    # phi_s = 2 * np.arcsin(Q_in[3])
    phi = quat2euler(Q_in)[2]  # extract z-axis euler angle
    Rz = np.array([[np.cos(phi), np.sin(phi),  0.0],
                   [-np.sin(phi), np.cos(phi), 0.0],
                   [0.0,         0.0,          1.0]])
    return Rz


def quat2euler(quat):
    w, x, y, z = quat
    y_sqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y_sqr)
    X = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y_sqr + z * z)
    Z = np.arctan2(t3, t4)

    result = np.zeros(3)
    result[0] = X
    result[1] = Y
    result[2] = Z

    return result


class Runner:
    def __init__(self, ctrl='mpc', dt=1e-3):
        self.ctrl = ctrl
        self.dt = dt
        self.total_run = 5000
        self.tol = 1e-3  # desired mpc tolerance
        self.m = 7.5  # mass of the robot, kg
        self.J = []  # TODO: Fill in
        self.r = []
        self.N = 10  # mpc horizon length
        self.g = 9.81  # gravitational acceleration, m/s2
        self.t_p = 1  # gait period, seconds
        self.phi_switch = 0.5  # switching phase, must be between 0 and 1. Percentage of gait spent in contact.
        # for now, mpc sampling time is equal to gait period
        self.mpc_dt = self.t_p * self.phi_switch  # mpc sampling time
        self.N_time = self.N*self.mpc_dt  # mpc horizon time
        n_x = 13  # number of states
        n_u = 6  # number of controls

        self.X_0 = np.array([0, 0, 0.7, 1, 0, 0, 0, 0, 0, 0, 0, 0, self.g])
        self.X_f = np.hstack([2, 2, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0, self.g]).T  # desired final state

        self.A = np.zeros((n_x, n_x))
        self.A[0:3, 6:9] = np.eye(3)
        self.A[3:6, 9:13] = rz_phi(self.X_0[3:7])
        self.A[8, :-1] = -1

        self.B = np.zeros((n_x, n_u))
        self.B[6:9, 0:3] = np.eye(3)*1/self.m
        J_w_inv = rz_phi(self.X_0[3:7]) @ np.linalg.inv(self.J) @ rz_phi(self.X_0[3:7]).T
        self.B[9:12, 0:3] = J_w_inv @ hat(self.r)
        self.B[9:12, 6:9] = J_w_inv

        mu = 0.3  # coeff of friction
        self.mpc = mpc_cvx.Mpc(t=self.mpc_dt, A=self.A, B=self.B, N=self.N, m=self.m, g=self.g, mu=mu)
        self.mpc_factor = self.mpc_dt * 2 / self.dt  # repeat mpc every x seconds

    def run(self):
        total = self.total_run + 1  # number of timesteps to plot
        t = 0  # time
        t0 = t  # starting time

        mpc_factor = self.mpc_factor  # repeat mpc every x seconds
        mpc_counter = copy.copy(mpc_factor)
        X_traj = np.zeros((total, self.n_x))
        X_traj[0, :] = self.X_0  # initial conditions
        f_hist = np.zeros((total, self.n_u))
        s_hist = np.zeros(total)
        U_pred = np.zeros((self.N, self.n_u))
        X_pred = np.zeros((self.N, self.n_x))
        pf_ref = np.zeros(self.n_u)
        j = int(self.mpc_factor)
        X_pred_hist = np.zeros((self.N+1, self.n_u))
        f_pred_hist = np.zeros((total, self.n_u))
        p_pred_hist = np.zeros((total, self.n_u))
        for k in range(0, self.total_run):
            t = t + self.dt

            s = self.gait_scheduler(t, t0)

            if self.ctrl == 'mpc':
                if mpc_counter == mpc_factor:  # check if it's time to restart the mpc
                    mpc_counter = 0  # restart the mpc counter
                    X_ref = self.path_plan(X_in=X_traj[k, :])
                    X_refN = X_ref[::int(self.mpc_dt / self.dt)]
                    U_pred, X_pred = self.mpc.mpcontrol(X_in=X_traj[k, :], X_ref=X_refN)
                    p_pred = (X_pred[2, 0:3]+(X_pred[2, 0:3]+X_pred[3, 0:3])/2)/2  # next pred body pos over next ftstep
                    f_pred = U_pred[2, :]  # next predicted foot force vector
                    p_pred_hist = np.vstack((p_pred_hist, p_pred))
                    f_pred_hist = np.vstack((f_pred_hist, 0.5*f_pred/np.sqrt(np.sum(f_pred**2))))
                    pf_ref = np.vstack((pf_ref, projection(p_pred, f_pred)))
                    X_pred_hist = np.dstack((X_pred_hist, X_pred[:, 0:self.n_u]))
                mpc_counter += 1
                f_hist[k, :] = U_pred[0, :]*s  # take first timestep

            else:  # Open loop traj opt, this will fail if total != mpc_factor
                if int(total/self.N) != mpc_factor:
                    print("ERROR: Incorrect settings", total/self.N, mpc_factor)
                if k == 0:
                    X_ref = self.path_plan(X_in=X_traj[k, :])
                    X_refN = X_ref[::int(self.mpc_factor)]  # self.traj_N(X_ref)
                    force_f, X_pred = self.mpc.mpcontrol(X_in=X_traj[k, :], X_ref=X_refN)
                    for i in range(0, self.N):
                        f_hist[int(i*j):int(i*j+j), :] = list(itertools.repeat(force_f[i, :], j))

            s_hist[k] = s
            X_traj[k+1, :] = self.rk4(xk=X_traj[k, :], uk=f_hist[k, :])
            # X_traj[k + 1, :] = self.dynamics_dt(X=X_traj[k, :], U=f_hist[k, :], t=self.dt)

        # print(X_traj[-1, :])
        # print(f_hist[4500, :])
        plots.fplot(total, p_hist=X_traj[:, 0:self.n_u], f_hist=f_hist, s_hist=s_hist)
        plots.posplot(p_ref=self.X_f[0:self.n_u], p_hist=X_traj[:, 0:self.n_u])
        plots.posfplot(p_ref=self.X_f[0:self.n_u], p_hist=X_traj[:, 0:self.n_u],
                       p_pred_hist=p_pred_hist, f_pred_hist=f_pred_hist, pf_hist=pf_ref)
        # plots.posplot(p_ref=self.X_f[0:self.n_u], p_hist=X_pred_hist[:, 0:self.n_u, 1], dims=self.dims)
        # plots.posplot_t(p_ref=self.X_ref[0:self.n_u], p_hist=X_traj[:, 0:2], total=total)

        return None

    def dynamics_ct(self, X, U):
        # SE(3) nonlinear dynamics
        # Unpack state vector
        m = self.m
        g = self.g
        J = self.J
        p = X[0:3]  # W frame
        q = X[3:7]  # B to N
        v = X[7:10]  # B frame
        w = X[10:13]  # B frame
        F = U[0:3]  # W frame
        tau = U[3:]  # W frame
        Q = L(q) @ R(q).T
        dp = H.T @ Q @ H @ v  # rotate v from body to world frame
        dq = 0.5 * L(q) * H * w
        Fgn = np.array([0, 0, -g]) * m  # Gravitational force in world frame
        Fgb = H.T @ Q.T @ H @ Fgn  # rotate Fgn from world frame to body frame
        Ft = F + Fgb  # total force
        dv = 1 / m * Ft - np.cross(w, v)
        dw = np.linalg.solve(J, tau - np.cross(w, J * w))
        dx = np.vstack(dp, dq, dv, dw)

        return dx

    def rk4_normalized(self, xk, uk):
        # RK4 integrator solves for new X
        dynamics = self.dynamics_ct
        h = self.dt
        f1 = dynamics(xk, uk)
        f2 = dynamics(xk + 0.5 * h * f1, uk)
        f3 = dynamics(xk + 0.5 * h * f2, uk)
        f4 = dynamics(xk + h * f3, uk)
        xn = xk + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
        xn[4:7] = xn[3:7] / np.linalg.norm(xn[3:7])  # normalize the quaternion term
        return xn

    def gait_scheduler(self, t, t0):
        phi = np.mod((t - t0) / self.t_p, 1)
        if phi > self.phi_switch:
            s = 0  # scheduled swing
        else:
            s = 1  # scheduled stance
        return s

    def path_plan(self, X_in):
        # Path planner--generate reference trajectory
        dt = self.dt
        size_mpc = int(self.mpc_factor*self.N)  # length of MPC horizon in s TODO: Perhaps N should vary wrt time?
        t_ref = int(np.minimum(size_mpc, np.linalg.norm(self.X_f[0:2] - X_in[0:2]) * 1000))
        X_ref = np.linspace(start=X_in, stop=self.X_f, num=t_ref)  # interpolate positions
        # interpolate velocities
        X_ref[:-1, 3] = [(X_ref[i + 1, 0] - X_ref[i, 0]) / dt for i in range(0, np.shape(X_ref)[0] - 1)]
        X_ref[:-1, 4] = [(X_ref[i + 1, 1] - X_ref[i, 1]) / dt for i in range(0, np.shape(X_ref)[0] - 1)]
        X_ref[:-1, 5] = [(X_ref[i + 1, 2] - X_ref[i, 2]) / dt for i in range(0, np.shape(X_ref)[0] - 1)]

        if (size_mpc - t_ref) == 0:
            pass
        elif t_ref == 0:
            X_ref = np.array(list(itertools.repeat(self.X_f, int(size_mpc))))
        else:
            X_ref = np.vstack((X_ref, list(itertools.repeat(self.X_f, int(size_mpc - t_ref)))))

        return X_ref



