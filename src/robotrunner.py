"""
Copyright (C) 2020-2022 Benjamin Bokser
"""
import plots
import mpc_cvx

import numpy as np
import copy
from utils import H, T, hat, L, R, rz, quat2euler, convert
# from scipy.linalg import expm
import itertools

np.set_printoptions(suppress=True, linewidth=np.nan)


class Runner:
    def __init__(self, dyn='euler', dt=1e-3):
        self.dyn = dyn
        self.dt = dt
        self.total_run = 5000
        self.tol = 1e-3  # desired mpc tolerance
        self.m = 7.5  # mass of the robot, kg
        self.J = np.array([[76148072.89, 70089.52, 2067970.36],
                           [70089.52, 45477183.53, -87045.58],
                           [2067970.36, -87045.58, 76287220.47]])*1000  # g/mm2 to kg/m2
        self.r = np.array([0.02201854, 6.80044366, 0.97499173]) * 1000  # mm to m
        rhat = hat(self.r)
        self.g = 9.807  # gravitational acceleration, m/s2
        self.t_p = 0.8  # gait period, seconds
        self.phi_switch = 0.5  # switching phase, must be between 0 and 1. Percentage of gait spent in contact.
        self.N = 20  # mpc prediction horizon length (mpc steps)  # TODO: Modify
        self.mpc_dt = 0.05  # mpc sampling time (s), needs to be a factor of N
        self.mpc_factor = int(self.mpc_dt / self.dt)  # mpc sampling time (timesteps), repeat mpc every x timesteps
        self.N_time = self.N * self.mpc_dt  # mpc horizon time
        self.N_k = self.N * self.mpc_factor  # total mpc prediction horizon length (timesteps)
        n_x = 12+1  # number of states + gravity
        n_u = 6  # number of controls
        # simulator uses SE(3) states! (X)
        # mpc uses euler-angle based states! (x)
        # need to convert between these carefully. Pay attn to X vs x !!!
        self.X_0 = np.array([0, 0, 0.7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # in rqvw form!!!
        self.X_f = np.hstack([2, 2, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).T  # desired final state
        phi = quat2euler(self.X_0[3:7])[2]  # extract z-axis euler angle
        self.A = np.zeros((n_x, n_x))
        self.A[0:3, 6:9] = np.eye(3)
        self.A[3:6, 9:13] = rz(phi)
        self.B = np.zeros((n_x, n_u))
        self.B[6:9, 0:3] = np.eye(3) / self.m
        J_w_inv = rz(phi) @ np.linalg.inv(self.J) @ rz(phi).T
        self.B[9:12, 0:3] = J_w_inv @ rhat
        self.B[9:12, 6:9] = J_w_inv

        mu = 0.3  # coeff of friction
        self.mpc = mpc_cvx.Mpc(t=self.mpc_dt, N=self.N, A=self.A, B=self.B, J=self.J, rhat=rhat,
                               m=self.m, g=self.g, mu=mu)
        self.n_x = n_x
        self.n_u = n_u

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
        U = np.zeros(self.n_u)
        pf_ref = np.zeros(self.n_u)
        j = int(self.mpc_factor)
        f_pred_hist = np.zeros((total, self.n_u))
        p_pred_hist = np.zeros((total, self.n_u))
        for k in range(0, self.total_run):
            t = t + self.dt

            s = self.gait_scheduler(t, t0)

            if mpc_counter == mpc_factor:  # check if it's time to restart the mpc
                mpc_counter = 0  # restart the mpc counter
                C = self.gait_map(t, t0)
                x_in = convert(X_traj[k, :])  # convert to mpc states
                x_ref = self.path_plan(x_in=x_in)
                x_refN = x_ref[::int(mpc_factor)]
                U = self.mpc.mpcontrol(x_in=x_in, x_ref=x_refN, C=C)

            mpc_counter += 1
            f_hist[k, :] = U * s  # take first timestep

            s_hist[k] = s
            X_traj[k + 1, :] = self.rk4_normalized(xk=X_traj[k, :], uk=f_hist[k, :])

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

    def gait_map(self, ts, t0):
        # generate vector of scheduled contact states over the mpc's prediction horizon
        C = np.zeros(self.N + 1)
        for k in range(0, (self.N + 1)):
            C[k] = self.gait_scheduler(t=ts, t0=t0)
            ts += self.mpc_dt
        return C

    def path_plan(self, x_in):
        # Path planner--generate reference trajectory in MPC state space!!!
        dt = self.dt
        xf = convert(self.X_f)
        size_mpc = int(self.mpc_factor * self.N)  # length of MPC horizon in s TODO: Perhaps N should vary wrt time?
        t_ref = int(np.minimum(size_mpc, np.linalg.norm(xf[0:2] - x_in[0:2]) * 1000))
        x_ref = np.linspace(start=x_in, stop=xf, num=t_ref)  # interpolate positions
        # interpolate linear velocities
        x_ref[:-1, 6] = [(x_ref[i + 1, 0] - x_ref[i, 0]) / dt for i in range(0, np.shape(x_ref)[0] - 1)]
        x_ref[:-1, 7] = [(x_ref[i + 1, 1] - x_ref[i, 1]) / dt for i in range(0, np.shape(x_ref)[0] - 1)]
        x_ref[:-1, 8] = [(x_ref[i + 1, 2] - x_ref[i, 2]) / dt for i in range(0, np.shape(x_ref)[0] - 1)]

        if (size_mpc - t_ref) == 0:
            pass
        elif t_ref == 0:
            x_ref = np.array(list(itertools.repeat(xf, int(size_mpc))))
        else:
            x_ref = np.vstack((x_ref, list(itertools.repeat(xf, int(size_mpc - t_ref)))))

        return x_ref
