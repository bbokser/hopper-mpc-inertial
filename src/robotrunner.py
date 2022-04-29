"""
Copyright (C) 2020-2022 Benjamin Bokser
"""
import plots
import mpc_euler_cas
import mpc_euler
from utils import H, L, R, convert

from tqdm import tqdm
import numpy as np
import copy
import sys
from scipy.signal import find_peaks

np.set_printoptions(suppress=True, linewidth=np.nan)


class Runner:
    def __init__(self, dt=1e-3, tool='cvxpy', dyn='euler', ctrl='closed', runtime=5000):
        self.dyn = dyn
        self.ctrl = ctrl
        self.dt = dt
        self.total_run = runtime
        # self.tol = 1e-3  # desired mpc tolerance
        self.m = 7.5  # mass of the robot, kg
        self.J = np.array([[76148072.89, 70089.52, 2067970.36],
                           [70089.52, 45477183.53, -87045.58],
                           [2067970.36, -87045.58, 76287220.47]])*(10**(-9))  # g/mm2 to kg/m2
        self.Jinv = np.linalg.inv(self.J)
        self.rh = np.array([0.02201854, 6.80044366, 0.97499173]) / 1000  # mm to m
        self.g = 9.807  # gravitational acceleration, m/s2
        self.t_p = 0.8  # 0.8 gait period, seconds
        self.phi_switch = 0.5  # 0.5  # switching phase, must be between 0 and 1. Percentage of gait spent in contact.
        self.N = 20  # mpc prediction horizon length (mpc steps)  # TODO: Modify
        self.mpc_dt = 0.01  # mpc sampling time (s), needs to be a factor of N
        self.mpc_factor = int(self.mpc_dt / self.dt)  # mpc sampling time (timesteps), repeat mpc every x timesteps
        self.N_time = self.N * self.mpc_dt  # mpc horizon time
        self.N_k = int(self.N * self.mpc_factor)  # total mpc prediction horizon length (low-level timesteps)

        self.n_X = 13  # number of SE(3) states
        self.n_U = 6
        # simulator uses SE(3) states! (X)
        # mpc uses euler-angle based states! (x)
        # need to convert between these carefully. Pay attn to X vs x !!!
        self.X_0 = np.array([0, 0, 0.4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # in rqvw form!!!
        self.X_f = np.hstack([2, 0, 0.4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).T  # desired final state
        mu = 1  # coeff of friction

        mpc_tool = None
        if tool == 'cvxpy':
            mpc_tool = mpc_euler
        elif tool == 'casadi':
            mpc_tool = mpc_euler_cas

        self.mpc = mpc_tool.Mpc(t=self.mpc_dt, N=self.N, m=self.m, g=self.g, mu=mu, Jinv=self.Jinv, rh=self.rh)

        if self.ctrl == "open":
            self.total_run = int(self.N * self.mpc_factor)

    def run(self):
        total = self.total_run + 1  # number of timesteps to plot
        t = 0  # time
        t0 = t  # starting time

        mpc_factor = self.mpc_factor  # repeat mpc every x seconds
        mpc_counter = copy.copy(mpc_factor)
        X_traj = np.zeros((total, self.n_X))
        X_traj[0, :] = self.X_0  # initial conditions
        f_hist = np.zeros((total, self.n_U))
        s_hist = np.zeros(total)
        U = np.zeros(self.n_U)
        j = int(self.mpc_factor)
        # pf_ref = np.zeros((total, self.n_U))
        # f_pred_hist = np.zeros((total, self.n_U))
        # p_pred_hist = np.zeros((total, self.n_U))
        x_ref, pf_ref = self.path_plan_init(x_in=convert(X_traj[0, :]), xf=convert(self.X_f))
        init = True
        for k in tqdm(range(0, self.total_run)):
            t = t + self.dt

            s = self.gait_scheduler(t, t0)

            if self.ctrl == 'closed':
                if mpc_counter == mpc_factor:  # check if it's time to restart the mpc
                    mpc_counter = 0  # restart the mpc counter
                    C = self.gait_map(self.N, self.mpc_dt, t, t0)
                    x_refk = self.path_plan_grab(x_ref=x_ref, k=k)
                    pf_refk = self.path_plan_grab(x_ref=pf_ref, k=k)
                    x_in = convert(X_traj[k, :])  # convert to mpc states
                    U = self.mpc.mpcontrol(x_in=x_in, x_ref_in=x_refk, pf=pf_refk, C=C, init=init)
                    init = False  # after the first mpc run, change init to false

                mpc_counter += 1
                f_hist[k, :] = U[0, :]  # * s  # take first timestep

            else:  # Open loop traj opt, this will fail if total != mpc_factor
                if int(total/self.N) != mpc_factor:
                    print("ERROR: Incorrect settings", total/self.N, mpc_factor)
                if k == 0:
                    C = self.gait_map(self.N, self.mpc_dt, t, t0)
                    x_refk = x_ref
                    x_in = convert(X_traj[k, :])  # convert to mpc states
                    U = self.mpc.mpcontrol(x_in=x_in, x_ref_in=x_refk, pf=pf_cur, C=C, init=init)
                    for i in range(0, self.N):
                        f_hist[int(i*j):int(i*j+j), :] = np.tile(U[i, :], (j, 1))

            s_hist[k] = s
            X_traj[k + 1, :] = self.rk4_normalized(xk=X_traj[k, :], uk=f_hist[k, :])

        # plots.posplot(p_ref=self.X_f[0:3], p_hist=X_traj[:, 0:3],
        #   p_pred_hist=p_pred_hist, f_pred_hist=f_pred_hist, pf_hist=pf_ref)
        plots.posplot_animate(p_ref=self.X_f[0:3], p_hist=X_traj[::mpc_factor, 0:3],
                              ref_traj=x_ref[::mpc_factor, 0:3], pf_ref=pf_ref)
        plots.fplot(total, p_hist=X_traj[:, 0:3], f_hist=f_hist, s_hist=s_hist)
        # plots.posplot_animate_cube(p_ref=self.X_f[0:3], X_hist=X_traj[::50, :])

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
        dq = 0.5 * L(q) @ H @ w
        Fgn = np.array([0, 0, -g]) * m  # Gravitational force in world frame
        Fgb = H.T @ Q.T @ H @ Fgn  # rotate Fgn from world frame to body frame
        Ft = F + Fgb  # total force in body frame
        dv = 1 / m * Ft - np.cross(w, v)
        dw = np.linalg.solve(J, tau - np.cross(w, J @ w))
        dx = np.hstack((dp, dq, dv, dw)).T
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
        xn[3:7] = xn[3:7] / np.linalg.norm(xn[3:7])  # normalize the quaternion term
        return xn

    def gait_scheduler(self, t, t0):
        phi = np.mod((t - t0) / self.t_p, 1)
        if phi > self.phi_switch:
            s = 0  # scheduled swing
        else:
            s = 1  # scheduled stance
        return s

    def gait_map(self, N, dt, ts, t0):
        # generate vector of scheduled contact states over the mpc's prediction horizon
        C = np.zeros(N)
        for k in range(0, N):
            C[k] = self.gait_scheduler(t=ts, t0=t0)
            ts += dt
        return C

    def path_plan_init(self, x_in, xf):
        # Path planner--generate low-level reference trajectory for the entire run
        dt = self.dt
        t_sit = int(0)  # timesteps spent "sitting" at goal
        t_ref = int(self.total_run - t_sit)
        x_ref = np.linspace(start=x_in, stop=xf, num=t_ref)  # interpolate positions
        period = self.t_p  # *1.2  # * self.mpc_dt / 2
        amp = self.t_p/4  # amplitude
        phi = np.pi*1  # np.pi*3/2  # phase offset
        x_ref[:, 2] = [x_in[2] + amp + amp*np.sin(2*np.pi/period*(i*dt)+phi) for i in range(t_ref)]
        # sit at the goal
        if t_sit != 0:
            x_ref = np.vstack((x_ref, np.tile(xf, (t_sit, 1))))
            x_ref[-t_sit:, 2] = [x_ref[-t_sit, 2] + amp +
                                 amp * np.sin(2 * np.pi / period * (i * dt)) for i in range(t_sit)]
        # interpolate linear velocities
        x_ref[:-1, 6] = [(x_ref[i + 1, 0] - x_ref[i, 0]) / dt for i in range(self.total_run - 1)]
        x_ref[:-1, 7] = [(x_ref[i + 1, 1] - x_ref[i, 1]) / dt for i in range(self.total_run - 1)]
        x_ref[:-1, 8] = [(x_ref[i + 1, 2] - x_ref[i, 2]) / dt for i in range(self.total_run - 1)]

        C = self.gait_map(self.total_run, dt, 0, 0)  # low-level contact map for the entire run
        idx_pf = find_peaks(-x_ref[:, 2])[0]  # indexes of footstep positions
        idx_pf[0] = 0  # enforce first footstep idx to correspond to first timestep
        idx_pf = np.hstack((idx_pf, self.total_run-1))  # add final footstep idx based on last timestep
        n_pf = np.shape(idx_pf)[0]  # number of footstep positions
        pf_ref = np.zeros((self.total_run, 3))
        j = int(period/dt)  # number of low-level timesteps in one gait cycle
        for i in range(0, n_pf):
            if int(i * j + j) > self.total_run:
                jf = self.total_run - int(i * j)
                pf_ref[int(i * j):int(i * j + jf), 0:2] = np.tile(x_ref[idx_pf[i], 0:2], (jf, 1))
            else:
                pf_ref[int(i * j):int(i * j + j), 0:2] = np.tile(x_ref[idx_pf[i], 0:2], (j, 1))

        # np.set_printoptions(threshold=sys.maxsize)
        # print(pf_ref)
        return x_ref, pf_ref

    def path_plan_grab(self, x_ref, k):
        # Grab appropriate timesteps of pre-planned trajectory for mpc
        N_k = self.N_k  # length of MPC horizon in low-level ts
        N_kleft = np.shape(x_ref[k:, :])[0]  # number of remaining timesteps in the plan
        xf = x_ref[-1, :]
        if N_k <= N_kleft:
            x_refk = x_ref[k:(k+N_k), :]
        elif N_kleft == 0:
            x_refk = np.tile(xf, (N_k, 1))
        else:
            x_refk = x_ref[k:, :]
            x_refk = np.vstack((x_refk, np.tile(xf, (N_k - N_kleft, 1))))

        return x_refk[::self.mpc_factor, :]  # change to mpc-level timesteps
