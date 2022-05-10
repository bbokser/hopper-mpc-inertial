"""
Copyright (C) 2020-2022 Benjamin Bokser
"""
import plots
import mpc_euler_cas
import mpc_euler
from utils import H, L, R, quat2euler

from tqdm import tqdm
import numpy as np
import copy
import sys
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline

np.set_printoptions(suppress=True, linewidth=np.nan)


def convert(X_in):
    # convert from simulator states to mpc states (SE3 to euler)
    x0 = np.zeros(12)
    x0[0:3] = X_in[0:3]
    q = X_in[3:7]
    x0[3:6] = quat2euler(q)  # q -> euler
    Q = L(q) @ R(q).T
    x0[6:9] = H.T @ Q @ H @ X_in[7:10]  # body frame v -> world frame pdot
    x0[9:] = H.T @ Q @ H @ X_in[10:13]  # body frame w -> world frame w
    return x0


class Runner:
    def __init__(self, dt=1e-3, tool='cvxpy', curve=False, t_run=5000):
        self.dt = dt
        self.t_run = t_run
        self.curve = curve
        # self.tol = 1e-3  # desired mpc tolerance
        self.m = 7.5  # mass of the robot, kg
        self.J = np.array([[76148072.89, 70089.52, 2067970.36],
                           [70089.52, 45477183.53, -87045.58],
                           [2067970.36, -87045.58, 76287220.47]])*(10**(-9))  # g/mm2 to kg/m2
        self.Jinv = np.linalg.inv(self.J)
        # TODO: Check this, axes are likely wrong
        # self.rh = np.array([0.02201854, 6.80044366, 0.97499173]) / 1000  # mm to m
        self.rh = np.array([0., 0., 0.])  # mm to m
        self.g = 9.807  # gravitational acceleration, m/s2
        self.t_p = 0.8  # 0.8 gait period, seconds
        self.phi_switch = 0.5  # 0.5  # switching phase, must be between 0 and 1. Percentage of gait spent in contact.
        self.N = 40  # mpc prediction horizon length (mpc steps)  # TODO: Modify
        self.mpc_dt = 0.02  # mpc sampling time (s), needs to be a factor of N
        self.mpc_factor = int(self.mpc_dt / self.dt)  # mpc sampling time (timesteps), repeat mpc every x timesteps
        self.N_time = self.N * self.mpc_dt  # mpc horizon time
        self.N_k = int(self.N * self.mpc_factor)  # total mpc prediction horizon length (low-level timesteps)

        self.n_X = 13  # number of SE(3) states
        self.n_U = 6
        # simulator uses SE(3) states! (X)
        # mpc uses euler-angle based states! (x)
        # need to convert between these carefully. Pay attn to X vs x !!!
        self.X_0 = np.array([0, 0, 0.27, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # in rqvw form!!!
        self.X_f = np.hstack([1, 1, 0.27, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).T  # desired final state
        mu = 1  # coeff of friction

        mpc_tool = None
        if tool == 'cvxpy':
            mpc_tool = mpc_euler
        elif tool == 'casadi':
            mpc_tool = mpc_euler_cas

        self.mpc = mpc_tool.Mpc(t=self.mpc_dt, N=self.N, m=self.m, g=self.g, mu=mu, Jinv=self.Jinv, rh=self.rh)

        # TODO: don't forget this in application
        self.t_start = 0.5*self.t_p*self.phi_switch  # start halfway through stance phase

    def run(self):
        t_run = self.t_run + 1  # number of timesteps to plot
        t = self.t_start  # time
        t0 = 0
        mpc_factor = self.mpc_factor  # repeat mpc every x seconds
        mpc_counter = copy.copy(mpc_factor)
        X_traj = np.tile(self.X_0, (t_run, 1))  # initial conditions
        f_hist = np.zeros((t_run, self.n_U))
        s_hist = np.zeros(t_run)
        U = np.zeros(self.n_U)
        j = self.mpc_factor
        x_ref, pf_ref = self.path_plan_init(x_in=convert(X_traj[0, :]), xf=convert(self.X_f))
        init = True
        plots.posplot_animate(p_ref=self.X_f[0:3], p_hist=X_traj[::mpc_factor, 0:3],
                              ref_traj=x_ref[::mpc_factor, 0:3], pf_ref=pf_ref[::mpc_factor, :])
        # plots.posplot_animate_cube(p_ref=self.X_f[0:3], X_hist=x_ref[::mpc_factor, :])
        for k in tqdm(range(0, self.t_run)):
            t = t + self.dt

            s = self.gait_scheduler(t, t0)

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
            s_hist[k] = s
            X_traj[k + 1, :] = self.rk4_normalized(xk=X_traj[k, :], uk=f_hist[k, :], pfk=pf_ref[k, :])
            #if k >= 3059:
            #    break

        # plots.posplot(p_ref=self.X_f[0:3], p_hist=X_traj[:, 0:3],
        #   p_pred_hist=p_pred_hist, f_pred_hist=f_pred_hist, pf_hist=pf_ref)
        plots.posplot_animate(p_ref=self.X_f[0:3], p_hist=X_traj[::mpc_factor, 0:3],
                              ref_traj=x_ref[::mpc_factor, 0:3], pf_ref=pf_ref[::mpc_factor, :])
        plots.fplot(t_run, p_hist=X_traj[:, 0:3], f_hist=f_hist, s_hist=s_hist)
        plots.posplot_animate_cube(p_ref=self.X_f[0:3], X_hist=X_traj[::mpc_factor, :])

        return None

    def dynamics_ct(self, X, U, pf):
        # SE(3) nonlinear dynamics
        # Unpack state vector
        m = self.m
        g = self.g
        J = self.J
        rh = self.rh
        p = X[0:3]  # W frame
        q = X[3:7]  # B to W
        v = X[7:10]  # B frame
        w = X[10:13]  # B frame
        Fw = U[0:3]  # W frame
        tau = U[3:]  # B frame

        Q = L(q) @ R(q).T
        Fgw = np.array([0, 0, -g]) * m  # Gravitational force in world frame
        Ftb = H.T @ Q.T @ H @ (Fgw + Fw)  # rotate Fgw + Fw from world frame to body frame
        r = rh + H.T @ Q.T @ H @ (pf - p)  # vec from CoM to step location in body frame
        Fb = H.T @ Q.T @ H @ Fw  # rotate Fw from world frame to body frame
        tautb = tau + np.cross(r, Fb)  # sum body frame rw torque with torque due to footstep vector and leg force

        dp = H.T @ Q @ H @ v  # rotate v from body to world frame
        dq = 0.5 * L(q) @ H @ w
        dv = 1 / m * Ftb - np.cross(w, v)
        dw = np.linalg.solve(J, tautb - np.cross(w, J @ w))
        dx = np.hstack((dp, dq, dv, dw)).T
        return dx

    def rk4_normalized(self, xk, uk, pfk):
        # RK4 integrator solves for new X
        dynamics = self.dynamics_ct
        h = self.dt
        f1 = dynamics(xk, uk, pfk)
        f2 = dynamics(xk + 0.5 * h * f1, uk, pfk)
        f3 = dynamics(xk + 0.5 * h * f2, uk, pfk)
        f4 = dynamics(xk + h * f3, uk, pfk)
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
        N_k = self.N_k  # total MPC horizon in low-level timesteps
        t_run = self.t_run
        dt = self.dt
        t_sit = 0  # timesteps spent "sitting" at goal
        t_traj = int(t_run - t_sit)  # timesteps for trajectory not including sit time
        t_ref = t_run + N_k  # timesteps for reference (extra for MPC)
        x_ref = np.linspace(start=x_in, stop=xf, num=t_traj)  # interpolate positions
        if self.curve is True:
            spline_t = np.array([0, t_traj*0.3, t_traj])
            spline_y = np.array([x_in[1], xf[1]*0.7, xf[1]])
            csy = CubicSpline(spline_t, spline_y)
            spline_psi = np.array([0, np.sin(45*np.pi/180) * 0.7, np.sin(45*np.pi/180)])
            cspsi = CubicSpline(spline_t, spline_psi)
            for k in range(t_traj):
                x_ref[k, 1] = csy(k)  # create evenly spaced sample points of desired trajectory
                x_ref[k, 5] = cspsi(k)  # create evenly spaced sample points of desired trajectory
                # interpolate angular velocity
            x_ref[:-1, 11] = [(x_ref[i + 1, 11] - x_ref[i, 11]) / dt for i in range(t_run - 1)]

        x_ref = np.vstack((x_ref, np.tile(xf, (N_k + t_sit, 1))))  # sit at the goal
        period = self.t_p  # *1.2  # * self.mpc_dt / 2
        amp = self.t_p / 4  # amplitude
        phi = np.pi * 3 / 2  # np.pi*3/2  # phase offset
        # make height sine wave
        x_ref[:, 2] = [x_in[2] + amp + amp * np.sin(2 * np.pi / period * (i * dt) + phi) for i in range(t_ref)]
        # interpolate linear velocities
        x_ref[:-1, 6:9] = [(x_ref[i + 1, 0:3] - x_ref[i, 0:3]) / dt for i in range(t_ref - 1)]

        C = self.gait_map(t_ref, dt, self.t_start, 0)  # low-level contact map for the entire run
        idx_pf = find_peaks(-x_ref[:, 2])[0]  # indexes of footstep positions
        idx_pf = np.hstack((0, idx_pf))  # add initial footstep idx based on first timestep
        # idx_pf[0] = 0  # enforce first footstep idx to correspond to first timestep
        idx_pf = np.hstack((idx_pf, t_ref-1))  # add final footstep idx based on last timestep
        pf_ref = np.zeros((t_ref, 3))
        kf = 0
        n_idx = np.shape(idx_pf)[0]
        for k in range(1, t_ref):
            if C[k-1] == 0 and C[k] == 1 and kf < n_idx:
                kf += 1
            pf_ref[k, 0:2] = x_ref[idx_pf[kf], 0:2]

        # np.set_printoptions(threshold=sys.maxsize)
        return x_ref, pf_ref

    def path_plan_grab(self, x_ref, k):
        # Grab appropriate timesteps of pre-planned trajectory for mpc
        return x_ref[k:(k+self.N_k):self.mpc_factor, :]  # change to mpc-level timesteps
