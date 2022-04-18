"""
Copyright (C) 2021-2022 Benjamin Bokser
"""

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
plt.style.use(['science', 'no-latex'])
plt.rcParams['lines.linewidth'] = 2
import matplotlib.ticker as plticker
import itertools
plt.rcParams['font.size'] = 16


def fplot(total, p_hist, f_hist, s_hist, dims):

    fig, axs = plt.subplots(dims+2, sharex="all")
    plt.xlabel("Timesteps")

    if dims == 2:
        axs[0].plot(range(total), p_hist[:, 1], color='blue')
        axs[0].set_title('Base z position w.r.t. time')
        axs[0].set_ylabel("Z position (m)")

        axs[1].plot(range(total), f_hist[:, 0], color='blue')
        axs[1].set_title('Magnitude of X Output Force')
        axs[1].set_ylabel("Force, N")
        axs[2].plot(range(total), f_hist[:, 1], color='blue')
        axs[2].set_title('Magnitude of Z Output Force')
        axs[2].set_ylabel("Force, N")
        axs[3].plot(range(total), s_hist, color='blue')
        axs[3].set_title('Scheduled Contact')
        axs[3].set_ylabel("True/False")

    elif dims == 3:
        axs[0].plot(range(total), p_hist[:, 2], color='blue')
        axs[0].set_title('Base z position w.r.t. time')
        axs[0].set_ylabel("Z position (m)")

        axs[1].plot(range(total), f_hist[:, 0], color='blue')
        axs[1].set_title('Magnitude of X Output Force')
        axs[1].set_ylabel("Force, N")
        axs[2].plot(range(total), f_hist[:, 1], color='blue')
        axs[2].set_title('Magnitude of Y Output Force')
        axs[2].set_ylabel("Force, N")
        axs[3].plot(range(total), f_hist[:, 2], color='blue')
        axs[3].set_title('Magnitude of Z Output Force')
        axs[3].set_ylabel("Force, N")
        axs[4].plot(range(total), s_hist, color='blue')
        axs[4].set_title('Scheduled Contact')
        axs[4].set_ylabel("True/False")

    plt.show()


def posplot(p_ref, p_hist, dims):

    if dims == 2:
        plt.plot(p_hist[:, 0], p_hist[:, 1], color='blue', label='body position')
        plt.title('Body XZ Position')
        plt.ylabel("z (m)")
        plt.xlabel("x (m)")
        plt.scatter(p_hist[0, 0], p_hist[0, 1], color='green', marker="x", s=100, label='starting position')
        plt.scatter(p_ref[0], p_ref[1], color='orange', marker="x", s=100, label='position setpoint')
        plt.legend(loc="upper left")

    elif dims == 3:
        ax = plt.axes(projection='3d')
        ax.plot(p_hist[:, 0], p_hist[:, 1], p_hist[:, 2], color='red', label='Body Position')
        ax.set_title('Body Position')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.scatter(*p_hist[0, :], color='green', marker="x", s=200, label='Starting Position')
        ax.scatter(*p_ref, marker="x", s=200, color='orange', label='Target Position')
        ax.legend()
        intervals = 2
        loc = plticker.MultipleLocator(base=intervals)
        ax.xaxis.set_minor_locator(loc)
        ax.yaxis.set_minor_locator(loc)
        ax.zaxis.set_minor_locator(loc)
        # Add the grid
        ax.grid(which='minor', axis='both', linestyle='-')
        ax.xaxis.labelpad = 30
        ax.yaxis.labelpad = 30
        ax.zaxis.labelpad = 30

    plt.show()


def posfplot(p_ref, p_hist, p_pred_hist, f_pred_hist, pf_hist, dims):

    if dims == 2:
        plt.plot(p_hist[:, 0], p_hist[:, 1], color='blue', label='body position')
        plt.title('Body XZ Position')
        plt.ylabel("z (m)")
        plt.xlabel("x (m)")
        plt.scatter(p_hist[0, 0], p_hist[0, 1], color='green', marker="x", s=100, label='starting position')
        plt.scatter(p_ref[0], p_ref[1], color='orange', marker="x", s=100, label='position setpoint')
        plt.legend(loc="upper left")

    elif dims == 3:
        ax = plt.axes(projection='3d')
        ax.plot(p_hist[:, 0], p_hist[:, 1], p_hist[:, 2], color='red', label='Body Position')
        ax.set_title('Body Position')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.scatter(*p_hist[0, :], color='green', marker="x", s=200, label='Starting Position')
        ax.quiver(p_pred_hist[:, 0], p_pred_hist[:, 1], p_pred_hist[:, 2],
                  -f_pred_hist[:, 0], -f_pred_hist[:, 1], -f_pred_hist[:, 2], label='Predicted Forces')
        ax.scatter(pf_hist[:, 0], pf_hist[:, 1], pf_hist[:, 2], marker=".", s=200, color='blue', label='Footstep Positions')
        ax.scatter(*p_ref, marker="x", s=200, color='orange', label='Target Position')
        ax.legend()
        intervals = 2
        loc = plticker.MultipleLocator(base=intervals)
        ax.xaxis.set_minor_locator(loc)
        ax.yaxis.set_minor_locator(loc)
        ax.zaxis.set_minor_locator(loc)
        # Add the grid
        ax.grid(which='minor', axis='both', linestyle='-')
        ax.xaxis.labelpad = 30
        ax.yaxis.labelpad = 30
        ax.zaxis.labelpad = 30

    plt.show()


def posplot_t(p_ref, p_hist, total):
    ax = plt.axes(projection='3d')
    ax.plot(p_hist[:, 0], range(total), p_hist[:, 1], color='blue', label='body position')
    ax.set_title('Body XZ Position')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("time (s)")
    ax.set_zlabel("z (m)")
    ax.plot(np.zeros(total), range(total), np.zeros(total), color='green', label='starting position')
    pref0 = list(itertools.repeat(p_ref[0], total))
    pref1 = list(itertools.repeat(p_ref[1], total))
    ax.plot(pref0, range(total), pref1, color='orange', label='position setpoint')
    # ax.legend(loc="upper left")

    plt.show()