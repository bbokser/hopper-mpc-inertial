"""
Copyright (C) 2021-2022 Benjamin Bokser
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from utils import quat2rot

# plt.style.use(['science', 'no-latex'])
# plt.rcParams['lines.linewidth'] = 2
import matplotlib.ticker as plticker

plt.rcParams['font.size'] = 16


def fplot(total, p_hist, f_hist, s_hist):
    fig, axs = plt.subplots(8, sharex="all")
    plt.xlabel("Timesteps")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=0.7)

    axs[0].plot(range(total), p_hist[:, 2], color='blue')
    axs[0].set_title('Base z position w.r.t. time')
    axs[0].set_ylabel("Z position (m)")

    axs[1].plot(range(total), f_hist[:, 0], color='blue')
    axs[1].set_title('X Output Force')
    axs[1].set_ylabel("Force, N", rotation=0)
    axs[2].plot(range(total), f_hist[:, 1], color='blue')
    axs[2].set_title('Y Output Force')
    axs[2].set_ylabel("Force, N", rotation=0)
    axs[3].plot(range(total), f_hist[:, 2], color='blue')
    axs[3].set_title('Z Output Force')
    axs[3].set_ylabel("Force, N", rotation=0)

    axs[4].plot(range(total), f_hist[:, 3], color='blue')
    axs[4].set_title('X-axis Torque')
    axs[4].set_ylabel("Torque, Nm", rotation=0)
    axs[5].plot(range(total), f_hist[:, 4], color='blue')
    axs[5].set_title('Y-axis Torque')
    axs[5].set_ylabel("Torque, Nm", rotation=0)
    axs[6].plot(range(total), f_hist[:, 5], color='blue')
    axs[6].set_title('Z-axis Torque')
    axs[6].set_ylabel("Torque, Nm", rotation=0)

    axs[7].plot(range(total), s_hist, color='blue')
    axs[7].set_title('Scheduled Contact')
    axs[7].set_ylabel("True/False", rotation=0)

    plt.show()


def posplot(p_ref, p_hist, p_pred_hist, f_pred_hist, pf_hist):
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


def animate_line(num, dataSet, line):
    line.set_data(dataSet[0:2, :num])
    line.set_3d_properties(dataSet[2, :num])
    return line


def posplot_animate(p_ref, p_hist):
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax = plt.axes(projection='3d')
    ax.set_title('Body Position')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim3d(0, 2)
    ax.set_ylim3d(0, 2)
    ax.set_zlim3d(0, 2)
    # ax.plot(p_hist[:, 0], p_hist[:, 1], p_hist[:, 2], color='red', label='Body Position')
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

    n_data = len(p_hist)
    line = plt.plot(p_hist[:, 0], p_hist[:, 1], p_hist[:, 2], lw=2, c='g')[0]  # For line plot

    line_ani = animation.FuncAnimation(fig, animate_line, frames=n_data, fargs=(p_hist.T, line), interval=50, blit=False)
    # line_ani.save('basic_animation.mp4', fps=30, bitrate=4000, extra_args=['-vcodec', 'libx264'])

    plt.show()


def animate_cube(N, pt_hist, points):
    """perform animation step"""
    for i in range(8):
        points[i]._offsets3d = (pt_hist[i][0:3, :N])


def posplot_animate_cube(p_ref, X_hist):
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax = plt.axes(projection='3d')
    ax.set_title('Body Position')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim3d(0, 2)
    ax.set_ylim3d(0, 2)
    ax.set_zlim3d(0, 2)
    # ax.plot(p_hist[:, 0], p_hist[:, 1], p_hist[:, 2], color='red', label='Body Position')
    ax.scatter(*X_hist[0, 0:3], color='green', marker="x", s=200, label='Starting Position')
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

    vs = np.reshape(np.mgrid[-1:2:2, -1:2:2, -1:2:2].T, (8, 3)) * 0.1  # The cube vertices
    # Generate the list of connected vertices
    ed = [(j, k)
          for j in range(8)
          for k in range(j, 8)
          if sum(abs(vs[j] - vs[k])) == 2]
    # Number of frames.
    N = len(X_hist)
    # line = plt.plot(*X_hist[:, 0:3], lw=2, c='g')[0]  # For line plot
    pt_hist = np.zeros((N, 8, 3))
    for k in range(N):
        rotM = quat2rot(X_hist[k, 3:7])  # Get the rotation matrix for the current frame.
        vec = np.dot(vs, rotM)  # Calculate the 3D coordinates of the vertices of the rotated cube. 8x3
        pt_hist[k, :, :] = vec + X_hist[k, 0:3]  # # Now calculate the image coordinates of the points. 8 points in xyz
        # for j, k in ed:
        #     ax.plot(pt[[j, k], 0], pt[[j, k], 1], pt[[j, k], 2], 'g-', lw=3)  # Plot the edges.
        # ax.scatter(pt[:, 0], pt[:, 1], pt[:, 2], color='red', label='Body')  # Plot the vertices.

    p_hist = [None]*8
    points = [None]*8
    for i in range(0, 8):
        pi_hist = pt_hist[:, i, :]
        points[i] = ax.scatter(pi_hist[:, 0], pi_hist[:, 1], pi_hist[:, 2], marker="o", color='red')
        p_hist[i] = pi_hist.T

    anim = animation.FuncAnimation(fig, animate_cube, frames=N, fargs=(p_hist, points), interval=0.1, blit=False)
    # line_ani.save(r'AnimationNew.mp4')
    # anim.save('basic_animation.mp4', fps=30, bitrate=4000, extra_args=['-vcodec', 'libx264'])
    plt.show()
