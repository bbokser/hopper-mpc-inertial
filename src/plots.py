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
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=3, hspace=0.7)

    axs[0].plot(range(total), p_hist[:, 2], color='blue')
    axs[0].set_title('Base z position w.r.t. time')
    axs[0].set_ylabel("Z pos (m)")

    axs[1].plot(range(total), f_hist[:, 0], color='blue')
    axs[1].set_title('X Output Force')
    axs[1].set_ylabel("N")
    axs[2].plot(range(total), f_hist[:, 1], color='blue')
    axs[2].set_title('Y Output Force')
    axs[2].set_ylabel("N")
    axs[3].plot(range(total), f_hist[:, 2], color='blue')
    axs[3].set_title('Z Output Force')
    axs[3].set_ylabel("N")

    axs[4].plot(range(total), f_hist[:, 3], color='blue')
    axs[4].set_title('X-axis Torque')
    axs[4].set_ylabel("Nm")
    axs[5].plot(range(total), f_hist[:, 4], color='blue')
    axs[5].set_title('Y-axis Torque')
    axs[5].set_ylabel("Nm")
    axs[6].plot(range(total), f_hist[:, 5], color='blue')
    axs[6].set_title('Z-axis Torque')
    axs[6].set_ylabel("Nm")

    axs[7].plot(range(total), s_hist, color='blue')
    axs[7].set_title('Scheduled Contact')
    axs[7].set_ylabel("Truth")

    plt.show()


def set_axes_equal(ax: plt.Axes):
    """
    https://stackoverflow.com/questions/13685386/
    matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def posplot(p_ref, p_hist, ref_traj, pf_ref):
    ax = plt.axes(projection='3d')
    # ax.set_title('Body Position')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.scatter(*p_hist[0, :], color='green', marker="x", s=200, label='Starting Position')
    ax.scatter(*p_ref, marker="x", s=200, color='orange', label='Target Position')
    ax.plot(p_hist[:, 0], p_hist[:, 1], p_hist[:, 2], color='r', label='CoM Position')
    ax.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2], ls='--', c='g', label='Reference Trajectory')
    ax.scatter(pf_ref[:, 0], pf_ref[:, 1], pf_ref[:, 2], color='blue', label='Planned Footsteps')
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
    ax.set_box_aspect([1, 1, 1])  # make aspect ratio equal for all axes
    # ax.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
    set_axes_equal(ax)  # IMPORTANT - this is also required
    plt.show()


def animate_line(N, dataSet1, dataSet2, dataSet3, line, ref, pf, ax):
    line._offsets3d = (dataSet1[0:3, :N])
    ref._offsets3d = (dataSet2[0:3, :N])
    pf._offsets3d = (dataSet3[0:3, :N])
    ax.view_init(elev=10., azim=N)


def posplot_animate(p_ref, p_hist, ref_traj, pf_ref):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title('Body Position')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim3d(0, 2)
    ax.set_ylim3d(0, 2)
    ax.set_zlim3d(0, 2)

    ax.scatter(*p_hist[0, :], color='green', marker="x", s=200, label='Starting Position')
    ax.scatter(*p_ref, marker="x", s=200, color='orange', label='Target Position')
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

    N = len(p_hist)
    line = ax.scatter(p_hist[:, 0], p_hist[:, 1], p_hist[:, 2], lw=2, c='r', label='CoM Position')  # For line plot
    ref = ax.scatter(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2], lw=2, c='g', label='Reference Trajectory')
    pf = ax.scatter(pf_ref[:, 0], pf_ref[:, 1], pf_ref[:, 2], color='blue', label='Planned Footsteps')
    ax.legend()
    line_ani = animation.FuncAnimation(fig, animate_line, frames=N,
                                       fargs=(p_hist.T, ref_traj.T, pf_ref.T, line, ref, pf, ax),
                                       interval=2, blit=False)
    # line_ani.save('basic_animation.mp4', fps=30, bitrate=4000, extra_args=['-vcodec', 'libx264'])

    plt.show()


def animate_cube(N, pt_hist, points):
    """perform animation step"""
    for i in range(8):
        points[i]._offsets3d = (pt_hist[i][0:3, :N])


def posplot_animate_cube(p_ref, X_hist):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
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
