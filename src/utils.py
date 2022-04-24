import numpy as np
import transforms3d

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


def rz(phi):
    # linearized rotation matrix Rz(phi) using commanded yaw
    Rz = np.array([[np.cos(phi), np.sin(phi), 0.0],
                   [-np.sin(phi), np.cos(phi), 0.0],
                   [0.0, 0.0, 1.0]])
    return Rz


def quat2z(Q):
    # get ZYX Z-axis euler angle from quaternion
    return np.arctan2(Q[:, 1] * Q[:, 3] - Q[:, 2] * Q[:, 0], -(Q[:, 2] * Q[:, 3] + Q[:, 1] * Q[:, 0]))


def quat2euler(Q):
    # ZYX Euler angles, not tait-bryan? Output roll-pitch-yaw order # this is why euler angles suck ass
    zyx = transforms3d.euler.quat2euler(Q, axes='rzyx')
    xyz = np.zeros(3)
    xyz[0] = zyx[2]
    xyz[1] = zyx[1]
    xyz[2] = zyx[0]
    return xyz


def convert(X_in):
    # convert from simulator states to mpc states (SE3 to euler)
    X0 = np.zeros(12)
    X0[0:3] = X_in[0:3]
    q = X_in[3:7]
    X0[3:6] = quat2euler(q)  # q -> euler
    Q = L(q) @ R(q).T
    X0[6:9] = H.T @ Q @ H @ X_in[7:10]  # v -> pdot
    X0[9:] = H.T @ Q @ H @ X_in[10:13]  # body frame w -> world frame w
    return X0


def quat2rot(Q):
    w, x, y, z = Q
    R = np.array([[2 * (w ** 2 + x ** 2) - 1, 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 2 * (w ** 2 + y ** 2) - 1, 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 2 * (w ** 2 + z ** 2) - 1]])
    return R
