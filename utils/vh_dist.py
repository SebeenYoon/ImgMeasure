import numpy as np
import math


def RX(rx):
    return np.array([[1.,0,0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]], dtype=np.float32)


def RY(ry):
    return np.array([[math.cos(ry),0,math.sin(ry)], [0,1,0], [-math.sin(ry), 0, math.cos(ry)]], dtype=np.float32)


def RZ(rz):
    return np.array([[math.cos(rz), -math.sin(rz),0],[math.sin(rz), math.cos(rz),0],[0,0,1]], dtype=np.float32)


def vertical_dist(start, end, f, cx, cy, L, R):
    # Calculate object location and vertical distance(height)
    a, b = np.array([[start.x() - cx, start.y() - cy, f]], dtype=np.float32), np.array(
        [[end.x() - cx, end.y() - cy, f]], dtype=np.float32)
    c, h = R.T @ a.T, R.T @ b.T
    X = c[0] / c[1] * L
    Z = c[2] / c[1] * L
    H = (c[1] / c[2] - h[1] / h[2]) * Z
    return H


def horizon_dist(start, end, f, cx, cy, L, R):
    # Calculate object location and horizontal distance
    a = np.array([[start.x() - cx, start.y() - cy, f]], dtype=np.float32)
    c = R.T @ a.T

    Z1 = c[2] / c[1] * L
    X1 = c[0] / c[2] * Z1
    H1 = [0]

    a = np.array([[end.x() - cx, end.y() - cy, f]], dtype=np.float32)
    c = R.T @ a.T

    Z2 = c[2] / c[1] * L
    X2 = c[0] / c[2] * Z2
    H2 = [0]

    dist = ((X1[0] - X2[0]) ** 2 + (Z1[0] - Z2[0]) ** 2) ** (1/2)
    return dist


def calib_height(start, end, f, cx, cy, H, R):
    '''
    :param start:
    :param end:
    :param f:
    :param cx:
    :param cy:
    :param H: Object Height
    :param R:
    :return L: Camera Height
    '''
    # Calculate object location and height
    a, b = np.array([[start.x() - cx, start.y() - cy, f]], dtype=np.float32), np.array(
        [[end.x() - cx, end.y() - cy, f]], dtype=np.float32)
    c, h = R.T @ a.T, R.T @ b.T

    if ((c[1] / c[2] - h[1] / h[2]) * c[2] / c[1]) > 0:
        L_div_H = 1/ ((c[1] / c[2] - h[1] / h[2]) * c[2] / c[1])
    L = L_div_H * H  # L: 카메라 설치 높이
    return L


def cam_orientation(vp, cx, cy, f, L):
    x_delta = (cy-vp[1])
    y_delta = (cx-vp[0])
    x_angle = - np.arctan(x_delta / f)
    y_angle = np.arctan(y_delta / f)
    z_angle = np.arctan(0 / f)
    cam_ori = np.array([x_angle, y_angle, z_angle])

    # Draw grids on the ground
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    Rc = RZ(cam_ori[2]) @ RY(cam_ori[1]) @ RX(cam_ori[0])
    R = Rc.T
    tc = np.array([[0, -L, 0]], dtype=np.float32)
    t = -Rc.T @ tc.T

    return K, R, t, cam_ori
