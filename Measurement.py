import numpy as np
import cv2
import math
import copy
from utils import vanishing_point


def RX(rx):
    return np.array([[1., 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]], dtype=np.float32)


def RY(ry):
    return np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]], dtype=np.float32)


def RZ(rz):
    return np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]], dtype=np.float32)


class MouseDrag():
    def __init__(self):
        self._dragged = False
        self.start, self.end = (1, 1), (1, 1)


def MouseEventHandler(event, x, y, flags, params):
    if params == None: return
    if event == cv2.EVENT_LBUTTONDOWN:
        params._dragged = True
        params.start = (x, y)
        params.end = (0, 0)
    elif event == cv2.EVENT_MOUSEMOVE:
        if params._dragged:
            params.end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if params._dragged:
            params._dragged = False
            params.end = (x, y)


def main():
    # Load an image
    img = "./data/KakaoTalk_20230515_165448739.jpg"
    image = cv2.imread(img)
    # if image == None: return -1
    h, w = image.shape[:2]
    f, cx, cy, L = 1607, h/2, w/2, 1.45

    # vanishing point & camera rotation
    vp = vanishing_point.find_theta(image)
    x_delta = (cy-vp[1])
    y_delta = (cx-vp[0])
    x_angle = - np.arctan(x_delta / f)
    y_angle = np.arctan(y_delta / f)
    z_angle = np.arctan(0 / f)

    cam_ori = np.array([x_angle, y_angle,z_angle])

    grid_x, grid_z = (-5, 3), (4, 35)

    # Configure mouse callback
    drag = MouseDrag()
    cv2.namedWindow("3DV Tutorial: Pbject Localization and Measurement")
    cv2.setMouseCallback("3DV Tutorial: Pbject Localization and Measurement", MouseEventHandler, drag)  # ?

    # Draw grids on the ground
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    Rc = RZ(cam_ori[2]) @ RY(cam_ori[1]) @ RX(cam_ori[0])
    R = Rc.T
    tc = np.array([[0, -L, 0]], dtype=np.float32)
    t = -Rc.T @ tc.T
    # for z in range(grid_z[0], grid_z[1], 1):
    #     a, b = np.array([[grid_x[0], 0, z]], dtype=np.float32), np.array([[grid_x[1], 0, z]], dtype=np.float32)
    #     p = K @ (R @ a.T + t)
    #     q = K @ (R @ b.T + t)
    #     image = cv2.line(image, (int(p[0] / p[2]), int(p[1] / p[2])), (int(q[0] / q[2]), int(q[1] / q[2])),
    #                      (64, 128, 64), 1)
    #     # image = cv2.line(image, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (64, 128, 64), 1)
    #
    # for x in range(grid_x[0], grid_x[1]):
    #     a, b = np.array([[x, 0, grid_z[0]]], dtype=np.float32), np.array([[x, 0, grid_z[1]]], dtype=np.float32)
    #     p = K @ (R @ a.T + t)
    #     q = K @ (R @ b.T + t)
    #     image = cv2.line(image, (int(p[0] / p[2]), int(p[1] / p[2])), (int(q[0] / q[2]), int(q[1] / q[2])),
    #                      (64, 128, 64), 1)
    #     # image = cv2.line(image, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (64, 128, 64), 1)

    # 카메라 높이 추정 - (Known: 초점거리(f), 물체 world 위치 및 높이, 물체 image 위치 및 높이)
    while True:
        image_copy = copy.deepcopy(image)
        if drag.end[0] > 0 and drag.end[1] > 0:

            # Calculate object location and height
            a, b = np.array([[drag.start[0] - cx, drag.start[1] - cy, f]], dtype=np.float32), np.array(
                [[drag.end[0] - cx, drag.end[1] - cy, f]], dtype=np.float32)
            c, h = R.T @ a.T, R.T @ b.T
            # if c[1] # 정수비교가 필요하지만, 파이썬에서는 어떻게 하는지 모르므로 패스

            image_copy = cv2.line(image_copy, drag.start, drag.end, (0, 0, 255), 2)
            image_copy = cv2.circle(image_copy, drag.end, 2, (255, 0, 0), -1)
            image_copy = cv2.circle(image_copy, drag.start, 2, (0, 255, 0), -1)
            if ((c[1] / c[2] - h[1] / h[2]) * c[2] / c[1]) > 0:
                L_div_H = 1/ ((c[1] / c[2] - h[1] / h[2]) * c[2] / c[1])

        cv2.imshow("3DV Tutorial: Pbject Localization and Measurement", image_copy)
        if cv2.waitKey(1) == ord('q'): break

    H = float(input("Enter height of object(m): "))  # H: 물체 world 높이
    L = L_div_H * H  # L: 카메라 설치 높이
    print(f"Camera height is {L[0]:.3f} m")

    while True:
        image_copy = copy.deepcopy(image)
        if drag.end[0] > 0 and drag.end[1] > 0:
            # info = f"Camera height: {L[0]:.3f}m, Object height: {H}"
            # x, y = 20, 20
            # image_copy = cv2.putText(image_copy, info, (x, y), cv2.FONT_HERSHEY_PLAIN,
            #                          1, (0, 255, 0))  # start location

            image_copy = cv2.circle(image_copy, drag.start, 2, (0, 255, 0), -1) # draw circle
            image_copy = cv2.circle(image_copy, drag.end, 2, (255, 0, 0), -1) # draw circle
            image_copy = cv2.line(image_copy, drag.start, drag.end, (0, 0, 255), 2)  # draw line

            # Calculate object location and height
            a = np.array([[drag.start[0] - cx, drag.start[1] - cy, f]], dtype=np.float32)
            c = R.T @ a.T

            # if c[1] # 정수비교가 필요하지만, 파이썬에서는 어떻게 하는지 모르므로 패스
            Z1 = c[2] / c[1] * L
            X1 = c[0] / c[2] * Z1
            H1 = [0]

            info = f"X: {X1[0]:.3f}, Z: {Z1[0]:.3f}, H: {H1[0]:.3f}"
            image_copy = cv2.putText(image_copy, info, (drag.start[0] - 20, drag.start[1] + 20), cv2.FONT_HERSHEY_PLAIN,
                                     1, (0, 255, 0))  # start location

            # Calculate object location and height
            a = np.array([[drag.end[0] - cx, drag.end[1] - cy, f]], dtype=np.float32)
            c = R.T @ a.T

            # if c[1] # 정수비교가 필요하지만, 파이썬에서는 어떻게 하는지 모르므로 패스
            Z2 = c[2] / c[1] * L
            X2 = c[0] / c[2] * Z1
            H2 = [0]
            info = f"X: {X2[0]:.3f}, Z: {Z2[0]:.3f}, H: {H2[0]:.3f}"
            image_copy = cv2.putText(image_copy, info, (drag.end[0] - 20, drag.end[1] + 20), cv2.FONT_HERSHEY_PLAIN,
                                     1, (0, 255, 0))  # end location

            info = f"{((X1[0]-X2[0])**2 + (Z1[0]-Z2[0])**2)**(1/2):.3f}"  # distance
            image_copy = cv2.putText(image_copy, info, (int((drag.start[0] + drag.end[0])/2), int((drag.start[1] + drag.end[1])/2)), cv2.FONT_HERSHEY_PLAIN,
                                     1, (0, 255, 255))  # distance

        cv2.imshow("3DV Tutorial: Pbject Localization and Measurement", image_copy)
        if cv2.waitKey(1) == ord('q'): break


if __name__ == "__main__":
    main()