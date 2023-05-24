from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, qRgb, QIcon, QPen, QFont
from PyQt5.QtWidgets import QAction, QApplication, QFileDialog, QLabel,\
    QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy, QInputDialog, \
    QLineEdit, QWidget, QHBoxLayout, QVBoxLayout, QLayout, QPushButton, qApp, QGroupBox, QRadioButton, QButtonGroup
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

import cv2
import sys
import glob
import os
import numpy as np
from utils import vh_dist, vp


class ImageViewer(QMainWindow):
    def __init__(self):

        super(ImageViewer, self).__init__()

        self.setMouseTracking(True)

        # ### Submenu in Menubar ###
        exitAction = QAction(QIcon('./assets/exit.png'), 'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

        # ### Menubar ###
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        filemenu = menubar.addMenu('File')
        filemenu.addAction(exitAction)

        # ### Toolbar ###
        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(exitAction)

        self.gray_color_table = [qRgb(i, i, i) for i in range(256)]

        self.base_path = './data'
        self.img_list = glob.glob(os.path.join(self.base_path, '*.jpg'))
        self.pos = 0
        self.total = len(self.img_list)

        self.printer = QPrinter()
        self.width = 1024
        self.height = 1024

        # widget
        self.lb_height = QLabel()
        self.lb_cam_intrinsic = QLabel()
        self.lb_xy_start = QLabel()
        self.lb_xy_move = QLabel()
        self.lb_xy_end = QLabel()
        self.lb_file_name = QLabel()

        self.lineedit = QLineEdit()
        self.lineedit.returnPressed.connect(self.onChanged)

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)
        self.imageLabel.mousePressEvent = self.mousePressEvent
        self.imageLabel.mouseMoveEvent = self.mouseMoveEvent
        self.imageLabel.mouseReleaseEvent = self.mouseReleaseEvent

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)

        self.radio_btn1 = QRadioButton("Calib: Camera Height")
        self.radio_btn1.setChecked(True)
        self.radio_btn2 = QRadioButton("Measure: Horizon")
        self.radio_btn3 = QRadioButton("Measure: Vertical")
        self.radio_btn1.toggled.connect(self.onMode)
        self.radio_btn2.toggled.connect(self.onMode)
        self.radio_btn3.toggled.connect(self.onMode)

        hwidget = QWidget()
        hbox = QVBoxLayout(hwidget)
        hbox.addWidget(self.lb_file_name)
        hbox.addWidget(self.lb_cam_intrinsic)

        grp_widget = QWidget()
        grp_layout = QHBoxLayout(grp_widget)
        grp_layout.addWidget(self.radio_btn1)
        grp_layout.addWidget(self.radio_btn2)
        grp_layout.addWidget(self.radio_btn3)
        grp_layout.addStretch(1)

        h2widget = QWidget()
        h2box = QHBoxLayout(h2widget)
        h2box.addWidget(self.lb_height)
        h2box.addWidget(self.lineedit)

        # self.button_zoom_in = QPushButton('Zoom IN', self)
        # self.button_zoom_in.clicked.connect(self.on_zoom_in)
        # self.button_zoom_out = QPushButton('Zoom OUT', self)
        # self.button_zoom_out.clicked.connect(self.on_zoom_out)
        # h2box.addWidget(self.button_zoom_in)
        # h2box.addWidget(self.button_zoom_out)

        vwidget = QWidget()
        vbox = QVBoxLayout(vwidget)
        vbox.addWidget(hwidget)
        vbox.addWidget(grp_widget)
        vbox.addWidget(h2widget)
        vbox.addWidget(self.scrollArea)

        self.setCentralWidget(vwidget)

        self.createActions()

        self.setWindowTitle("Image Measurement in Construction Sites")
        self.resize(self.width, self.height)

        image = cv2.imread(self.img_list[self.pos])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.openImage(image=self.toQImage(image))

        # default setting
        self.f = 1607
        self.image_shape = image.shape[0:2]
        self.cx, self.cy = self.image_shape[0]/2, self.image_shape[1]/2
        self.L = 1.45

        self.vp = vp.find_theta(image)
        self.K, self.R, self.t, self.cam_ori = vh_dist.cam_orientation(self.vp, self.cx, self.cy, self.f, self.L)

        self.lb_file_name.setText(self.img_list[self.pos])
        self.lb_cam_intrinsic.setText(
            f"Image Shape: {self.image_shape}, focal: {self.f}, cx:{self.cx}, cy:{self.cy}, L(m):{self.L}, vp: {self.vp}, cam_orig(deg): {self.cam_ori * 180 / 3.14}")

        self.drawing = False
        self.startPoint = None
        self.endPoint = None

        self.mode_calib = True
        self.mode_measure_h = False
        self.mode_measure_v = False

        self.dist = 0


    # def on_zoom_in(self, event):
    #     self.height += 100
    #     self.resize_image()
    #
    # def on_zoom_out(self, event):
    #     self.height -= 100
    #     self.resize_image()
    #
    # def resize_image(self):
    #     scaled_pixmap = self.pixmap.scaledToHeight(self.height)
    #     self.imageLabel.setPixmap(scaled_pixmap)

    def draw_Line(self, start, end):
        image = cv2.imread(self.img_list[self.pos])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.openImage(image=self.toQImage(image))

        painter = QPainter(self.imageLabel.pixmap())
        painter.setPen(QPen(Qt.green, 2, Qt.SolidLine))
        # print(self.past_x, self.past_y, self.present_x, self.present_y)
        painter.drawLine(start, end)
        painter.end()
        # self.imageLabel.update()

    def draw_Text(self, start, end):
        image = cv2.imread(self.img_list[self.pos])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.openImage(image=self.toQImage(image))

        painter = QPainter(self.imageLabel.pixmap())
        painter.setPen(QPen(Qt.green, 2, Qt.SolidLine))
        # print(self.past_x, self.past_y, self.present_x, self.present_y)

        painter.drawLine(start, end)

        if self.mode_measure_v:
            self.dist = vh_dist.vertical_dist(start, end, self.f, self.cx, self.cy, self.L, self.R)[0]
        elif self.mode_measure_h:
            self.dist = vh_dist.horizon_dist(start, end, self.f, self.cx, self.cy, self.L, self.R)
        else:
            print("뭡니까??")

        x_pos = int((start.x() + end.x())/2)
        y_pos = int((start.y() + end.y())/2)
        painter.setPen(QPen(Qt.blue, 2, Qt.SolidLine))
        painter.setFont(QFont('Consolas', 14))
        info = f"{self.dist:.3f}m"
        painter.drawText(x_pos, y_pos, info)

        painter.end()
        # self.imageLabel.update()

    def draw_grid(self):
        pass

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.startPoint = event.pos()
            self.drawing = True

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.endPoint = event.pos()
            self.draw_Line(self.startPoint, self.endPoint)
            self.draw_Text(self.startPoint, self.endPoint)

            # Status (bottom)
            self.statusBar().showMessage(f'start {self.startPoint.x()}, {self.startPoint.y()} end {self.endPoint.x()}, {self.endPoint.y()}')

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.endPoint = event.pos()
            self.drawing = False
            self.draw_Line(self.startPoint, self.endPoint)
            self.draw_Text(self.startPoint, self.endPoint)

            # Status (bottom)
            self.statusBar().showMessage(f'start {self.startPoint.x()}, {self.startPoint.y()} end {self.endPoint.x()}, {self.endPoint.y()}')

    def onMode(self):
        if self.radio_btn1.isChecked():  # calibration
            self.mode_calib = True
            self.mode_measure_h = False
            self.mode_measure_v = False

        if self.radio_btn2.isChecked():  # measurement
            self.mode_calib = False
            self.mode_measure_h = True
            self.mode_measure_v = False

        if self.radio_btn3.isChecked():  # measurement
            self.mode_calib = False
            self.mode_measure_h = False
            self.mode_measure_v = True


    def onChanged(self):
        try:
            height = float(self.lineedit.text())
            self.lb_height.setText(f"{height:.3f}m")
            self.lineedit.setText("")

            if self.mode_calib:
                self.L = vh_dist.calib_height(self.startPoint, self.endPoint, self.f, self.cx, self.cy, height, self.R)
                self.lb_cam_intrinsic.setText(
                    f"Image Shape: {self.image_shape}, focal: {self.f}, cx:{self.cx}, cy:{self.cy}, L(m):{self.L}, vp: {self.vp}, cam_orig(deg): {self.cam_ori * 180 / 3.14}")


        except:
            self.lb_height.setText("숫자로 지정하세요!")
            self.lineedit.setText("")

    def normalSize(self):
        self.imageLabel.adjustSize()

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()
        self.updateActions()

    def createActions(self):
        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S", enabled=False, triggered=self.normalSize)
        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False, checkable=True, shortcut="Ctrl+F", triggered=self.fitToWindow)

    def updateActions(self):
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))

    def toQImage(self, im, copy=False):
        if im is None:
            return QImage()
        if im.dtype == np.uint8:
            if len(im.shape) == 2:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
                qim.setColorTable(self.gray_color_table)
                return qim.copy() if copy else qim
            elif len(im.shape) == 3:
                if im.shape[2] == 3:
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
                    return qim.copy() if copy else qim
                elif im.shape[2] == 4:
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32)
                    return qim.copy() if copy else qim

    def openImage(self, image=None, fileName=None):
        if image == None:
            image = QImage(fileName)
        if image.isNull():
            QMessageBox.information(self, "Image Viewer",
                                    "Cannot load %s." % fileName)
            return
        self.pixmap = QPixmap.fromImage(image)
        self.imageLabel.setPixmap(self.pixmap)
        self.fitToWindowAct.setEnabled(True)
        self.updateActions()
        if not self.fitToWindowAct.isChecked():
            self.imageLabel.adjustSize()

    def keyPressEvent(self, e):
        if e.key() == 65:
            if not self.pos == 0:
                self.pos -= 1
                image = cv2.imread(self.img_list[self.pos])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                """
                이미지 처리
                """
                self.openImage(image=self.toQImage(image))
                # print('\r' + self.img_list[self.pos], end="")

        elif e.key() == 68:
            self.pos += 1
            if self.total == self.pos:
                self.pos -= 1
            image = cv2.imread(self.img_list[self.pos])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            """
            이미지 처리
            """
            self.openImage(image=self.toQImage(image))
            # print('\r' + self.img_list[self.pos], end="")

        try:
            self.lb_file_name.setText(self.img_list[self.pos])

            self.image_shape = image.shape[0:2]
            self.cx, self.cy = self.image_shape[0]/2, self.image_shape[1]/2

            self.vp = vp.find_theta(image)
            self.K, self.R, self.t, self.cam_ori = vh_dist.cam_orientation(self.vp, self.cx, self.cy, self.f, self.L)

            self.lb_file_name.setText(self.img_list[self.pos])
            self.lb_cam_intrinsic.setText(
                f"Image Shape: {self.image_shape}, focal: {self.f}, cx:{self.cx}, cy:{self.cy}, L(m):{self.L}, vp: {self.vp}, cam_orig(deg): {self.cam_ori * 180 / 3.14}")
        except:
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    imageViewer = ImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())