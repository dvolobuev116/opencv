import cv2 as cv
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QImage, qRgb, QPixmap
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget, qApp, QAction, QPushButton

gray_color_table = [qRgb(i, i, i) for i in range(256)]

timer = QtCore.QTimer()
success = True
vidcap = cv.VideoCapture(
    'D:\\Видеонаблюдение\\date 2017.06.01\\object_code 720594\\aud 0021\\channel eeb96cbc-ecd3-11e4-aac3-f0def1c12a34-72059401\\02 eeb96cbc-ecd3-11e4-aac3-f0def1c12a34-72059401-1496286000-1496289600.mp4')


# Наследуемся от QMainWindow
class MainWindow(QMainWindow):
    # Переопределяем конструктор класса

    def __init__(self):
        # Обязательно нужно вызвать метод супер класса
        QMainWindow.__init__(self)

        self.setMinimumSize(QSize(480, 320))  # Устанавливаем размеры
        self.setWindowTitle("Hello world!!!")  # Устанавливаем заголовок окна
        central_widget = QWidget(self)  # Создаём центральный виджет
        self.setCentralWidget(central_widget)  # Устанавливаем центральный виджет

        grid_layout = QGridLayout(self)  # Создаём QGridLayout
        central_widget.setLayout(grid_layout)  # Устанавливаем данное размещение в центральный виджет

        self.image_label = QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)  # Устанавливаем позиционирование текста
        grid_layout.addWidget(self.image_label, 0, 0)  # и добавляем его в размещение

        self.play_button = QPushButton()
        self.play_button.setText("play")
        grid_layout.addWidget(self.play_button, 1, 0)

        self.render_button = QPushButton()
        self.render_button.setText("render")
        grid_layout.addWidget(self.render_button, 2, 0)

        exit_action = QAction("&Exit", self)  # Создаём Action с помощью которого будем выходить из приложения
        exit_action.setShortcut('Ctrl+Q')  # Задаём для него хоткей
        # Подключаем сигнал triggered к слоту quit у qApp.
        # синтаксис сигналов и слотов в PyQt5 заметно отличается от того,
        # который используется Qt5 C++
        exit_action.triggered.connect(qApp.quit)
        # Устанавливаем в панель меню данный Action.
        # Отдельного меню создавать пока не будем.
        file_menu = self.menuBar()
        file_menu.addAction(exit_action)

    def toQImage(self, im, copy=False):
        if im is None:
            return QImage()

        if im.dtype == np.uint8:
            if len(im.shape) == 2:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
                qim.setColorTable(gray_color_table)
                return qim.copy() if copy else qim

            elif len(im.shape) == 3:
                if im.shape[2] == 3:
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
                    return qim.copy() if copy else qim
                elif im.shape[2] == 4:
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32)
                    return qim.copy() if copy else qim


def play_action():
    if timer.isActive():
        timer.stop()
    else:
        timer.start()


def timer_action():
    # success, image = vidcap.read()
    # edges = cv.Canny(image, 100, 200)
    # im = mw.toQImage(edges)
    # mw.image_label.setPixmap(QPixmap.fromImage(im))

    success, img = vidcap.read()

    h, w = img.shape[:2]
    img = img[20:h, 0:w]

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    pp_lines(edges, img)

    im = mw.toQImage(img)
    mw.image_label.setPixmap(QPixmap.fromImage(im))


def s_lines(edges, img):
    lines = cv.HoughLines(edges, 1, np.pi / 180, 120)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)


def pp_lines(edges, img):
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=50)
    h, w = img.shape[:2]
    left_border_line = [0, 0, 0, h - 1]
    top_border_line = [0, 0, w - 1, 0]
    right_border_line = [w - 1, 0, w - 1, h - 1]
    bottom_border_line = [0, h - 1, w - 1, h - 1]
    h_lines = []
    v_lines = []

    x1, y1, x2, y2 = left_border_line
    for line in lines:
        x3, y3, x4, y4 = line[0]
        x12 = x1 - x2
        x34 = x3 - x4
        y12 = y1 - y2
        y34 = y3 - y4
        c = x12 * y34 - y12 * x34
        if np.math.fabs(c) >= 0.01:
            a = x1 * y2 - y1 * x2
            b = x3 * y4 - y3 * x4
            x = (a * x34 - b * x12) / c
            y = (a * y34 - b * y12) / c
            if 0 < y < h:
                h_lines.append([x3, y3, x4, y4, y])

    x1, y1, x2, y2 = top_border_line
    for line in lines:
        x3, y3, x4, y4 = line[0]
        x12 = x1 - x2
        x34 = x3 - x4
        y12 = y1 - y2
        y34 = y3 - y4
        c = x12 * y34 - y12 * x34
        if np.math.fabs(c) >= 0.01:
            a = x1 * y2 - y1 * x2
            b = x3 * y4 - y3 * x4
            x = (a * x34 - b * x12) / c
            y = (a * y34 - b * y12) / c
            if 0 < x < w:
                v_lines.append([x3, y3, x4, y4, x])

    h_lines.append(top_border_line + [0])
    h_lines.append(bottom_border_line + [h - 1])
    h_lines.sort(key=lambda e: e[4])
    r = 32
    for line in h_lines:
        x1, y1, x2, y2 = line[:4]
        r12 = np.math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
        k1 = -1000 / r12
        k2 = 1000 / r12
        x3 = int(x1 + (x2 - x1) * k1)
        y3 = int(y1 + (y2 - y1) * k1)
        x4 = int(x2 + (x2 - x1) * k2)
        y4 = int(y2 + (y2 - y1) * k2)
        cv.line(img, (x3, y3), (x4, y4), (r % 256, 0, 0), 1)
        r += 32

    v_lines.append(left_border_line + [0])
    v_lines.append(right_border_line + [w - 1])
    v_lines.sort(key=lambda e: e[4])
    g = 32
    for line in v_lines:
        x1, y1, x2, y2 = line[:4]
        r12 = np.math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
        k1 = -1000 / r12
        k2 = 1000 / r12
        x3 = int(x1 + (x2 - x1) * k1)
        y3 = int(y1 + (y2 - y1) * k1)
        x4 = int(x2 + (x2 - x1) * k2)
        y4 = int(y2 + (y2 - y1) * k2)
        cv.line(img, (x3, y3), (x4, y4), (0, g % 256, 0), 1)
        g += 32

    regions = []
    for i in range(1, len(h_lines)):
        for j in range(1, len(v_lines)):
            left_top = get_intersection_point(h_lines[i - 1][:4], v_lines[j - 1][:4])
            left_bottom = get_intersection_point(h_lines[i][:4], v_lines[j - 1][:4])
            right_top = get_intersection_point(h_lines[i - 1][:4], v_lines[j][:4])
            right_bottom = get_intersection_point(h_lines[i][:4], v_lines[j][:4])
            if left_top is not None and left_bottom is not None and right_top is not None and right_bottom is not None:
                regions.append([left_top, right_top, right_bottom, left_bottom])
    regions = regions


def get_intersection_point(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    x12 = x1 - x2
    x34 = x3 - x4
    y12 = y1 - y2
    y34 = y3 - y4
    c = x12 * y34 - y12 * x34
    if np.math.fabs(c) >= 0.01:
        a = x1 * y2 - y1 * x2
        b = x3 * y4 - y3 * x4
        x = (a * x34 - b * x12) / c
        y = (a * y34 - b * y12) / c
        return [int(x), int(y)]
    return None


def p_lines(edges, img):
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=50)
    h, w = img.shape[:2]
    lines = np.append(lines, [[[1, 1, 1, h - 2]]], axis=0)
    lines = np.append(lines, [[[1, 1, w - 2, 1]]], axis=0)
    lines = np.append(lines, [[[1, h - 2, w - 2, h - 2]]], axis=0)
    lines = np.append(lines, [[[w - 2, 1, w - 2, h - 2]]], axis=0)
    i_points = []
    g = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]

        for line1 in lines:
            x3, y3, x4, y4 = line1[0]
            x12 = x1 - x2
            x34 = x3 - x4
            y12 = y1 - y2
            y34 = y3 - y4
            c = x12 * y34 - y12 * x34
            if np.math.fabs(c) >= 0.01:
                a = x1 * y2 - y1 * x2
                b = x3 * y4 - y3 * x4
                x = (a * x34 - b * x12) / c
                y = (a * y34 - b * y12) / c
                i_points.append((x, y))

        r12 = np.math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
        k1 = -1000 / r12
        k2 = 1000 / r12
        x3 = int(x1 + (x2 - x1) * k1)
        y3 = int(y1 + (y2 - y1) * k1)
        x4 = int(x2 + (x2 - x1) * k2)
        y4 = int(y2 + (y2 - y1) * k2)
        cv.line(img, (x3, y3), (x4, y4), (0, g % 256, 0), 1)
        g += 16

        for px, py in i_points:
            if 0 < int(px) < img.shape[1]:
                if 0 < int(py) < img.shape[0]:
                    img[int(py)][int(px)] = [255, 0, 0]


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.play_button.clicked.connect(play_action)
    mw.render_button.clicked.connect(timer_action)
    mw.show()

    timer.timeout.connect(timer_action)
    # timer.start(10)

    # print(cv.__version__)
    # ret = vidcap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    # ret = vidcap.set(cv.CAP_PROP_FRAME_HEIGHT, 220)

    sys.exit(app.exec())
