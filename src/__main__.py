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
np.seterr(all="warn")
np.seterr(all="raise")


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
        self.image_label.setScaledContents(True)
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
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888).rgbSwapped()
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
    regions = pp_lines(edges, img)

    im = mw.toQImage(img)
    mw.image_label.setPixmap(QPixmap.fromImage(im))


def render_action():
    src = cv.imread("D:\\opencv\\IMG_20181121_182224.jpg")

    edgeThresh = 1
    lowThreshold = 100
    max_lowThreshold = 100
    ratio = 3
    kernel_size = 3
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    dst = cv.blur(gray, (5, 5))
    edges = cv.Canny(dst, lowThreshold, lowThreshold * ratio, kernel_size)
    regions = pp_lines(edges, src)

    templ_src = cv.imread("D:\\opencv\\IMG_20181121_182353.png")
    generate_templates(templ_src, regions, "D:\\opencv\\temp\\")

    im = mw.toQImage(src)
    mw.image_label.setPixmap(QPixmap.fromImage(im))


def pp_lines(edges, img):
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=50)
    h, w = img.shape[:2]
    left_border_line = [0, 0, 0, h - 1]
    top_border_line = [0, 0, w - 1, 0]
    right_border_line = [w - 1, 0, w - 1, h - 1]
    bottom_border_line = [0, h - 1, w - 1, h - 1]
    h_lines = []
    v_lines = []

    except_lines = []
    for line in lines:
        x3, y3, x4, y4 = line[0]
        pi = get_intersection_point(left_border_line, line[0])
        if pi is not None and 0 < pi[1] < h:
            h_lines.append([x3, y3, x4, y4, pi[1]])
            except_lines += [[x3, y3, x4, y4]]

    for line in lines:
        x3, y3, x4, y4 = line[0]
        pi = get_intersection_point(top_border_line, line[0])
        if pi is not None and 0 < pi[0] < w and [x3, y3, x4, y4] not in except_lines:
            v_lines.append([x3, y3, x4, y4, pi[0]])

    i = 0
    l = len(h_lines)
    while i < l:
        j = i + 1
        while j < l:
            pi = get_intersection_point(h_lines[i][:4], h_lines[j][:4])
            if pi is None or np.math.fabs(pi[0]) > 100000 or np.math.fabs(pi[1]) > 100000:
                h_lines.remove(h_lines[j])
                l = len(h_lines)
            else:
                j += 1
        i += 1

    regions = []
    for i in range(0, len(h_lines)):
        for j in range(0, len(v_lines)):
            left_top = get_intersection_point(h_lines[i - 1][:4], v_lines[j - 1][:4])
            left_bottom = get_intersection_point(h_lines[i][:4], v_lines[j - 1][:4])
            right_top = get_intersection_point(h_lines[i - 1][:4], v_lines[j][:4])
            right_bottom = get_intersection_point(h_lines[i][:4], v_lines[j][:4])
            if left_top is not None and left_bottom is not None and right_top is not None and right_bottom is not None:
                regions.append([left_top, right_top, right_bottom, left_bottom])

    h_lines.append(top_border_line + [0])
    h_lines.append(bottom_border_line + [h - 1])
    h_lines.sort(key=lambda e: e[4])

    v_lines.append(left_border_line + [0])
    v_lines.append(right_border_line + [w - 1])
    v_lines.sort(key=lambda e: e[4])

    l_scale = 1
    for line in h_lines:
        x1, y1, x2, y2 = line[:4]
        r12 = np.math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
        k1 = -l_scale / r12
        k2 = l_scale / r12
        x3 = int(x1 + (x2 - x1) * k1)
        y3 = int(y1 + (y2 - y1) * k1)
        x4 = int(x2 + (x2 - x1) * k2)
        y4 = int(y2 + (y2 - y1) * k2)
        cv.line(img, (x3, y3), (x4, y4), (255, 0, 0), 3)

    for line in v_lines:
        x1, y1, x2, y2 = line[:4]
        r12 = np.math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
        k1 = -l_scale / r12
        k2 = l_scale / r12
        x3 = int(x1 + (x2 - x1) * k1)
        y3 = int(y1 + (y2 - y1) * k1)
        x4 = int(x2 + (x2 - x1) * k2)
        y4 = int(y2 + (y2 - y1) * k2)
        cv.line(img, (x3, y3), (x4, y4), (0, 255, 0), 3)

    return regions


def points_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.math.sqrt(np.math.pow(x2 - x1, 2) + np.math.pow(y2 - y1, 2))


def get_intersection_point(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    x12 = x1 - x2
    x34 = x3 - x4
    y12 = y1 - y2
    y34 = y3 - y4
    c = x12 * y34 - y12 * x34
    if np.math.fabs(c) >= 0.001:
        a = x1 * y2 - y1 * x2
        b = x3 * y4 - y3 * x4
        try:
            x = (a * x34 - b * x12) / c
            y = (a * y34 - b * y12) / c
        except Exception:
            return None
        return [int(x), int(y)]
    return None


def generate_templates(src_img, regions, output_path=None):
    templates = []
    sh, sw = src_img.shape[:2]

    input_quad = np.float32([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]])

    input_center_point = get_intersection_point(
        [input_quad[0][0], input_quad[0][1], input_quad[2][0], input_quad[2][1]],
        [input_quad[1][0], input_quad[1][1], input_quad[3][0], input_quad[3][1]])

    rc = 0
    for region in regions:
        region_center_point = get_intersection_point(
            region[0] + region[2],
            region[1] + region[3])
        x_offset = input_center_point[0] - region_center_point[0]
        y_offset = input_center_point[1] - region_center_point[1]
        region_center_point = [region_center_point[0] + x_offset, region_center_point[1] + y_offset]
        offset_region = []
        for p in region:
            new_p = [p[0] + x_offset, p[1] + y_offset]
            offset_region += [new_p]

        sides = {0: [[3, 0], [0, 1]], 1: [[0, 1], [1, 2]], 2: [[1, 2], [2, 3]], 3: [[2, 3], [3, 0]], }
        # min_d = None
        # коэфициент изменения длины
        mm = None
        for p in offset_region:
            min_d = None
            min_d_p = None
            for i in sides[offset_region.index(p)]:
                # сторона исходного изображения
                il = [input_quad[i[0]][0], input_quad[i[0]][1], input_quad[i[1]][0], input_quad[i[1]][1]]
                # точка пересечения диагонали внутреннего сегмента
                il_ip = get_intersection_point([region_center_point[0], region_center_point[1], p[0], p[1]], il)
                # растояние до точки пересечения
                if il_ip is not None:
                    d = points_distance(p, il_ip)
                    # m = points_distance(region_center_point, il_ip) / points_distance(region_center_point, p)
                    if min_d is None or min_d > d:
                        min_d = d
                        min_d_p = il_ip
                        # m = points_distance(region_center_point, il_ip) / points_distance(region_center_point, p)
                        # m = m
            m = points_distance(region_center_point, min_d_p) / points_distance(region_center_point, p)
            if mm is None or m < mm:
                mm = m

        for p in offset_region:
            p[0] = np.math.floor(region_center_point[0] + (p[0] - region_center_point[0]) * mm)
            p[1] = np.math.floor(region_center_point[1] + (p[1] - region_center_point[1]) * mm)

        output_quad = np.float32(offset_region)
        m = cv.getPerspectiveTransform(input_quad, output_quad)
        output = cv.warpPerspective(src_img, m, (sh, sw))
        templates += [output]
        if output_path is not None:
            cv.imwrite(output_path + "im" + rc.__str__() + ".jpg", output)
        rc += 1
    return templates


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.play_button.clicked.connect(play_action)
    mw.render_button.clicked.connect(render_action)
    mw.show()

    timer.timeout.connect(timer_action)
    # timer.start(10)

    # print(cv.__version__)
    # ret = vidcap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    # ret = vidcap.set(cv.CAP_PROP_FRAME_HEIGHT, 220)

    sys.exit(app.exec())
