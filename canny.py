import sys
import numpy as np
import configuration as config

sys.setrecursionlimit(10**5)

func = lambda x, y, center, sigma: np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)) / (
        2 * np.pi * sigma ** 2)


# ЧБ
def black_and_white(arr):
    X, Y, U = arr.shape
    result = np.zeros((X, Y), dtype=int)
    for i in range(X):
        for j in range(Y):
            result[i, j] = np.mean(arr[i, j])
    return np.array(result, dtype=np.uint8)


# Формирование фильтра Гаусса определённой размерности
def gauss(shape):
    matrix = np.zeros((shape, shape))
    center = shape // 2
    for i in range(shape):
        for j in range(shape):
            matrix[i, j] = func(i, j, center, config.sigma)
    matrix /= np.sum(matrix)
    return matrix


# Наложение фильтра Гаусса
def filter(arr, shape):
    X, Y = arr.shape
    center = shape // 2
    borderX, borderY = arr.shape
    # Добавляем рамку
    borderX += center * 2
    borderY += center * 2
    result = np.zeros((borderX, borderY), dtype=np.uint8)
    # Внутрь помещаем изображение
    result[center:-center, center:-center] = arr
    # Генерим фильтр
    matrix = gauss(shape)
    # Накладываем фильтр
    for i in range(X):
        for j in range(Y):
            result[i + center, j + center] = int(np.sum(result[i:i + shape, j:j + shape] * matrix))
    return result[center:-center, center:-center]


# Магнитуда и направление из оценок частных производных для всей картинки
def gradient(arr):
    X, Y = arr.shape
    angles = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Magn, Dir = np.zeros((X, Y)), np.zeros((X, Y))
    for i in range(1, X - 1):
        for j in range(1, Y - 1):
            neighbours = arr[i - 1: i + 2, j - 1: j + 2]
            # Получаем оценки частных производных
            Ix = np.sum(Gx * neighbours)
            Iy = np.sum(Gy * neighbours)
            # По ним оцениваем магнитуду и направление
            magn, dir = magnitude_and_direction(Ix, Iy)
            # Направление округляем до кратности 45 градусов
            dir = round(dir, angles)
            Magn[i][j], Dir[i][j] = magn, dir
    return Magn.astype(int), Dir


# Магнитуда и направление градиента
def magnitude_and_direction(Ix, Iy):
    magnitude = np.sqrt(Ix ** 2 + Iy ** 2)
    direction = np.arctan2(Iy, Ix) * 180 / np.pi
    return magnitude, direction


# Округление направлений
def round(dir, angles):
    if dir < 0:
        dir += 360
    elif dir > 337.5:
        return 0
    return min(angles, key=lambda x: abs(x - dir))


# Подавление немаксимумов
def suppression(magn, dir):
    for i in range(1, magn.shape[0] - 1):
        for j in range(1, magn.shape[1] - 1):
            if dir[i][j] == 0 or 180:
                if magn[i][j] <= magn[i][j - 1] or magn[i][j] <= magn[i][j + 1]:
                    magn[i][j] = 0
            elif dir[i][j] == 90 or 270:
                if magn[i][j] <= magn[i - 1][j] or magn[i][j] <= magn[i + 1][j]:
                    magn[i][j] = 0
            elif dir[i][j] == 45 or 225:
                if magn[i][j] <= magn[i - 1][j + 1] or magn[i][j] <= magn[i + 1][j - 1]:
                    magn[i][j] = 0
            else:
                if magn[i][j] <= magn[i - 1][j - 1] or magn[i][j] <= magn[i + 1][j + 1]:
                    magn[i][j] = 0


# Рекурсивно двигаемся от пикселя для которого 𝑔(𝑝) > 𝑇ℎ𝑖𝑔ℎ
def search(magn, borders, t_low, x, y):
    square = magn[x - 1: x + 2, y - 1: y + 2]
    for m in range(-1, 2):
        for n in range(-1, 2):
            if (m == 0 and n == 0) or (x + m <= 0 or x + m >= magn.shape[0] - 1) or (
                    y + n <= 0 or y + n >= magn.shape[1] - 1):
                continue
            if square[m][n] > t_low and borders[x + m][y + n] == 0:
                borders[x + m][y + n] = 255
                search(magn, borders, t_low, x + m, y + n)
    return


def clarification(magn):
    borders = np.zeros((magn.shape[0], magn.shape[1]), dtype=int)
    for i in range(magn.shape[0] - 1):
        for j in range(1, magn.shape[1] - 1):
            if magn[i][j] > config.t_high and borders[i][j] == 0:
                borders[i][j] = 255
                search(magn, borders, config.t_low, i, j)
    return borders


def apply(img):
    img = black_and_white(img)
    img = filter(img, config.gauss_shape)
    img, dir = gradient(img)
    suppression(img, dir)
    img = clarification(img)
    return img


