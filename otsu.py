import numpy as np
import configuration as config

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


def gradient(arr):
    X, Y = arr.shape
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
            magn = np.sqrt(Ix ** 2 + Iy ** 2)
            # Направление округляем до кратности 45 градусов
            Magn[i][j] = magn
    return Magn.astype(int)


def otsu_binarization(arr):
    X, Y = arr.shape
    max_sigma = 0
    max_t = 0
    for t in range(1, 255):
        class0 = arr[np.where(arr < t)]
        mean0 = np.mean(class0) if len(class0) > 0 else 0
        weight0 = len(class0) / (X * Y)
        class1 = arr[np.where(arr >= t)]
        mean1 = np.mean(class1) if len(class1) > 0 else 0
        weight1 = len(class1) / (X * Y)
        sigma = weight0 * weight1 * ((mean0 - mean1) ** 2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = t
    arr[arr < max_t] = 0
    arr[arr >= max_t] = 255


def apply(img):
    img = black_and_white(img)
    img = filter(img, config.gauss_shape)
    img = gradient(img)
    otsu_binarization(img)
    return img
