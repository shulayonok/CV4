import numpy as np
import configuration as config
from tqdm import trange
import time

func = lambda x, y, center, sigma: np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)) / (
        2 * np.pi * sigma ** 2)


def get_cum_matrix(arr):
    coordsArr = np.argwhere(arr == 255)
    np.random.shuffle(coordsArr)
    coordsArr = coordsArr[: len(coordsArr) // 5]
    diagonal = int(np.sqrt(arr.shape[0] ** 2 + arr.shape[1] ** 2))
    cum_matrix = np.zeros((arr.shape[0], arr.shape[1], diagonal))
    for pix in trange(coordsArr.shape[0]):
        for a in range(arr.shape[0]):
            for b in range(arr.shape[1]):
                r = round(np.sqrt((coordsArr[pix][1] - a) ** 2 + (coordsArr[pix][0] - b) ** 2))
                if r < diagonal:
                    cum_matrix[a][b][r] += 1
    return cum_matrix


def suppression(arr):
    X, Y, Z = arr.shape
    result = np.zeros((X, Y, Z))
    time.sleep(0.5)
    for i in trange(1, X - 1):
        for j in range(1, Y - 1):
            for k in range(1, Z - 1):
                cube = arr[i - 1: i + 2, j - 1: j + 2, k - 1: k + 2]
                count = np.sum(cube)
                if count > config.threshold2:
                    centroid = np.unravel_index(np.argmax(cube, axis=None), cube.shape)
                    if centroid == (1, 1, 1):
                        result[i][j][k] = 255
    print(np.count_nonzero(result))
    return result


def drawing(arr, bord, img):
    coordsArr = np.argwhere(arr == 255)
    for point in trange(coordsArr.shape[0]):
        for x in range(bord.shape[1]):
            if coordsArr[point][2] ** 2 - (x - coordsArr[point][0]) ** 2 >= 0:
                y1 = round(np.sqrt(coordsArr[point][2] ** 2 - (x - coordsArr[point][0]) ** 2) + coordsArr[point][1])
                y2 = round(-np.sqrt(coordsArr[point][2] ** 2 - (x - coordsArr[point][0]) ** 2) + coordsArr[point][1])
                if 0 <= y1 < bord.shape[0]:
                    img[y1, x] = [255, 0, 0]
                if 0 <= y2 < bord.shape[0]:
                    img[y2, x] = [255, 0, 0]


# Наложение фильтра
def filter(arr, shape):
    X, Y, U = arr.shape
    center = shape // 2
    borderX, borderY, borderZ = arr.shape
    # Добавляем рамку
    borderX += center * 2
    borderY += center * 2
    result = np.zeros((borderX, borderY, borderZ), dtype=np.uint8)
    # Внутрь помещаем изображение
    result[center:-center, center:-center] = arr
    # Генерим фильтр
    matrix = gauss(shape)
    # Накладываем фильтр
    for i in trange(X):
        for j in range(Y):
            for k in range(U):
                result[i + center, j + center, k] = int(np.sum(result[i:i + shape, j:j + shape, k] * matrix))
    return result[center:-center, center:-center]


# Формирование фильтра определённой размерности
def gauss(shape):
    matrix = np.zeros((shape, shape))
    center = shape // 2
    for i in range(shape):
        for j in range(shape):
            matrix[i, j] = func(i, j, center, config.sigma)
    matrix /= np.sum(matrix)
    return matrix


def apply(img, img1):
    print("Заполнение кумулятивной матрицы")
    cum_matrix = get_cum_matrix(img1)
    print("Сглаживание")
    cum_matrix = filter(cum_matrix, config.gauss_shape)
    print("Подавление немаксимумов")
    cum_matrix = suppression(cum_matrix)
    print("Отрисовка кругалей")
    drawing(cum_matrix, img1, img)
    print("Кругаля тоже")
