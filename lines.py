import matplotlib.pyplot as plt
import numpy as np
import canny
import otsu
import configuration as config


def get_cum_matrix(arr):
    coordsArr = np.argwhere(arr == 255)
    np.random.shuffle(coordsArr)
    coordsArr = coordsArr[: len(coordsArr) // 2]
    diagonal = int(np.sqrt(arr.shape[0] ** 2 + arr.shape[1] ** 2))
    cum_matrix = np.zeros((diagonal, 271))
    for pix in range(coordsArr.shape[0]):
        for angle in range(-90, 181):
            p = round(
                coordsArr[pix][1] * np.cos(angle * np.pi / 180) + coordsArr[pix][0] * np.sin(angle * np.pi / 180))
            cum_matrix[p][angle] += 1
    return cum_matrix


def suppression(arr):
    X, Y = arr.shape
    result = np.zeros((X, Y))
    for i in range(1, X - 1):
        for j in range(1, Y - 1):
            square = arr[i - 1: i + 2, j - 1: j + 2]
            count = np.sum(square)
            if count > config.threshold:
                centroid = np.unravel_index(np.argmax(square, axis=None), square.shape)
                if centroid == (1, 1):
                    result[i][j] = 255
    return result


def drawing(arr, bord, img):
    coordsArr = np.argwhere(arr == 255)
    for point in range(coordsArr.shape[0]):
        for x in range(bord.shape[1]):
            y = round((coordsArr[point][0] - x * np.cos(coordsArr[point][1] * (np.pi / 180))) / np.sin(
                coordsArr[point][1] * (np.pi / 180)))
            if 0 <= y < bord.shape[0]:
                img[y, x] = [255, 0, 0]


def apply(image):
    # plt.subplot(1, 2, 1)
    # plt.imshow(image)

    # Границы через Кэнни и бинаризацию
    # image1 = canny.apply(image)
    image1 = otsu.apply(image)
    # plt.subplot(1, 2, 2)
    plt.imshow(image1, cmap='gray')
    plt.show()

    # Получаем сглаженный куммулятивный массив
    cum_matrix = canny.filter(get_cum_matrix(image1), config.gauss_shape)
    plt.subplot(1, 2, 1)
    plt.imshow(cum_matrix, cmap='gray')

    # Подавление немаксимумов для куммулятивного массива
    cum_matrix = suppression(cum_matrix)
    plt.subplot(1, 2, 2)
    plt.imshow(cum_matrix, cmap='gray')
    plt.show()

    # Отрисовка линий
    drawing(cum_matrix, image1, image)
    print("Линии нарисованы")

    return image1
