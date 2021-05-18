import cv2
import numpy as np


def thresholding(img):
    imhHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([80, 0, 0])
    upperWhite = np.array([255, 160, 255])

    # красит белым те пиксели, которые лежат между lowerWhite и upperWhite
    # lowerWhite = (low_H, low_S, low_V)
    # upperWhite = (high_H, high_S, high_V)
    maskWhite = cv2.inRange(imhHSV, lowerWhite, upperWhite)
    return maskWhite


def warpImg(img, points, w, h, inv=False):
    # процедура принимает 4 точки pts1, которые например отразуют трапецию и разворачивает их в прямоугольник,
    # координаты которого обозначены в pts2. Начальная координата (0, 0) и высота h и ширина w.
    # то есть делается преобразование. В данном случае нам необходимо нашу картинку преобразовать в
    # вид сверху, чтобы точно понимать, сколько поворачивать вправо или влево.
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    return imgWarp


def nothing(a):
    pass


def initializeTrackbars(intialTracbarVals, wT=480, hT=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 420, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0], wT // 2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2], wT // 2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, nothing)


def valTrackbars(wT=480, hT=240):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT - widthTop, heightTop),
                         (widthBottom, heightBottom), (wT - widthBottom, heightBottom)])
    return points


def drawPoints(img, points):
    for x in range(0, 4):
        cv2.circle(img, (int(points[x][0]), int(points[x][1])), 15, (0, 0, 255), cv2.FILLED)
    return img


def getHistogram(img, display=False, minVal=0.1, region=1):
    # находим сумму значений пикселей для каждого столбца в изображении.
    # если сумма в столбце большая, значит там много белых пикселей=255, значит идет поворот в эту сторону
    # мы делим на region, потому что мы хотим знать середину нашей полосы. Если мы возьмем все изображение,
    # то при резком повороте середина сместится с середины полосы в сторону поворота, а нам нужна именно середина полосы,
    # поэтому мы возьмем нижнюю часть изображения и суммируем ее, чтобы найти середину полосы
    if region == 1:
        histValues = np.sum(img, axis=0)
    else:
        # histValues = np.sum(img[img.shape[0]//region:,:], axis=0)
        histValues = np.sum(img[img.shape[0] - img.shape[0] // region:, :], axis=0)

    # находим макимльное значение
    maxValue = np.max(histValues)
    # все, что меньше минимального значения будем считать шумом
    minValue = minVal * maxValue

    # оставляем индексы столбцов с суммами выше минимального
    indexArray = np.where(histValues >= minValue)  # ALL INDICES WITH MIN VALUE OR ABOVE
    # print(indexArray)

    # среднее и будет серединой нашей полосы
    basePoint = int(np.average(indexArray))  # AVERAGE ALL MAX INDICES VALUES
    # print(basePoint)

    if display:
        imgHist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for x, intensity in enumerate(histValues):
            # print(intensity)
            if intensity > minValue:
                # фиолетовый
                color = (255, 0, 255)
            else:
                # красный
                color = (0, 0, 255)
            # линия из точки 0, 240 в точку 0, интенсивность/255. Еще делим на регион, чтобы показать, что мы не все изображение смотрим, в только ту часть,
            # которую разворачиваем в прямоугольник
            cv2.line(imgHist, (x, img.shape[0]), (x, img.shape[0] - (intensity // 255 // region)), color, 1)
        # х - это середина нашей полосы, а y - это высота изображения
        cv2.circle(imgHist, (basePoint, img.shape[0]), 20, (0, 255, 255), cv2.FILLED)
        return basePoint, imgHist

    return basePoint


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver
