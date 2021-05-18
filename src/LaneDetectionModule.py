import cv2
import numpy as np
import utils

curveList = []
avgVal = 10


def getLaneCurve(img, display=0):
    imgResult = img.copy()
    ### STEP 1
    # отделяем нашу белую полосу дороги от всего остального
    imgThres = utils.thresholding(img)

    ### STEP 2
    # развернем каждый кадр так, чтобы был вид сверху. Так мы поймем, насколько точно происходит поворот на нашей полосе движения
    hT, wT, c = img.shape

    global intialTrackbarVals

    widthTop = intialTrackbarVals[0]
    heightTop = intialTrackbarVals[1]
    widthBottom = intialTrackbarVals[2]
    heightBottom = intialTrackbarVals[3]

    # начальные точки кадра, которые мы размернем в прямоугольник
    points = np.float32([(widthTop, heightTop), (wT - widthTop, heightTop),
                         (widthBottom, heightBottom), (wT - widthBottom, heightBottom)])
    imgWarp = utils.warpImg(imgThres, points, wT, hT)

    # для поиска точек Perspective Transformation
    imgCopy = img.copy()
    # points = utils.valTrackbars()
    imgWarpPoints = utils.drawPoints(imgCopy, points)
    # cv2.imshow('WarpPoints', imgWarpPoints)

    ### STEP 3
    # находим середину всей картинки и середину нижней части изображения,
    # если мы их вычтем, то получим значение поворота
    middlePoint, imgHist = utils.getHistogram(imgWarp, display=True, minVal=0.5, region=3)
    curveAveragePoint, imgHist = utils.getHistogram(imgWarp, display=True, minVal=0.9, region=1)
    curveRaw = curveAveragePoint - middlePoint

    ### STEP 4
    # храним только последние avgVal значений поворота и усредняем их, чтобы понять какой надо сделать плавный поворот
    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList) / len(curveList))

    ### STEP 5 Display results
    if display != 0:
        imgInvWarp = utils.warpImg(imgWarp, points, wT, hT, inv=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:hT // 3, 0:wT] = 0, 0, 0
        imgLaneColor = np.zeros_like(img)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
        midY = 450
        cv2.putText(imgResult, str(curve), (wT // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
        cv2.line(imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5)
        cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY - 25), (wT // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)
        for x in range(-30, 30):
            w = wT // 20
            cv2.line(imgResult, (w * x + int(curve // 50), midY - 10),
                     (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)
       # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
       # cv2.putText(imgResult, 'FPS ' + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 50, 50), 3);
    if display == 2:
        imgStacked = utils.stackImages(0.7, ([img, imgWarpPoints, imgWarp],
                                             [imgHist, imgLaneColor, imgResult]))
        cv2.imshow('ImageStack', imgStacked)
    elif display == 1:
        cv2.imshow('Result', imgResult)

   # cv2.imshow('Thres', imgThres)
   # cv2.imshow('imgWarp', imgWarp)
   # cv2.imshow('Hist', imgHist)

    # Нормализация
    curve = curve/ 100
    if curve > 1: curve = 1
    if curve < -1: curve = -1

    return curve


if __name__ == '__main__':
    cap = cv2.VideoCapture('data/vid1.mp4')

    # получили точки, которые надо развернуть в прямоугольник, опытным путем
    intialTrackbarVals = [96, 107, 41, 218]

    # для поиска точек Perspective Transformation
    # utils.initializeTrackbars(intialTrackbarVals)

    frameCounter = 0

    while True:

        # делаем бесконечный луп видео
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        success, img = cap.read()
        img = cv2.resize(img, (480, 240))

        curve = getLaneCurve(img, display=2)
        print(curve)

       # cv2.imshow('vid', img)

        if (cv2.waitKey(1) & 0xFF == 27):
            break

    cap.release()
    cv2.destroyAllWindows()
