import numpy as np
import cv2 as cv
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input image")
    args = ap.parse_args()

    print(args.input)
    capture = cv.VideoCapture(args.input)
    frame = capture.read()[1]

    pause = False
    while True:

        # ------------- Controls ----------------
        keyboard = cv.waitKey(10)
        if keyboard == ord('p'):
            if pause:
                pause = False
            else:
                pause = True
        elif keyboard == ord('r'):
            record = True
        elif keyboard == 32:
            if not pause:
                for i in range(10):
                    ret, frame = capture.read()
            
        if pause:
            if keyboard != 32:
                continue
        # ---------------------------------------


        # ------------------------------------------ CORE ------------------------------------------
        ret, rawFrame = capture.read()
        rawFrame = cv.resize(rawFrame, (300, 300))

        gray = cv.cvtColor(rawFrame, cv.COLOR_BGR2GRAY)
        nPixels = gray.size
        nChannels = frame.shape[2]

        # Сглаживаем кадр и считаем первый градиент ∇I(x,y) (или второй градиент ∇(∇I(x,y)))
        ksize = 3
        sigmax = 1
        derivOrder = 1

        frame = rawFrame.astype(np.float32)
        frame = cv.GaussianBlur(frame, (ksize, ksize), sigmax)

        if derivOrder == 1:
            gradx = cv.Sobel(frame, cv.CV_32F, 1, 0)
            grady = cv.Sobel(frame, cv.CV_32F, 0, 1)
            absGradx = cv.convertScaleAbs(gradx)
            absGrady = cv.convertScaleAbs(grady)
            grad = cv.addWeighted(absGradx, 0.5, absGrady, 0.5, 0)
        elif derivOrder == 2:
            grad = cv.Laplacian(frame, cv.CV_32F)
            grad = cv.convertScaleAbs(grad)


        # Устраняем блики
        specularitiesMask = cv.threshold(gray, 254, 255, cv.THRESH_BINARY)[1]
        specularitiesMaskInv = cv.bitwise_not(specularitiesMask)
        for c in range(nChannels):
            grad[:, :, c] = cv.bitwise_and(grad[:, :, c], specularitiesMaskInv)

        p = 1
        lightSourceColor = np.zeros(nChannels, dtype = np.float32)
        for c in range(nChannels):
            lightSourceColor[c] = np.sum(np.power(grad[:, :, c], p))
        lightSourceColor /= nPixels
        lightSourceColor = np.power(lightSourceColor, 1 / p)
        lightSourceColorMagn = np.sqrt(np.sum(np.power(lightSourceColor, 2)))
        # ------------------------------------------ //// ------------------------------------------


        # ----------------- Show -----------------
        print(lightSourceColorMagn)
        cv.imshow('Result', np.hstack((rawFrame, grad)))
        # ----------------- //// -----------------
        

        # ------------- Controls ----------------
        keyboard = cv.waitKey(10)
        if keyboard == ord('q') or keyboard == 27:
            break
        elif keyboard == ord('p'):
            if pause:
                pause = False
            else:
                pause = True
        elif keyboard == ord('s'):
            capture.set(cv.CAP_PROP_POS_MSEC, 0)
        elif keyboard == ord('r'):
            record = True
        elif keyboard == 32:
            if not pause:
                for i in range(10):
                    ret, frame = capture.read()
        # ---------------------------------------

    capture.release()
    cv.destroyAllWindows()
