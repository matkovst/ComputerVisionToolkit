import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse
import math

U = np.array([ [1/math.sqrt(2), -1/math.sqrt(2), 0], [1/math.sqrt(6), 1/math.sqrt(6), -2/math.sqrt(6)] ]) # orthogonal matrix

def ShannonEntropy(I, bandwidth = 1):
    nbins = round((np.max(I) - np.min(I)) / bandwidth)
    P = np.histogram(I, nbins)[0] / I.size
    P = P[P != 0]
    return -np.sum(P * np.log2(P))

def projectOntoPlane(Rho, orthMatrix):
    return Rho @ orthMatrix.T

def getProjectedImage(Rho, radAngle):
    Chi = projectOntoPlane(Rho, U)
    I = Chi[:,:,0] * np.cos(radAngle) + Chi[:,:,1] * np.sin(radAngle)
    I = np.exp(I)
    return I

def convertToLogChromacitySpace(img):

    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    B[B == 0] = 1.0
    G[G == 0] = 1.0
    R[R == 0] = 1.0

    logChr = np.zeros_like(img, dtype = np.float64)
    gMean = np.power((R * G * B), 1.0/3)
    logChr[:, :, 0] = np.log((B / gMean))
    logChr[:, :, 1] = np.log((G / gMean))
    logChr[:, :, 2] = np.log((R / gMean))

    logChrB = np.atleast_3d(logChr[:, :, 0])
    logChrG = np.atleast_3d(logChr[:, :, 1])
    logChrR = np.atleast_3d(logChr[:, :, 2])
    Rho = np.concatenate((logChrR, logChrG, logChrB), axis = 2) # log chromaticity on a plane
    return Rho

def solveProjection(img, Rho, nangles = 181):
    
    N = img.shape[0] * img.shape[1]
    Chi = projectOntoPlane(Rho, U)

    radians = np.radians(np.linspace(0, 180, nangles))
    Entropies = np.zeros(nangles, dtype = np.float64)
    for i, rad in enumerate(radians):
        I = Chi[:,:,0] * np.cos(rad) + Chi[:,:,1] * np.sin(rad)
        IMean = np.mean(I)
        IStd = np.std(I)
        lbound = IMean + 3.0 * (-IStd)
        rbound = IMean + 3.0 * (+IStd)
        IClipped = np.clip(I, lbound, rbound)
        binWidth = 3.5 * IStd * N**(-1/3)
        Entropies[i] = ShannonEntropy(IClipped, binWidth)

    minEntropy = np.min(Entropies)
    minEntropyIdx = np.argmin(Entropies)
    minEntropyAngle = radians[minEntropyIdx]
    return minEntropy, minEntropyAngle, Entropies


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input image")
    args = ap.parse_args()
    print(">>> Input: ", args.input)

    # Prepare input
    img = cv.imread(args.input)
    img64 = cv.GaussianBlur(img.astype(np.float64), (3, 3), 1)

    # Infer intrinsic image
    Rho = convertToLogChromacitySpace(img64)
    minEntropy, minEntropyAngle, Entropies = solveProjection(img, Rho, nangles = 181)
    I1DAngle = minEntropyAngle + np.pi / 2 # I1DAngle = (158.0 * np.pi / 180.0)
    I1D = getProjectedImage(Rho, I1DAngle)

    # Display results

    print(">>> Min entropy: ", minEntropy)
    print(">>> Min entropy angle (deg): ", np.rad2deg(minEntropyAngle))

    Chi = projectOntoPlane(Rho, U)
    minX1, maxX1 = np.min(Chi[:,:,0]), np.max(Chi[:,:,0])
    minX2, maxX2 = np.min(Chi[:,:,1]), np.max(Chi[:,:,1])
    lightnintDirection = [np.linspace(minX1, maxX1, 2) * np.cos(minEntropyAngle), np.linspace(minX1, maxX1, 2) * np.sin(minEntropyAngle)]
    invatiantDirection = [np.linspace(minX2, maxX2, 2) * np.cos(minEntropyAngle + np.pi / 2), np.linspace(minX2, maxX2, 2) * np.sin(minEntropyAngle + np.pi / 2)]

    # 2D Log Chromacities plot
    figChrom = plt.figure()
    figChrom.set_size_inches(9, 5)
    axChrom = figChrom.add_subplot(111)
    axChrom.set_aspect('equal', 'box')
    axChrom.set_title('2D Log Chromacities')
    axChrom.set_xlabel('χ1')
    axChrom.set_ylabel('χ2', rotation = 0)
    axChrom.scatter(Chi[:,:,0], Chi[:,:,1], marker = '.', color = 'dodgerblue', label = '2D log chromacity point')
    axChrom.plot(lightnintDirection[0], lightnintDirection[1], 'r--', label = 'Lightning direction')
    axChrom.plot(invatiantDirection[0], invatiantDirection[1], 'r-', label = 'Invariant direction')
    axChrom.legend()

    # Entropies plot
    figEntropy = plt.figure()
    figEntropy.set_size_inches(9, 5)
    axEntropy = figEntropy.add_subplot(111)
    axEntropy.set_title('Entropies')
    axEntropy.set_xlabel('angle°')
    axEntropy.set_ylabel('entropy', rotation = 0)
    axEntropy.plot(Entropies, color = 'darkorange', label = 'Invariant direction')
    axEntropy.legend()

    # 1D Invariant image
    figI1D = plt.figure()
    figI1D.set_size_inches(9, 5)
    axI1D = figI1D.add_subplot(111)
    axI1D.set_title('1D illuminant invariant representation')
    axI1D.imshow(I1D, interpolation = 'bilinear', cmap = plt.cm.gray)

    plt.show()
    
    # cv.imshow("I1D", cv.normalize(I1D, 0, 255, cv.NORM_MINMAX))
    # cv.waitKey()

