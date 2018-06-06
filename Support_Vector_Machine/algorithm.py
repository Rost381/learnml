from numpy import *


def selectJrand(i, m):
    j = i  # we want to select any J not equal to i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def simpleSMO(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    #print(dataMatIn)
    labelMat = mat(classLabels).transpose()
    #print(labelMat)
    b = 0
    m, n = shape(dataMatrix) # 100 2
    alphas = mat(zeros((m, 1))) # [100 rows x 1 column's 0]
    iter = 0

    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            '''
            print('dataMatrix:{0}'.format(dataMatrix))
            print('dataMatrix[i, :]:{0}'.format(dataMatrix[i, :]))
            print('dataMatrix * dataMatrix[i, :].T:{0}'.format(dataMatrix * dataMatrix[i, :].T))
            '''
            
            # if checks if an example violates KKT conditions
            Ei = fXi - float(labelMat[i]) # 7.104
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                
                # random select the second alphas[j]
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b 
                Ej = fXj - float(labelMat[j]) # 7.104
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # make sure alpha is between 0 and C page 126
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L == H:
                    print("L==H")
                    continue
                
                eta = dataMatrix[i, :] * dataMatrix[i, :].T + dataMatrix[j, :] * dataMatrix[j, :].T - 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T # 7.106

                print('eta:{0}'.format(eta))

                # if eta >= 0, continue
                if eta < 0:
                    print("eta>=0")
                    continue
                
                # update alhpas[j]
                alphas[j] += labelMat[j] * (Ei - Ej) / eta # 7.106
                alphas[j] = clipAlpha(alphas[j], H, L)

                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                
                # update i by the same amount as j
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j]) # 7.109
                
                # the update is in the oppostie direction
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T \
                - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T # 7.115
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T \
                - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T # 7.116
                
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        
        print("iteration number: %d" % iter)
        
    return b, alphas[alphas > 0]
