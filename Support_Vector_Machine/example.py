# coding:UTF-8

import numpy as np
import algorithm as svm

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from algorithm import *

os.chdir(os.path.dirname(os.path.realpath(__file__)))


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


dataArr, labelArr = loadDataSet('testSet.txt')

b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
