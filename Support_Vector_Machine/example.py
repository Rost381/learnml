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

def load_data_libsvm(data_file):
    '''导入训练数据
    input:  data_file(string):训练数据所在文件
    output: data(mat):训练样本的特征
            label(mat):训练样本的标签
    '''
    data = []
    label = []
    f = open(data_file)
    for line in f.readlines():
        lines = line.strip().split(' ')
        
        # 提取得出label
        label.append(float(lines[0]))
        # 提取出特征，并将其放入到矩阵中
        index = 0
        tmp = []
        for i in range(1, len(lines)):
            li = lines[i].strip().split(":")
            if int(li[0]) - 1 == index:
                tmp.append(float(li[1]))
            else:
                while(int(li[0]) - 1 > index):
                    tmp.append(0)
                    index += 1
                tmp.append(float(li[1]))
            index += 1
        while len(tmp) < 13:
            tmp.append(0)
        data.append(tmp)
    f.close()
    return np.mat(data), np.mat(label).T

if __name__ == "__main__":
    # 1、导入训练数据
    print("------------ 1load data --------------")
    dataSet, labels = load_data_libsvm("heart_scale")
    # 2、训练SVM模型
    print("------------ 2training ---------------")
    C = 0.6
    toler = 0.001
    maxIter = 500
    svm_model = svm.SVM_training(dataSet, labels, C, toler, maxIter)
    # 3、计算训练的准确性
    print("------------ 3cal accuracy --------------")
    accuracy = svm.cal_accuracy(svm_model, dataSet, labels)  
    print("The training accuracy is: %.3f%%" % (accuracy * 100))
    # 4、保存最终的SVM模型
    print("------------ 4save model ----------------")
    svm.save_svm_model(svm_model, "model_file")