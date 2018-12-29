# -*-coding:utf-8 -*-
from numpy import *
import numpy as np
import AdaBoost

if __name__ == '__main__':
    dataArr, LabelArr = AdaBoost.loadDataSet(r'F:\Machine-Learning-master\Machine-Learning-master\AdaBoost\horseColicTraining2.txt')
    weakClassArr, aggClassEst = AdaBoost.adaBoostTrainDS(dataArr, LabelArr)
    testArr, testLabelArr = AdaBoost.loadDataSet(r'F:\Machine-Learning-master\Machine-Learning-master\AdaBoost\horseColicTest2.txt')
    #print(weakClassArr)
    predictions = AdaBoost.adaClassify(dataArr, weakClassArr)
    errArr = np.mat(np.ones((len(dataArr), 1)))
    print('训练集的错误率:%.3f%%' % float(errArr[predictions != np.mat(LabelArr).T].sum() / len(dataArr) * 100))
    predictions = AdaBoost.adaClassify(testArr, weakClassArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率:%.3f%%' % float(errArr[predictions != np.mat(testLabelArr).T].sum() / len(testArr) * 100))