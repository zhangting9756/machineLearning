# -*-coding:utf-8 -*-
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import AdaBoost

if __name__ == '__main__':
    dataArr, classLabels = AdaBoost.loadDataSet(r'F:\Machine-Learning-master\Machine-Learning-master\AdaBoost\horseColicTraining2.txt')
    testArr, testLabelArr = AdaBoost.loadDataSet(r'F:\Machine-Learning-master\Machine-Learning-master\AdaBoost\horseColicTest2.txt')
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2), algorithm = "SAMME", n_estimators = 10)
    bdt.fit(dataArr, classLabels)
    predictions = bdt.predict(dataArr)
    errArr = np.mat(np.ones((len(dataArr), 1)))
    print('训练集的错误率:%.3f%%' % float(errArr[predictions != classLabels].sum() / len(dataArr) * 100))
    predictions = bdt.predict(testArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率:%.3f%%' % float(errArr[predictions != testLabelArr].sum() / len(testArr) * 100))