# -*-coding:utf-8 -*-
from numpy import *
import AdaBoost

if __name__ == '__main__':
    dataArr,classLabels = AdaBoost.loadSimpData()
    weakClassArr, aggClassEst = AdaBoost.adaBoostTrainDS(dataArr, classLabels)
    #print("zt %r"%weakClassArr)
    AdaBoost.adaClassify([[5,5],[0,0]], weakClassArr)
    AdaBoost.showDataSet(dataArr,classLabels)