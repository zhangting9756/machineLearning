# -*-coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import AdaBoost

def plotROC(predStrengths, classLabels):
    """
    绘制ROC
    Parameters:
        predStrengths - 分类器的预测强度
        classLabels - 类别
    Returns:
        无
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    cur = (1.0, 1.0)                                                         #绘制光标的位置
    ySum = 0.0                                                                 #用于计算AUC
    numPosClas = np.sum(np.array(classLabels) == 1.0)                        #统计正类的数量
    yStep = 1 / float(numPosClas)                                             #y轴步长    
    xStep = 1 / float(len(classLabels) - numPosClas)                         #x轴步长

    sortedIndicies = predStrengths.argsort()                                 #预测强度排序,从低到高
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep
        else:
            delX = xStep; delY = 0
            ySum += cur[1]                                                     #高度累加
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c = 'r')     #绘制ROC
        cur = (cur[0] - delX, cur[1] - delY)                                 #更新绘制光标的位置
    ax.plot([0,1], [0,1], 'b--')
    plt.title('AdaBoost ROC', FontProperties = font)
    plt.xlabel('false sun rate', FontProperties = font)
    plt.ylabel('ture sun rate', FontProperties = font)
    ax.axis([0, 1, 0, 1])
    print('AUC :', ySum * xStep)                                         #计算AUC
    plt.show()
    
if __name__ == '__main__':
    dataArr, LabelArr = AdaBoost.loadDataSet(r'F:\Machine-Learning-master\Machine-Learning-master\AdaBoost\horseColicTraining2.txt')
    weakClassArr, aggClassEst = AdaBoost.adaBoostTrainDS(dataArr, LabelArr, 50)
    plotROC(aggClassEst.T, LabelArr)