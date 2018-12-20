# -*- coding:UTF-8 -*-
from sklearn.linear_model import LogisticRegression
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import svm

def loadDataSet():
    dataMat = []                                                        #创建数据列表
    labelMat = []                                                    #创建标签列表
    fr = open(r'F:\Machine-Learning-master\Machine-Learning-master\SVM\ex2data2.txt')                                            #打开文件    
    for line in fr.readlines():                                          #逐行读取
        lineArr = line.strip().split(',')                                #去回车，放入列表
        dataMat.append([float(lineArr[0]), float(lineArr[1])])        #添加数据
        labelMat.append(int(lineArr[2]))                                #添加标签
    fr.close()                                                            #关闭文件
    return dataMat, labelMat                                            #返回

def plotDataSet():
    dataMat, labelMat = loadDataSet()                                    #加载数据集
    dataArr = np.array(dataMat)                                            #转换成numpy的array数组
    n = np.shape(dataMat)[0]                                            #数据个数
    xcord1 = []; ycord1 = []                                            #正样本
    xcord2 = []; ycord2 = []                                            #负样本
    for i in range(n):                                                    #根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,0]); ycord1.append(dataArr[i,1])    #1为正样本
        else:
            xcord2.append(dataArr[i,0]); ycord2.append(dataArr[i,1])    #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                            #添加subplot
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)            #绘制负样本
    plt.title('DataSet')                                                #绘制title
    plt.xlabel('X1'); plt.ylabel('X2')                                    #绘制label
    plt.show()                                                            #显示
    
def calculateW(dataMat,labelMat):
    dataMat1=np.array(dataMat)
    labelMat1 = np.array(labelMat)
    clf = svm.SVC(C=0.8, kernel='linear', gamma=20, decision_function_shape='ovr')
    clf.fit(dataMat1,labelMat1)
    print(dataMat)
    
if __name__ == '__main__':
    plotDataSet()
    dataMat,labelMat = loadDataSet()
    calculateW(dataMat,labelMat)
    print(dataMat)
    