# -*- coding:UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import random

def loadDataSet():
    dataMat = []                                                        #创建数据列表
    labelMat = []                                                        #创建标签列表
    fr = open(r'F:\Machine-Learning-master\Machine-Learning-master\Logistic\ex2data1.txt')                                            #打开文件    
    for line in fr.readlines():                                            #逐行读取
        lineArr = line.strip().split(',')                                    #去回车，放入列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])        #添加数据
        labelMat.append(int(lineArr[2]))                                #添加标签
    fr.close()                                                            #关闭文件
    return dataMat, labelMat                                            #返回  连个list类型的

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)                                        #转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()                            #转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)                                            #返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.001                                                        #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500                                                        #最大迭代次数
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)                                #梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()                                                #将矩阵转换为数组，返回权重数组

def stocGradAscent1(dataMatrix, classLabels, numIter=150):                    #转换成numpy的mat
    m,n = np.shape(dataMatrix)                                                #返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)                                                    #参数初始化                                        #存储每次更新的回归系数
    for j in range(numIter):                                            
        dataIndex = list(range(m))
        for i in range(m):            
            alpha = 4/(1.0+j+i)+0.01                                            #降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0,len(dataIndex)))                   #随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex]*weights))                    #选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h                                 #计算误差
            weights = weights + alpha * error.T * np.array(dataMatrix[randIndex])       #更新回归系数
            del(dataIndex[randIndex])                                         #删除已经使用的样本
    return weights                                                             #返回


def plotDataSet():
    dataMat, labelMat = loadDataSet()                                    #加载数据集
    dataArr = np.array(dataMat)                                            #转换成numpy的array数组
    n = np.shape(dataMat)[0]                                            #数据个数
    xcord1 = []; ycord1 = []                                            #正样本
    xcord2 = []; ycord2 = []                                            #负样本
    for i in range(n):                                                    #根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])    #1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])    #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                            #添加subplot
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)            #绘制负样本
    plt.title('DataSet')                                                #绘制title
    plt.xlabel('X1'); plt.ylabel('X2')                                    #绘制label
    plt.show()                                                            #显示
    
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()                                    #加载数据集
    dataArr = np.array(dataMat)                                            #转换成numpy的array数组
    n = np.shape(dataMat)[0]                                            #数据个数
    xcord1 = []; ycord1 = []                                            #正样本
    xcord2 = []; ycord2 = []                                            #负样本
    for i in range(n):                                                    #根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])    #1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])    #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                            #添加subplot
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)            #绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')                                                #绘制title
    plt.xlabel('X1'); plt.ylabel('X2')                                    #绘制label
    plt.show()        
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0
   
def caculateErrorRate(dataMat,labelMat,weights):
    numVec = 0.0
    errorCount = 0
    errorRate = 0.0
    for j in range(len(dataMat)):
        numVec += 1.0
        if int(classifyVector(dataMat[j], weights) != int(labelMat[j])):
            errorCount += 1
            '''
            print("taningSet is:%r"%dataMat[j])
            print("prediction label is:%r"%classifyVector(dataMat[j], weights))
            print("true label is:%r"%labelMat[j])
            '''
    errorRate = (float(errorCount)/numVec)*100                             #错误率计算
    return errorRate
if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()    
    weights = gradAscent(dataMat, labelMat)
    plotDataSet()
    weights1 = np.mat(weights)
    errorRate1 = caculateErrorRate(dataMat,labelMat,weights1)
    print("total sample gradAscent errorRate is:%r"%errorRate1)
    weights2= stocGradAscent1(dataMat, labelMat, numIter=150)
    errorRate2 = caculateErrorRate(dataMat,labelMat,weights2)
    print("total sample stocGradAscent1 errorRate is:%r"%errorRate2)
    plotBestFit(weights)