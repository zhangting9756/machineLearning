# -*- coding:UTF-8 -*-

from sklearn.linear_model import LogisticRegression
import numpy as np
import random
import matplotlib.pyplot as plt

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0
    
def colicSklearn():
    trainingSet,trainingLabels = loadDataSet()
    classifier = LogisticRegression(penalty='l1',solver = 'liblinear',max_iter = 4000).fit(trainingSet, trainingLabels)      #最优参数
    test_accurcy = classifier.score(trainingSet, trainingLabels) * 100
    #print('正确率:%f%%' % test_accurcy)
    
def randomErrorRate():
    trainingSet,trainingLabels=loadDataSet()
    classifier = LogisticRegression(penalty='l1',solver = 'liblinear',max_iter = 5000).fit(trainingSet, trainingLabels) 
    numVec = 0.0
    errorCount = 0
    errorRate = 0.0
    for j in range(len(trainingSet)):
        numVec += 1.0
        trainingArry = [1, trainingSet[j][0], trainingSet[j][1]]
        weights = [classifier.intercept_, classifier.coef_[0][0],classifier.coef_[0][1]]
        if int(classifyVector(np.array(trainingArry), np.array(weights)) != int(trainingLabels[j])):
            errorCount += 1
            print("taningSet is:%r"%trainingSet[j])
            print("prediction label is:%r"%classifyVector(np.array(trainingArry), np.array(weights)))
            print("true label is:%r"%trainingLabels[j])
    errorRate = (float(errorCount)/numVec)*100                             #错误率计算
    print("numVec is:%d"%numVec)
    print("errorCount is:%r"%errorCount)
    print("测试集错误率为: %.2f%%" % errorRate)
 
def loadDataSet():
    dataMat = []                                                        #创建数据列表
    labelMat = []                                                    #创建标签列表
    fr = open(r'F:\Machine-Learning-master\Machine-Learning-master\Logistic\ex2data1.txt')                                            #打开文件    
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
    
def plotBestFit():
    dataMat, labelMat = loadDataSet()
    classifier = LogisticRegression(penalty='l1',solver = 'liblinear',max_iter = 5000).fit(dataMat, labelMat)                                    #加载数据集
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
    x = np.arange(20, 100, 0.1)
    y = (-classifier.intercept_[0] - classifier.coef_[0][0] * x) / classifier.coef_[0][1]
    ax.plot(x, y)
    plt.title('BestFit')                                                #绘制title
    plt.xlabel('X1'); plt.ylabel('X2')                                    #绘制label
    plt.show()        
   
if __name__ == '__main__':
    colicSklearn()
    randomErrorRate()
    plotDataSet()
    plotBestFit()