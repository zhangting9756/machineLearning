# -*- coding:UTF-8 -*-

from sklearn.linear_model import LogisticRegression
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0
    
def colicSklearn():
    frTrain = open(r'F:\Machine-Learning-master\Machine-Learning-master\Logistic\ex2data1.txt')                                        #打开训练集
    frTest = open(r'F:\Machine-Learning-master\Machine-Learning-master\Logistic\ex2data1.txt')                                         #打开测试集
    trainingSet = []; trainingLabels = []
    testSet = []; testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split(',')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    for line in frTest.readlines():
        currLine = line.strip().split(',')
        lineArr =[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    classifier = LogisticRegression(penalty='l1',solver = 'liblinear',max_iter = 4000).fit(trainingSet, trainingLabels)
    test_accurcy = classifier.score(testSet, testLabels) * 100
    #print('正确率:%f%%' % test_accurcy)
    
def randomErrorRate():
    dataTrainingMat,dataTestMat,dataTrainingLabels,dataTestLabels=loadDataSet()
    classifier = LogisticRegression(penalty='l1',solver = 'liblinear',max_iter = 5000).fit(dataTrainingMat, dataTrainingLabels) 
    numVec = 0.0
    errorCount = 0
    errorRate = 0.0
    for j in range(len(dataTrainingMat)):
        numVec += 1.0
        trainingArry = [1, dataTrainingMat[j][0], dataTrainingMat[j][1]]
        weights = [classifier.intercept_, classifier.coef_[0][0],classifier.coef_[0][1]]
        if int(classifyVector(np.array(trainingArry), np.array(weights)) != int(dataTrainingLabels[j])):
            errorCount += 1
            print("taningSet is:%r"%dataTrainingMat[j])
            print("prediction label is:%r"%classifyVector(np.array(trainingArry), np.array(weights)))
            print("true label is:%r"%dataTrainingLabels[j])
    errorRate = (float(errorCount)/numVec)*100                             #错误率计算
    print("numVec is:%d"%numVec)
    print("errorCount is:%r"%errorCount)
    print("训练集错误率为: %.2f%%" % errorRate)
    numVec1 = 0
    for j in range(len(dataTestMat)):
        numVec1 += 1.0
        trainingArry = [1, dataTestMat[j][0], dataTestMat[j][1]]
        weights = [classifier.intercept_, classifier.coef_[0][0],classifier.coef_[0][1]]
        if int(classifyVector(np.array(trainingArry), np.array(weights)) != int(dataTestLabels[j])):
            errorCount += 1
            print("taningSet is:%r"%dataTestMat[j])
            print("prediction label is:%r"%classifyVector(np.array(trainingArry), np.array(weights)))
            print("true label is:%r"%dataTestLabels[j])
    errorRate = (float(errorCount)/numVec1)*100                             #错误率计算
    print("numVec is:%d"%numVec1)
    print("errorCount is:%r"%errorCount)
    print("测试集错误率为: %.2f%%" % errorRate)
 
def loadDataSet():
    dataMat = []                                                        #创建数据列表
    labelMat = []                                                    #创建标签列表
    dataTrainingMat=[]
    dataTestMat=[]
    dataTrainingLabels=[]
    dataTestLabels=[]
    fr = open(r'F:\Machine-Learning-master\Machine-Learning-master\Logistic\ex2data1.txt')                                            #打开文件    
    for line in fr.readlines():                                          #逐行读取
        lineArr = line.strip().split(',')                                #去回车，放入列表
        dataMat.append([float(lineArr[0]), float(lineArr[1])])        #添加数据
        labelMat.append(int(lineArr[2]))                                #添加标签
    fr.close()                                                            #关闭文件
    dataTrainingMat,dataTestMat,dataTrainingLabels,dataTestLabels = train_test_split(dataMat,labelMat,test_size=0.2,random_state=None)
    return dataTrainingMat,dataTestMat,dataTrainingLabels,dataTestLabels                                            #返回
   
def plotDataSet():
    dataTrainingMat,dataTestMat,dataTrainingLabels,dataTestLabels = loadDataSet()                                    #加载数据集
    dataArr = np.array(dataTrainingMat)                                            #转换成numpy的array数组
    n = np.shape(dataTrainingLabels)[0]                                            #数据个数
    xcord1 = []; ycord1 = []                                            #正样本
    xcord2 = []; ycord2 = []                                            #负样本
    for i in range(n):                                                    #根据数据集标签进行分类
        if int(dataTrainingLabels[i]) == 1:
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
    dataTrainingMat,dataTestMat,dataTrainingLabels,dataTestLabels = loadDataSet()
    classifier = LogisticRegression(solver = 'sag',max_iter = 5000).fit(dataTrainingMat, dataTrainingLabels)                                    #加载数据集
    dataArr = np.array(dataTrainingMat)                                            #转换成numpy的array数组
    n = np.shape(dataTrainingMat)[0]                                            #数据个数
    xcord1 = []; ycord1 = []                                            #正样本
    xcord2 = []; ycord2 = []                                            #负样本
    for i in range(n):                                                    #根据数据集标签进行分类
        if int(dataTrainingLabels[i]) == 1:
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