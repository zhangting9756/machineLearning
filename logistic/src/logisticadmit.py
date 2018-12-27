# -*- coding:UTF-8 -*-
from sklearn.linear_model import LogisticRegression
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)                                                #返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)                                                       #参数初始化                                        #存储每次更新的回归系数
    for j in range(numIter):                                            
        dataIndex = list(range(m))
        for i in range(m):            
            alpha = 4/(1.0+j+i)+0.01                                            #降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0,len(dataIndex)))                #随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex]*weights))                    #选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h                                 #计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]       #更新回归系数
            del(dataIndex[randIndex])                                         #删除已经使用的样本
    return weights                                                             #返回

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)                                        #转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()                            #转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)                                            #返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.01                                                        #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500                                                        #最大迭代次数
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)                                #梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights                                                #将矩阵转换为数组，并返回

def colicTest():
    frTrain = open(r'F:\Machine-Learning-master\Machine-Learning-master\Logistic\horseColicTraining.txt')                                        #打开训练集
    frTest = open(r'F:\Machine-Learning-master\Machine-Learning-master\Logistic\horseColicTest.txt')                                                #打开测试集
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels,500)        #使用改进的随即上升梯度训练
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[-1]):
            errorCount += 1
    errorRatezt = (float(errorCount)/numTestVec)                                  #错误率计算
    print("测试集错误率为: %.2f%%" %errorRatezt)
    
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0
    
def colicSklearn():
    frTrain = open(r'F:\Machine-Learning-master\Machine-Learning-master\Logistic\ex2data1.txt')                                        #打开训练集
    frTest = open(r'F:\Machine-Learning-master\Machine-Learning-master\Logistic\ex2data1.txt')                                                #打开测试集
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
    classifier = LogisticRegression(penalty='l1',solver = 'liblinear',max_iter = 6000).fit(trainingSet, trainingLabels)
    test_accurcy = classifier.score(testSet, testLabels) * 100
    print('正确率:%f%%' % test_accurcy)
   
def randomErrorRate():
    frTest = open(r'F:\Machine-Learning-master\Machine-Learning-master\Logistic\ex2data1.txt')
    frTrain = open(r'F:\Machine-Learning-master\Machine-Learning-master\Logistic\ex2data1.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split(',')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    weights = gradAscent(trainingSet,trainingLabels)
    numVec = 0.0
    errorCount = 0
    errorRate = 0.0
    for j in range(len(trainingSet)):
        numVec += 1.0
        if int(classifyVector(trainingSet[j], weights) != int(trainingLabels[j])):
            errorCount += 1
            print("taningSet is:%r"%trainingSet[j])
            print("prediction label is:%r"%classifyVector(trainingSet[j], weights))
            print("true label is:%r"%trainingLabels[j])
    errorRate = (float(errorCount)/numVec)*100                             #错误率计算
    
    print("测试集错误率为: %.2f%%" % errorRate)
    print("numVec is:%d"%numVec)
    print("errorCount is:%r"%errorCount)
 
def loadDataSet():
    dataMat = []                                                        #创建数据列表
    labelMat = []                                                        #创建标签列表
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
    classifier = LogisticRegression(solver = 'sag',max_iter = 5000).fit(dataMat, labelMat)                                    #加载数据集
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
    
