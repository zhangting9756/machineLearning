#-*-coding:utf-8 -*-

from numpy import *
from math import sqrt

def loadData(fileName):
    data = []
    fr = open(fileName)
    for line in fr.readlines():
        curline = line.strip().split('\t')
        frline = map(float,curline)
        data.append(frline)
    return data

"""
函数说明:计算两个向量的欧式距离
Parameters: vecA、vecB
Returns: 两个向量的欧式距离
"""
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

#初始化聚类中心
def randCent(dataSet,k):
    n = shape(dataSet)[1]
    center = mat(zeros((k,n)))
    for j in range(n):
        rangeJ = float(max(dataSet[:,j]) - min(dataSet[:,j]))
        center[:,j] = min(dataSet[:,j]) + rangeJ * random.rand(k,1)
    return center

def kMeans(dataSet,k,dist = distEclud,createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    center = createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = dist(dataSet[i,:],center[j,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:#判断是否收敛
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist ** 2
        #print center
        for cent in range(k):#更新聚类中心
            dataCent = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            center[cent,:] = mean(dataCent,axis = 0)#axis是普通的将每一列相加，而axis=1表示的是将向量的每一行进行相加
    return center,clusterAssment

"""
函数说明:二分K-均值聚类算法
Parameters:
    dataSet - 数据集
    k - 簇的数目，即质心得数量
    distEclud - 距离计算函数
Returns: 
    centroids - 质心
    clusterAssment - 第一列记录簇索引值 ，第二列记录储存误差
"""
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #create a list with one centroid,存放每个特征的均值
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2    #存放每个样本距均值的距离
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat             #类质心
                bestClustAss = splitClustAss.copy()    #每个样本属于哪个类
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment


if __name__=='__main__':
    dataMat = mat(loadData('testSet.txt'))
    myCentroids, clustAssing = biKmeans(dataMat, 4)
    myCentroids1, clustAssing1 = kMeans(dataMat, 4)
    print('K均值算法，质心：%r' % myCentroids1)
    print('K二分算法，质心：%r' % myCentroids)
    #print('K均值算法，索引以及储存误差： %r' % clustAssing)
    