#-*- coding:utf-8 -*-
import numpy as np
import regTree


if __name__ == '__main__':
    train_filename = 'F:\Machine-Learning-master\Machine-Learning-master\Regression Trees\exp2.txt'
    train_Data = regTree.loadDataSet(train_filename)
    train_Mat = np.mat(train_Data)
    tree = regTree.createTree(train_Mat, regTree.modelLeaf, regTree.modelErr)
    print('模型树:')
    print(tree)