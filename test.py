#coding:utf-8
#朴素贝叶斯算法   贝叶斯估计， λ=1  K=2， S=3； λ=1 拉普拉斯平滑
import pandas as pd
import numpy as np

class NavieBayesB(object):
    def __init__(self):
        self.A = 1    # 即λ=1
        self.K = 2
        self.S = 3

    def getTrainSet(self):
        trainSet = pd.read_csv("training sample.csv", sep=",")
        trainSetNP = np.array(trainSet)     #由dataframe类型转换为数组类型
        trainData = trainSetNP[:,0:trainSetNP.shape[1]-1]     #训练数据x1,x2
        labels = trainSetNP[:,trainSetNP.shape[1]-1]          #训练数据所对应的所属类型Y
        return trainData, labels

    def classify(self, trainData, labels, features):
        labels = list(labels)    #转换为list类型
        #求先验概率
        P_y = {}
        for label in labels:
            P_y[label] = (labels.count(label) + self.A) / float(len(labels) + self.K*self.A)

        #求条件概率
        P = {}
        for y in P_y.keys():
            y_index = [i for i, label in enumerate(labels) if label == y]   # y在labels中的所有下标
            y_count = labels.count(y)     # y在labels中出现的次数
            for j in range(len(features)):
                pkey = str(features[j]) + '|' + str(y)
                x_index = [i for i, x in enumerate(trainData[:,j]) if x == features[j]]   # x在trainData[:,j]中的所有下标
                xy_count = len(set(x_index) & set(y_index))   #x y同时出现的次数
                P[pkey] = (xy_count + self.A) / float(y_count + self.S*self.A)   #条件概率

        #features所属类
        F = {}
        for y in P_y.keys():
            F[y] = P_y[y]
            for x in features:
                F[y] = F[y] * P[str(x)+'|'+str(y)]

        features_y = max(F, key=F.get)   #概率最大值对应的类别
        return features_y


if __name__ == '__main__':
    nb = NavieBayesB()
    # 训练数据
    trainData, labels = nb.getTrainSet()
    # x1,x2
    features = [10]
    # 该特征应属于哪一类
    result = nb.classify(trainData, labels, features)
    print(features,'属于',result)