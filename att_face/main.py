# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os, glob, random


def pca(data, k):
    data = np.float32(np.mat(data))
    rows, cols = data.shape  # 取大小
    data_mean = np.mean(data, 0)  # 求均值
    Z = data - np.tile(data_mean, (rows, 1))
    D, V = np.linalg.eig(Z * Z.T)  # 特征值与特征向量
    V1 = V[:, :k]  # 取前k个特征向量
    V1 = Z.T * V1
    for i in xrange(k):  # 特征向量归一化
        V1[:, i] /= np.linalg.norm(V1[:, i])
    return np.array(Z * V1), data_mean, V1


def loadImageSet(folder=u'D:\\Download\\att_faces', sampleCount=5):  # 加载图像集，随机选择sampleCount张图片用于训练
    trainData = []
    testData = []
    yTrain = []
    yTest = []
    for k in range(40):
        folder2 = os.path.join(folder, 's%d' % (k + 1))
        data = [cv2.imread(d.encode('gbk'), 0) for d in glob.glob(os.path.join(folder2, '*.pgm'))]
        # if k == 0:
        #     print(data)
        # sample = random.sample(range(10), sampleCount)
        sample = range(5)
        trainData.extend([data[i].ravel() for i in range(10) if i in sample])
        testData.extend([data[i].ravel() for i in range(10) if i not in sample])
        yTest.extend([k] * (10 - sampleCount))
        yTrain.extend([k] * sampleCount)
    return np.array(trainData), np.array(yTrain), np.array(testData), np.array(yTest)


def main():
    for vertorNum in range(1,201):
        xTrain_, yTrain, xTest_, yTest = loadImageSet()
        num_train, num_test = xTrain_.shape[0], xTest_.shape[0]

        xTrain_, yTrain, xTest_, yTest = loadImageSet()
        num_train, num_test = xTrain_.shape[0], xTest_.shape[0]

        xTrain, data_mean, V = pca(xTrain_, vertorNum)
        xTest = np.array((xTest_ - np.tile(data_mean, (num_test, 1))) * V)  # 得到测试脸在特征向量下的数据

        # yPredict = [yTrain[np.sum((xTrain - np.tile(d, (num_train, 1))) ** 2, 1).argmin()] for d in xTest]
        # print u'欧式距离法识别率: %.2f%%' % ((yPredict == yTest).mean() * 100)

        svm = cv2.SVM()
        svm.train(np.float32(xTrain), np.float32(yTrain), params={'kernel_type': cv2.SVM_LINEAR})
        yPredict = [svm.predict(d) for d in np.float32(xTest)]
        # yPredict = svm.predict_all(xTest.astype(np.float64))
        # average = 0
        # for index in range(1,11):
        #     average += ((yPredict == yTest).mean() * 100)
        #     if(index == 10 ):
        #         print '%d' % (vertorNum) + u' ： 支持向量机识别率: %.2f%% \n' %  (average/10)
        #         # print '%d' % (vertorNum) + u' %.2f%% \n' % (average / 10)
        print "选择的特征值的个数：%d" %vertorNum
        print "准确度：%.2f%%" %((yPredict == yTest).mean() * 100)
        print "测试的具体情况"
        for index, item in enumerate(yPredict):
            print index/5, item, index/5==item
        print '\n'



if __name__ == '__main__':
    main()
