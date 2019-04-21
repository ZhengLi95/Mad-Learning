# -*- coding: utf-8 -*-
"""
Pycharm

by Junjia Liu

"""

from numpy import *
import pandas as pd

def loadDataSet(filename): #读取数据

    f=pd.read_csv(filename)
    # train=f[ :630, :]
    # test=f[631: , :]
    labelMat=f['Survived']
    labelMat.replace(0, -1, inplace=True)
    dataMat=f.drop('Survived',axis=1)
    return dataMat,labelMat #返回数据特征和数据类别

def selectJrand(i,m): #在0-m中随机选择一个不是i的整数
    j=i
    while (j==i):
        j=int(random.uniform(0,m))
    return j

#最优解剪辑，《统计学习方法》p127 7.108
def clipAlpha(aj,H,L):  #保证a在L和H范围内（L <= a <= H）
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

def kernelTrans(X, A, kTup): #核函数，输入参数,X:支持向量的特征树；A：某一行特征数据；kTup：('lin',k1)核函数的类型和参数
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': #线性函数
        K = X * A.T
    elif kTup[0]=='rbf': # 径向基函数(radial bias function)
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #返回生成的结果
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


#定义类，方便存储数据
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):  # 存储各类参数
        self.X = dataMatIn  #数据特征
        self.labelMat = classLabels #数据类别
        self.C = C #软间隔参数C，参数越大，非线性拟合能力越强
        self.tol = toler #停止阀值
        self.m = shape(dataMatIn)[0] #数据行数
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0 #初始设为0
        self.eCache = mat(zeros((self.m,2))) #缓存
        self.K = mat(zeros((self.m,self.m))) #核函数的计算结果
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)


def calcEk(oS, k): #计算Ek（参考《统计学习方法》p127公式7.105）
    gXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b) #7.104
    # print(float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k]))
    Ek = gXk - float(oS.labelMat[k])
    return Ek

#随机选取aj，并返回其E值
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]  #返回矩阵中的非零位置的行数
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE): #返回步长最大的aj
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k): #更新os数据
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

#首先检验ai是否满足KKT条件，如果不满足，随机选择aj进行优化，更新ai,aj,b值
def innerL(i, oS): #输入参数i和所有参数数据
    Ei = calcEk(oS, i) #计算E值
    # print(float(oS.labelMat[i]*Ei)+oS.alphas[i])
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)): #检验这行数据是否符合KKT条件 参考《统计学习方法》p128公式7.111-113
        j,Ej = selectJ(i, oS, Ei) #随机选取aj，并返回其E值
        alphaIold = oS.alphas[i].copy() 
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]): #以下代码的公式参考《统计学习方法》p126下侧，L、H取值计算
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            # print("Error: L==H")
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]   #参考《统计学习方法》p127公式7.107
        if eta >= 0:
            # print("Error: eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta #参考《统计学习方法》p127公式7.106，未经剪辑的情况下沿着约束方向更新alpha[j]
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L) #参考《统计学习方法》p127公式7.108，对新alpha[j]进行剪辑
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < oS.tol): #alpha变化大小阀值（自己设定）
            # print("Error: j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#参考《统计学习方法》p127公式7.109， 由alpha[j]求得alpha[i]
        updateEk(oS, i) #更新数据
        #以下求解b的过程，参考《统计学习方法》p129公式7.115-7.116
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]<oS.C):
            oS.b = b1
        elif (0 < oS.alphas[j]<oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        # print("Error:")
        return 0


#SMO函数，用于快速求解出alpha
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)): #输入参数：数据特征，数据类别，参数C，阀值toler，最大迭代次数，核函数（默认线性核）
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m): #遍历所有数据
                alphaPairsChanged += innerL(i,oS)
                #print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)) #显示第多少次迭代，那行特征数据使alpha发生了改变，这次改变了多少次alpha
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs: #遍历非边界的数据
                alphaPairsChanged += innerL(i,oS)
                # print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas

def testRbf(data_train, data_predict):
    dataArr,labelArr = loadDataSet(data_train) #读取训练数据

    #因为题目中提供的test数据是没有标签的，所以测试集就按照3:7的比例从训练数据集中分出
    dataArr_train=dataArr.values[ :630, :]
    labelArr_train=labelArr.values[ :630]
    dataArr_test=dataArr.values[631: , :]
    labelArr_test=labelArr.values[631: ]
    
    b,alphas = smoP(dataArr_train, labelArr_train, 20, 0.0001, 12, ('rbf', 0.2)) #通过SMO算法得到b和alpha
    datMat=mat(dataArr_train)
    labelMat = mat(labelArr_train).transpose()
    svInd=nonzero(alphas)[0]  #选取不为0数据的行数（也就是支持向量）
    sVs=datMat[svInd] #支持向量的特征数据
    labelSV = labelMat[svInd] #支持向量的类别（1或-1）
    print("there are %d Support Vectors" % shape(sVs)[0]) #打印出共有多少的支持向量
    m,n = shape(datMat) #训练数据的行列数
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:], ('rbf', 0.2)) #将支持向量转化为核函数
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b  #这一行的预测结果（代码来源于《统计学习方法》p133里面最后用于预测的公式）注意最后确定的分离平面只有那些支持向量决定。
        # print(sign(predict), sign(labelArr_train[i]))
        if sign(predict) != sign(labelArr_train[i]): #sign函数 -1 if x < 0, 0 if x==0, 1 if x > 0
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m)) #打印出错误率

    
    errorCount_test = 0
    datMat_test=mat(dataArr_test)
    labelMat_test = mat(labelArr_test).transpose()
    m,n = shape(datMat_test)
    for i in range(m): #在测试数据上检验错误率
        kernelEval = kernelTrans(sVs,datMat_test[i,:], ('rbf', 0.2))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labelArr_test[i]):
            errorCount_test += 1
    print("the test error rate is: %f" % (float(errorCount_test)/m))

    #Prediction
    datMat_predict=mat(data_predict)
    m,n=shape(datMat_predict)
    predictions=zeros(m)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat_predict[i, :], ('rbf', 0.2))
        predictions[i] = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
    avg_predict=average(predictions)
    for i in range(m):
        if predictions[i] >= avg_predict:
            predictions[i] = 1
        else:
            predictions[i] = 0
    predictions.astype('int')
    test_rare = pd.read_csv('test.csv')
    submission = pd.DataFrame({'PassengerId': test_rare['PassengerId'],
                               'Survived': predictions}).astype('int')
    submission.to_csv("myresult.csv", index=False)

#主程序, 直接应用IPython程序处理过的数据集train_new和test_new
def main():
    filename_traindata='train_new.csv'
    filename_predictdata=pd.read_csv('test_new.csv')
    testRbf(filename_traindata, filename_predictdata)

if __name__=='__main__':
    main()
