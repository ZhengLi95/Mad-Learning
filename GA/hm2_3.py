# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        self.name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        self.M = 1  # 初始化M（目标维数）
        self.maxormins = [1] * self.M  # 初始化maxormins（目标最小最大化标记列表）
        self.Dim = 3  # 初始化Dim（决策变量维数）
        self.varTypes = np.array([0] * self.Dim)  # 初始化varTypes（决策变量的类型）
        lb = [0] * self.Dim  # 决策变量下界
        ub = [100] * self.Dim  # 决策变量上界
        self.ranges = np.array([lb, ub])  # 初始化ranges（决策变量范围矩阵）
        lbin = [1] * self.Dim
        ubin = [1] * self.Dim
        self.borders = np.array([lbin, ubin])  # 初始化borders（决策变量范围边界矩阵）

    def aimFuc(self, Vars, CV):
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        x3 = Vars[:, [2]]
        f = 1050 - x1 ** 2 - 2 * (x2 ** 2) - x3 ** 2 - x1 * x2 - x1 * x3
        # 利用可行性法则处理约束条件
        CV = np.hstack([x1 ** 2 + x2 ** 2 + x3 ** 2 - 25,
                        # -(x1 ** 2 + x2 ** 2 + x3 ** 2 - 25),
                        # -(9*x1+13*x2+7*x3-63),
                        9 * x1 + 13 * x2 + 7 * x3 - 63])  # 等式约束使用np.abs, 即绝对值小于等于0, 故等于0
        return f, CV

    def calBest(self):
        realBestObjV = None
        return realBestObjV
