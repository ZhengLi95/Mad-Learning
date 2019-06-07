> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 https://zhuanlan.zhihu.com/p/53622102

**相关连接：**

[ECKai：多智能体强化学习入门（一）——基础知识与博弈​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/53474965)[ECKai：多智能体强化学习入门（二）——基础算法（MiniMax-Q，NashQ，FFQ，WoLF-PHC）​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/53563792)[ECKai：多智能体强化学习入门（三）——矩阵博弈中的分布式学习算法​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/53622102)[ECKai：多智能体强化学习入门（四）——MADDPG 算法​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/53811876)[ECKai：多智能体强化学习入门（五）——QMIX 算法分析​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/55003734)

附录：

[ECKai：附：强化学习——DRQN 分析详解​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/54898904)

一、引言

多智能体系统一直在学术界或者工业届都是一个热点。其核心领域是关于如何将系统采用分布式的算法控制。在分布式算法中，没有一个中心节点进行总体控制，每个智能体通过与环境交互自己学习自己的最优策略，在不知道全局信息的情况下将策略收敛到纳什均衡点。通常意义下，智能体只知道自身获得的奖励值，不知道环境中的奖励函数，不知道其他智能体的策略，不知道其他智能体获得的奖励。分布式算法主要的核心难点是如何在只使用不完整信息的情况下，使每个智能体学到的策略收敛到纳什均衡点。

上一篇文章提到了很多多智能体强化学习算法，在很多算法中都要使用线性规划或者二次规划来求解矩阵博弈的纳什均衡点，在本章中将给出 4 种学习算法来求解每个矩阵博弈的纳什均衡策略。

矩阵博弈的学习算法可以分为两类，其中一类利用**学习自动机**（learning automata）思想，另一类利用**梯度提升**（gradient ascent）思想。下面根据这两种分类独立介绍每种算法。

## 二、矩阵博弈的学习算法

矩阵博弈的学习算法是指每个智能体不断与环境交互，通过获得的奖励值来更新优化其策略，最终使策略能够收敛到博弈的纳什均衡点。矩阵博弈的学习算法可以分为**学习自动机**和**梯度提升**两类，下面从这两个角度开始介绍。

## 1\. 学习自动机

学习自动机是一种通过与环境交互获得奖励来修改动作空间中每个动作的概率分布，从而提升优化策略的方法。学习自动机是一种完全分布式的算法，每个智能体只用知道自己的策略与奖励值，不需要知道环境信息以及其他智能体的信息。

学习自动机通常可以用一个元组表示为 ![](https://www.zhihu.com/equation?tex=%28A%2Cr%2Cp%2CU%29) ，其中 ![](https://www.zhihu.com/equation?tex=A%3D%7Ba_1%2C%5Ccdots%2Ca_m%7D) 表示动作集； ![](https://www.zhihu.com/equation?tex=r) 为奖励值； ![](https://www.zhihu.com/equation?tex=p) 为动作集的一个概率分布，即一个要学习的策略； ![](https://www.zhihu.com/equation?tex=U) 表示一个学习算法。下面主要介绍两种算法 ![](https://www.zhihu.com/equation?tex=L_%7BR-I%7D) （linear reward-inaction）算法与 ![](https://www.zhihu.com/equation?tex=L_%7BR-P%7D) （linear reward-penalty）算法。

## a. ![](https://www.zhihu.com/equation?tex=L_%7BR-I%7D)

![](https://www.zhihu.com/equation?tex=L_%7BR-I%7D) 可用于 n 智能体矩阵博弈或双智能体零和博弈中，收敛性条件为：博弈只具有纯策略严格纳什均衡，只有满足收敛性条件，所有的智能体才能够收敛到纳什均衡策略。对于第 i 个智能体，其策略更新公式为

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+p_c%5Ei%28k%2B1%29+%26%3D+p_c%5Ei%28k%29+%2B+%5Ceta+r%5Ei%28k%29%281-p_c%5Ei%28k%29%29+%5Cqquad+a_c%5C+%5Crm+%7Bis%5C+current%5C+action%5C+at%5C+step%5C+k%7D+%5C%5C+++p_j%5Ei%28k%2B1%29+%26%3D+p_j%5Ei%28k%29-%5Ceta+r%5Ei%28k%29p_j%5Ei%28k%29%5Cqquad%5Cqquad+%5Crm%7Bfor%5C+all%5C+a_j%5Ei%5Cne+a_c%5Ei%7D+%5Cend%7Baligned%7D)

如上式所示，k 为时刻； ![](https://www.zhihu.com/equation?tex=p_c%5Ei) 是指智能体的动作分布，上标是指智能体的编号，下标是指智能体的动作编号； ![](https://www.zhihu.com/equation?tex=%5Ceta) 是学习速率； ![](https://www.zhihu.com/equation?tex=+r%5Ei%28k%29) 是指智能体 i 在时刻 k 执行动作 ![](https://www.zhihu.com/equation?tex=a_c) 获得的奖励值。整个学习算法流程如下：

![](https://pic3.zhimg.com/v2-0636867a9ecf1649ee5dd0266ee655be_b.jpg)![](https://pic3.zhimg.com/v2-0636867a9ecf1649ee5dd0266ee655be_r.jpg)

其意义就是将优秀的动作的选择概率提高，为了保证归一性，则将其其他动作被选择的概率。该算法只能在博弈只具有纯策略的纳什均衡点的时候可以收敛。

## b. ![](https://www.zhihu.com/equation?tex=L_%7BR-P%7D)

在 ![](https://www.zhihu.com/equation?tex=L_%7BR-I%7D) 的基础上加上一个罚函数，就变成了 ![](https://www.zhihu.com/equation?tex=L_%7BR-P%7D) 算法。 ![](https://www.zhihu.com/equation?tex=L_%7BR-P%7D) 算法的收敛条件为：博弈只有混合策略的纳什均衡。当满足收敛条件， ![](https://www.zhihu.com/equation?tex=L_%7BR-P%7D) 算法就能够收敛到最优解。其更新公式如下

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+p_c%5Ei%28k%2B1%29+%26%3D+p_c%5Ei%28k%29+%2B+%5Ceta_1+r%5Ei%28k%29%5B1-p_c%5Ei%28k%29%5D-%5Ceta_2%5B1-r%5Ei%28k%29%5Dp_c%5Ei%28k%29+%5Cqquad+a_c%5C+%5Crm+%7Bis%5C+current%5C+action%5C+at%5C+step%5C+k%7D+%5C%5C+++p_j%5Ei%28k%2B1%29+%26%3D+p_j%5Ei%28k%29-%5Ceta_1+r%5Ei%28k%29p_j%5Ei%28k%29%2B%5Ceta_2%5B1-r%5Ei%28k%29%5D%5B%5Cfrac%7B1%7D%7Bm-1%7D-p_j%5Ei%28k%29%5D%5Cqquad%5Cqquad+%5Crm%7Bfor%5C+all%5C+a_j%5Ei%5Cne+a_c%5Ei%7D+%5Cend%7Baligned%7D)

算法中的学习速率应该满足 ![](https://www.zhihu.com/equation?tex=0%3C%5Ceta_1%2C%5Ceta_2%3C1) 和 ![](https://www.zhihu.com/equation?tex=%5Ceta_2%3C%5Ceta_1) . 其算法流程为：

![](https://pic1.zhimg.com/v2-ec5233fa0088f91aa621ede60bcdb204_b.jpg)![](https://pic1.zhimg.com/v2-ec5233fa0088f91aa621ede60bcdb204_r.jpg)

该算法只能应用于只有混合策略纳什均衡的博弈中。

## 2\. 梯度提升学习算法

梯度提升学习算法在很多地方都有用到，强化学习的随机策略梯度算法 SPG，DPG，AC，A2C，A3C 等算法都有用到这种思想。使策略的更新方向沿着累积回报增加最大的梯度方向。但是有人证明，梯度提升学习算法并不能在所有的矩阵博弈中的到收敛解。但是在某些特定的约束博弈中，有一些梯度提升算法可以求解其纳什均衡。下面介绍几种常见的算法。

## a. WoLF-IGA

WoLF-IGA 但并不是一个实用的分布式算法，后面会分析原因。该算法使用上一节介绍的 WoLF（Win or learn fast）思想与梯度提升结合。获胜或优秀策略的含义是指当前策略的累积预期奖励大于当前玩家纳什均衡策略和其他玩家实用当前策略的累积预期奖励。当前策略获胜时则谨慎缓慢学习，给其他智能体适应策略变化的时间；当前策略较差时，快速更新调整，使其能够快速调整适应其他智能体策略变化。WoLF-IGA 只适用于**双智能体双动作**矩阵博弈。WoLF-IGA 根据累计奖励关于策略的梯度来修正策略，其目的是使更新后的策略能够获得更大的奖励值。

令 ![](https://www.zhihu.com/equation?tex=p_1) 表示智能体 1 选择第 1 个动作的概率， ![](https://www.zhihu.com/equation?tex=1-p_1) 表示智能体 1 选择第 2 个动作的概率；同理 ![](https://www.zhihu.com/equation?tex=q_1) 表示智能体 2 选择第 1 个动作的概率，同理 ![](https://www.zhihu.com/equation?tex=1-q_1) 表示智能体 2 选择第 2 个动作的概率，其算法的核心公式为

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+p_1%28k%2B1%29%3Dp_1%28k%29%2B%5Ceta+%5Calpha_1%28k%29%5Cfrac%7B%5Cpartial+V_1%28p_1%28k%29%2Cq_1%28k%29%29%7D%7B%5Cpartial+p_1%7D%5C%5C+q_1%28k%2B1%29%3Dq_1%28k%29%2B%5Ceta+%5Calpha_w%28k%29%5Cfrac%7B%5Cpartial+V_w%28p_1%28k%29%2Cq_1%28k%29%29%7D%7B%5Cpartial+q_1%7D%5C%5C+%5Calpha_1%3D%5Cleft%5C%7B+%5Cbegin%7Barray%7D%7Brcl%7D+%5Calpha_%7B%5Cmin%7D++++++%26++++++%26+%7B%5Crm%7Bif%7D%5C+V_1%28p_1%28k%29%2Cq_1%28k%29%29%3EV_1%28p%5E%2A_1%2Cq_1%28k%29%29%7D%5C%5C+%5Calpha_%7B%5Cmax%7D+++++%26++++++%26+%7B%5Crm+otherwise%7D%5C%5C+%5Cend%7Barray%7D+%5Cright.+%5C%5C+%5Calpha_2%3D%5Cleft%5C%7B+%5Cbegin%7Barray%7D%7Brcl%7D+%5Calpha_%7B%5Cmin%7D++++++%26++++++%26+%7B%5Crm%7Bif%7D%5C+V_2%28p_1%28k%29%2Cq_1%28k%29%29%3EV_1%28p_1%28k%29%2Cq_1%5E%2A%7D%5C%5C+%5Calpha_%7B%5Cmax%7D+++++%26++++++%26+%7B%5Crm+otherwise%7D%5C%5C+%5Cend%7Barray%7D+%5Cright.+%5Cend%7Baligned%7D)

![](https://www.zhihu.com/equation?tex=%5Ceta) 为学习步长，需要足够小。 ![](https://www.zhihu.com/equation?tex=%5Calpha_i) 体现 WoLF 思想，满足 ![](https://www.zhihu.com/equation?tex=%5Calpha_%7B%5Cmax%7D%3E%5Calpha_%7B%5Cmin%7D) ，是一个可变的学习速率，根据当前策略的效果来调整学习快慢。 ![](https://www.zhihu.com/equation?tex=V_i%28p_1%28k%29%2Cq_1%28k%29%29) 表示在时刻 k 使用策略 ![](https://www.zhihu.com/equation?tex=%28p_1%28k%29%2Cq_1%28k%29%29) 获得的累积期望回报。 ![](https://www.zhihu.com/equation?tex=%28p_1%5E%2A%28k%29%2Cq_1%5E%2A%28k%29%29) 表示纳什均衡策略。

WoLF-IGA 算法的难点在于需要已知大量信息。其信息包括自身的奖励矩阵、其他玩家的策略以及自己的纳什均衡。虽然智能体知道自己的奖励矩阵，也会得到纳什均衡策略。但这样大量的已知信息导致这个算法并不是一个实用的算法，也不是一个分布式的算法。

该算法的收敛性条件为：双智能体双行动一般和矩阵博弈，且纳什均衡为纯策略或混合策略。

## b. Lagging Anchor 算法

Lagging Anchor 算法是应用于**双智能体零和博弈**的算法。其并不是一个分布式算法，具体原因后面会分析。

我们定义 ![](https://www.zhihu.com/equation?tex=v%3D%5Bp_1%2Cp_2%2C%5Ccdots%2Cp_%7Bm_1%7D%5D%5ET) 是智能体 1 对于其 ![](https://www.zhihu.com/equation?tex=m_1) 个动作的概率分布，即策略；同理 ![](https://www.zhihu.com/equation?tex=w%3D%5Bq_1%2Cq_2%2C%5Ccdots%2Cq_%7Bm_2%7D%5D%5ET) 为智能体 2 的策略。其策略迭代公式为

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+v%28k%2B1%29%26%3Dv%28k%29%2B%5Ceta+P_%7Bm_1%7DR_1Y%28k%29%2B%5Ceta%5Ceta_d%28%5Coverline+v%28k%29-v%28k%29%29%5C%5C+%5Coverline+v%28k%2B1%29+%26%3D+%5Coverline+v%28k%29%2B%5Ceta%5Ceta_d%28v%28k%29-%5Coverline+v%28k%29%29%5C%5C+w%28k%2B1%29%26%3Dw%28k%29%2B%5Ceta+P_%7Bm_2%7DR_2X%28k%29%2B%5Ceta%5Ceta_d%28%5Coverline+w%28k%29-w%28k%29%29%5C%5C+%5Coverline+w%28k%2B1%29+%26%3D+%5Coverline+w%28k%29%2B%5Ceta%5Ceta_d%28w%28k%29-%5Coverline+w%28k%29%29%5C%5C+%5Cend%7Baligned%7D)

其中， ![](https://www.zhihu.com/equation?tex=%5Ceta) 为学习步长， ![](https://www.zhihu.com/equation?tex=%5Ceta_d%3E0) 定义为拉锚因子（anchor drawing factor）， ![](https://www.zhihu.com/equation?tex=P_%7Bm_i%7D%3DI_%7Bm_i%7D-%5Cfrac%7B1%7D%7Bm_i%7D%5Ctextbf%7B1%7D_%7Bm_i%7D%5Ctextbf%7B1%7D_%7Bm_i%7D%5ET) 是一个用来维持策略 ![](https://www.zhihu.com/equation?tex=v%2Cw) 归一化的矩阵。 ![](https://www.zhihu.com/equation?tex=Y%28k%29) 是一个单位向量，若智能体 2 在时刻 k 采用第 i 个动作则 ![](https://www.zhihu.com/equation?tex=Y%28k%29_i%3D1) 第 i 个元素为 1，其余元素为 0； ![](https://www.zhihu.com/equation?tex=X%28k%29) 同理。 ![](https://www.zhihu.com/equation?tex=%5Coverline+v%2C%5Coverline+w) 为锚参数，表示每个智能体的策略的加权平均，是 Lagging Anchor 算法的核心。

收敛条件：在双智能体零和博弈中，只有完全混合策略。对于纯策略的收敛情况目前还没有人证明。

由于这个算法需要用到每一个智能体的奖励矩阵 ![](https://www.zhihu.com/equation?tex=R_1%2CR_2) ，因此严格来说其不能算作是一个分布式算法，但是在放宽条件以及智能体之间可以通信的情况下，也算是一个不错的算法。

![](https://pic2.zhimg.com/v2-2d0143f4ad909d358f47978926798a65_b.jpg)![](https://pic2.zhimg.com/80/v2-2d0143f4ad909d358f47978926798a65_hd.jpg)

**相关连接：**

[ECKai：多智能体强化学习入门（一）——基础知识与博弈​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/53474965)[ECKai：多智能体强化学习入门（二）——基础算法（MiniMax-Q，NashQ，FFQ，WoLF-PHC）​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/53563792)[ECKai：多智能体强化学习入门（三）——矩阵博弈中的分布式学习算法​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/53622102)[ECKai：多智能体强化学习入门（四）——MADDPG 算法​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/53811876)[ECKai：多智能体强化学习入门（五）——QMIX 算法分析​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/55003734)

附录：

[ECKai：附：强化学习——DRQN 分析详解​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/54898904)
写下你的评论...

博主您好，最近我想做的就是用强化学习方法求纳什均衡点，想用 Lr-i 算法，请问可以提供编程的思路吗？这几天看懂了混合博弈，但是算法编程还不大会