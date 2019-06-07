> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 https://zhuanlan.zhihu.com/p/66571753

> Thien, Nguyen & Kumar, Akshat & Lau, Hoong. (2017). Policy Gradient With Value Function Approximation For Collective Multiagent Planning.
> 
> 最近一直在忙自己的论文，很久都没有上知乎了。这篇文章的角度很新颖，算法设计跟公式推导都很巧妙，可以借鉴一下。

## 一、 引言

非中心部分可观马尔可夫决策过程（Dec-POMDP）是一种处理多智能体序列化决策问题的模型，这篇文章针对其中一个子类 CDec-POMDP 展开研究，CDec-POMDP 表示群体的集体行动（collective behavior）会影响联合奖励（joint-reward）和环境动力学。针对 CDec-POMDP，文章设计了一种 Actor-Critic 算法，利用局部奖励信号就能够训练 Critic，利用基于数量（在每个状态智能体的数量）方式训练 Actor，能够处理大规模多智能体强化学习问题，文章对 8000 规模的出租车调度问题取得了较好的仿真效果。

Dec-POMDP 假设每个智能体根据自身部分可观的局部观测信息来行动，并最大化全局目标值。但是针对大规模的多智能体问题，计算复杂度较高，往往比较难收敛。目前也有很多文献来解决大规模计算复杂的问题，本篇文章从解耦全局奖励和基于数量的方法给出了一种新的 AC 算法来降低计算复杂度，并且利用神经网络参数化 Actor 与 Critic，采用集中式学习分布式执行的方式。但是这个方法需要一直维护一个计数器，能够实时获取处于各个状态的智能体数量。

## 二、 CDec-POMDP 模型

本节简单介绍一下 CDec-POMDP 模型，在通常的 MDP 模型中做了一些变化，更针对性地处理智能体之间的交互问题和大规模数量问题，可以先以一个城市出租车的调度问题来理解模型。最大化一个区域的出租车收益，在每个时刻 ![](https://www.zhihu.com/equation?tex=t) ，出租车能够观测到自己所在的区域 ![](https://www.zhihu.com/equation?tex=s_t) 、在这个区域的出租车数量、在相邻区域的出租车数量和出租车的需求估计，每个出租车能够通过观测来判断自己是留在当前区域还是到另一个区域。在这个问题中，智能体都是同构的，则能够利用处于不同状态或动作的数量来简化算法结构。

T 步 CDec-POMDP 模型如下图所示：

![](https://pic3.zhimg.com/v2-d53bff4cd6840a1ee52dc829bbd1c7aa_b.jpg)![](https://pic3.zhimg.com/v2-d53bff4cd6840a1ee52dc829bbd1c7aa_r.jpg)

其中智能体总数为 M，单个智能体 ![](https://www.zhihu.com/equation?tex=m) 的状态空间为 ![](https://www.zhihu.com/equation?tex=S) ，简写为 ![](https://www.zhihu.com/equation?tex=i%5Cin+S) （后文中 ![](https://www.zhihu.com/equation?tex=i) 都表示为一个状态）， ![](https://www.zhihu.com/equation?tex=s_t%5Em) 代表智能体 ![](https://www.zhihu.com/equation?tex=m) 在时刻 ![](https://www.zhihu.com/equation?tex=t) 的状态，联合状态空间为 ![](https://www.zhihu.com/equation?tex=S%5EM) 。 ![](https://www.zhihu.com/equation?tex=A) 为每个智能体的动作空间，简写为 ![](https://www.zhihu.com/equation?tex=j%5Cin+A) （后文中 ![](https://www.zhihu.com/equation?tex=j) 都表示为一个动作）， ![](https://www.zhihu.com/equation?tex=a_t%5Em) 代表智能体 ![](https://www.zhihu.com/equation?tex=m) 在时刻 ![](https://www.zhihu.com/equation?tex=t) 的动作。

下面针对算法中要用到的数量进行定义：

![](https://pic4.zhimg.com/v2-395dd25bf876d3cfc83651d3ba7aef33_b.jpg)![](https://pic4.zhimg.com/v2-395dd25bf876d3cfc83651d3ba7aef33_r.jpg)

模型假设，每个智能体不仅能够观测到局部状态 ![](https://www.zhihu.com/equation?tex=s_t%5Em) ，还能够获得局部观测 ![](https://www.zhihu.com/equation?tex=o_t%5Em) ，局部观测 ![](https://www.zhihu.com/equation?tex=o_t%5Em) 是 ![](https://www.zhihu.com/equation?tex=s_t%5Em) 与 ![](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bn%7D_%7Bs_t%7D) 的函数。这里需要注意一个前提假设，局部状态为 ![](https://www.zhihu.com/equation?tex=s_t%5Em) 的智能体能够观测到当前时刻处于状态 ![](https://www.zhihu.com/equation?tex=i) 的所有智能体的数量 ![](https://www.zhihu.com/equation?tex=n_t%28i%29) ，更进一步，还能够观测到状态 ![](https://www.zhihu.com/equation?tex=i) 邻居状态的智能体数量 ![](https://www.zhihu.com/equation?tex=n_t%28i%27%29%5Cforall+i%27%5Cin+Nb%28i%29) （ ![](https://www.zhihu.com/equation?tex=Nb%28i%29) 表示状态 ![](https://www.zhihu.com/equation?tex=i) 的邻居状态）。为了便于理解邻居状态，举个后文中的仿真例子，对出租车调度问题，一个区域可以看做一个状态，邻居状态就是这个区域相邻的区域。

CDec-POMDP 模型的状态转移概率 ![](https://www.zhihu.com/equation?tex=%5Cphi_t%28s_%7Bt%2B1%7D%5Em%3Di%27%7Cs_t%5Em%3Di%2Ca_t%5Em%3Dj%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29) 对所有的智能体都是相同的，并且受群体中所有智能体动作的影响（ ![](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bn_%7Bs_t%7D%7D) 也是状态转移的条件）。

智能体 ![](https://www.zhihu.com/equation?tex=m) 的策略定义为 ![](https://www.zhihu.com/equation?tex=%5Cpi_t%5Em%28j%7Ci%2Co_t%5Em%28i%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29) ，收到的局部奖励定义为 ![](https://www.zhihu.com/equation?tex=r_t%5Em%3Dr_t%28i%2Cj%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29) ，初始状态分布为 ![](https://www.zhihu.com/equation?tex=b_o%3D%28P%28i%29%5Cforall+i%5Cin+S%29) 。

上述定义和假设将智能体都看作同构的，具有相同的状态转移函数、观测和奖励函数。但本文算法也能够拓展到异构的智能体模型中，并且可以引入与智能体动作无关的环境状态信息（环境自身的变化信息）。

## 三、 算法设计

### 3.1\. 值函数设计

令 ![](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bn%7D_%7B1%3AT%7D%3D%5C%7B%28%5Cmathbf%7Bn_%7Bs_t%7D%7D%2C%5Cmathbf%7Bn_%7Bs_t%2Ca_t%7D%7D%2C%5Cmathbf%7Bn_%7Bs_t%2Ca_t%2Cs_%7Bt%2B1%7D%7D%7D%29%5Cforall+t%3D1%3AT%5C%7D) 为 T 步计数向量，记录每一时刻 ![](https://www.zhihu.com/equation?tex=t) 的数量信息，用于后文 Actor 与 Critic 的学习。为了便于分析，这里假定所有智能体都是同构的，具有相同的策略 ![](https://www.zhihu.com/equation?tex=%5Cpi) 。定义联合值函数如下

![](https://www.zhihu.com/equation?tex=V%28%5Cpi%29%3D%5Csum_%7Bm%3D1%7D%5EM%5Csum_%7Bt%3D1%7D%5ETE%5Br_t%5Em%5D%3D%5Csum_%7B%5Cmathbf%7Bn%7D_%7B1%3AT%7D%5Cin+%5COmega_%7B1%3AT%7D%7DP%28%5Cmathbf%7Bn%7D_%7B1%3AT%7D%3B%5Cpi%29%5B%5Csum_%7Bt%3D1%7D%5ET%5Csum_%7Bi%5Cin+S%2Cj%5Cin+A%7Dn_t%28i%2Cj%29r_t%28i%2Cj%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%5D+%5Cqquad+%281%29%5C%5C)

![](https://www.zhihu.com/equation?tex=%5COmega_%7B1%3AT%7D) 指允许的计数空间，其内元素满足如下约束

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%5Csum_%7Bi%5Cin+S%7Dn_t%28i%29%26%3DM%5Cforall+t%5C%5C+%5Csum_%7Bj%5Cin+A%7Dn_t%28i%2Cj%29%26%3Dn_t%28i%29%5Cforall+j%2C%5Cforall+t%5C%5C+%5Csum_%7Bi%27%5Cin+S%7Dn_t%28i%2Cj%2Ci%27%29%26%3Dn_t%28i%2Cj%29%5Cforall+i%5Cin+S%2C%5Cforall+j%5Cin+A%2C+%5Cforall+t+%5Cend%7Baligned%7D)

使用 (1) 这种形式的能够直接通过计数采样 ![](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bn%7D_%7B1%3AT%7D) 来估计值函数，而不需要采集大量智能体的轨迹，减少了计算的复杂度。 ![](https://www.zhihu.com/equation?tex=P%28%5Cmathbf%7Bn%7D_%7B1%3AT%7D%3B%5Cpi%29) 是关于计数的分布，定义如下

![](https://www.zhihu.com/equation?tex=P%28%5Cmathbf%7Bn%7D_%7B1%3AT%7D%3B%5Cpi%29%3Dh%28%5Cmathbf%7Bn%7D_%7B1%3AT%7D%29f%28%5Cmathbf%7Bn%7D_%7B1%3AT%7D%3B%5Cpi%29)

![](https://www.zhihu.com/equation?tex=f%28%5Cmathbf%7Bn%7D_%7B1%3AT%7D%3B%5Cpi%29) 表示一条特定的轨迹的概率，由于智能体是同构的，则能够通过多种轨迹获得相同的计数采样结果， ![](https://www.zhihu.com/equation?tex=h%28%5Cmathbf%7Bn%7D_%7B1%3AT%7D%29) 表示能够组成这种计数采样结果的轨迹数量。分别定义为

![](https://www.zhihu.com/equation?tex=f%28%5Cmathbf%7Bn%7D_%7B1%3AT%7D%3B%5Cpi%29%3D%5Cprod+P%28i%29%5E%7Bn_1%28i%29%7D%5Cprod_%7Bt%3D1%7D%5E%7BT-1%7D%5Cprod_%7Bi%2Cj%2Ci%27%7D%5B%5Cpi_t%28j%7Ci%2Co_t%28i%2C%5Cmathbf%7Bn%7D_t%5Es%29%29%5E%7Bn_t%28i%2Cj%29%7D%5Cphi_t%28i%27%7Ci%2Cj%2C%5Cmathbf%7Bn%7D_t%5Es%29%5E%7Bn_t%28i%2Cj%2Ci%27%29%7D%5D%5Cprod_%7Bi%2Cj%7D%5Cpi_%7BT%7D%28j%7Ci%2Co_T%28i%2C%5Cmathbf%7Bn%7D_T%5Es%29%29%5E%7Bn_T%28i%2Cj%29%7D%5C%5C+h%28%5Cmathbf%7Bn%7D_%7B1%3AT%7D%29%3D%5Cfrac%7BM%21%7D%7B%5Cprod_%7Bi%5Cin+S%7Dn_1%28i%29%21%7D%5B%5Cprod_%7Bt%3D1%7D%5E%7BT-1%7D%5Cprod_%7Bi%5Cin+S%7D%5Cfrac%7Bn_t%28i%29%21%7D%7B%5Cprod_%7Bi%27%5Cin+S%2Cj%5Cin+A%7Dn_t%28i%2Cj%2Ci%27%29%7D%5D%5Ctimes%5B%5Cprod_%7Bi%5Cin+S%7D%5Cfrac%7Bn_t%28i%29%21%7D%7B%5Cprod_%7Bj%5Cin+A%7Dn_t%28i%2Cj%29%21%7D%5D)

![](https://www.zhihu.com/equation?tex=f%28%5Cmathbf%7Bn%7D_%7B1%3AT%7D%3B%5Cpi%29) 很容易理解，就是这条轨迹每一步的概率相乘。 ![](https://www.zhihu.com/equation?tex=h%28%5Cmathbf%7Bn%7D_%7B1%3AT%7D%29) 可以看做是一个组合数。举个例子，将 12 个球分成 ![](https://www.zhihu.com/equation?tex=%283%2C4%2C5%29) 三组，利用排列数公式 ![](https://www.zhihu.com/equation?tex=C_%7B12%7D%5E3+C_%7B9%7D%5E4+C_%7B5%7D%5E5%3D%5Cfrac%7B12%21%7D%7B3%214%215%21%7D) 。 ![](https://www.zhihu.com/equation?tex=h%28%5Cmathbf%7Bn%7D_%7B1%3AT%7D%29) 与这个例子相同，将时间序列 ![](https://www.zhihu.com/equation?tex=t) 的所有排列乘到一起。

### 3.2\. 策略梯度设计

令 ![](https://www.zhihu.com/equation?tex=%5Ctheta) 为策略的 ![](https://www.zhihu.com/equation?tex=%5Cpi_%5Ctheta) 的参数，采用随机策略梯度的方法，令 ![](https://www.zhihu.com/equation?tex=s_t%2Ca_t) 为联合状态和联合动作，关于策略 ![](https://www.zhihu.com/equation?tex=%5Cpi_%5Ctheta) 的全局值函数可以定义为

![](https://www.zhihu.com/equation?tex=V_t%28%5Cpi%29%3D%5Csum_%7Bs_t%2Ca_t%7DP%5E%7B%5Cpi%7D%28s_t%2Ca_t%7Cb_0%2C%5Cpi%29Q_t%5E%5Cpi%28s_t%2Ca_t%29)

其中 ![](https://www.zhihu.com/equation?tex=P%5E%7B%5Cpi%7D%28s_t%2Ca_t%7Cb_o%2C%5Cpi%29%3D%5Csum_%7Bs_%7B1%3At%7D%2Ca_%7B1%3At%7D%7DP%5E%5Cpi%28s_%7B1%3At%7D%2Ca_%7B1%3At%7D%7Cb_0%29) 指按照 ![](https://www.zhihu.com/equation?tex=%5Cpi) 执行联合状态 - 动作对的分布。 ![](https://www.zhihu.com/equation?tex=Q_t%5E%5Cpi%28s_t%2Ca_t%29) 是全局动作值函数定义如下

![](https://www.zhihu.com/equation?tex=Q_t%5E%5Cpi%28s_t%2Ca_t%29%3Dr_t%28s_t%2Ca_t%29%2B%5Csum_%7Bs_%7Bt%2B1%7D%2Ca_%7Bt%2B1%7D%7DP%5E%5Cpi%28s_%7Bt%2B1%7D%2Ca_%7Bt%2B1%7D%7Cs_t%2Ca_t%29Q_%7Bt%2B1%7D%5E%5Cpi%28s_%7Bt%2B1%7D%2Ca_%7Bt%2B1%7D%29%5Cqquad+%282%29)

若所有智能体同构，策略梯度公式可以写为

![](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Ctheta%7DV_1%28%5Cpi%29%3D%5Csum_%7Bt%3D1%7D%5ETE_%7Bs_t%2Ca_t%7Cb_0%2C%5Cpi%7D%5BQ_%7Bt%7D%5E%5Cpi%28s_t%2Ca_t%29%5Csum_%7Bi%5Cin+S%2Cj%5Cin+A%7Dn_t%28i%2Cj%29%5Cnabla_%7B%5Ctheta%7D%5Clog%5Cpi_t%28j%7Ci%2Co%28i%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29%5D%5Cqquad+%283%29)

推导过程与随机策略梯度算法类似，如下：

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%5Cfrac%7B%5Cpartial+V_0%7D%7B%5Cpartial+%5Ctheta%7D%26%3D%5Csum_%7Bs_0%2Ca_0%7D%5Cnabla_%7B%5Ctheta%7D%28P%5E%5Cpi%28s_0%2Ca_0%7Cb_0%2C%5Cpi%29Q_%7B0%7D%5E%5Cpi%28s_0%2Ca_0%29%29%5C%5C+%26%3D%5Csum_%7Bs_0%2Ca_0%7DQ_%7B0%7D%5E%5Cpi%28s_0%2Ca_0%29%5Cnabla_%7B%5Ctheta%7DP%5E%5Cpi%28s_0%2Ca_0%7Cb_0%2C%5Cpi%29%2B%5Csum_%7Bs_0%2Ca_0%7DP%5E%5Cpi%28s_0%2Ca_0%7Cb_0%2C%5Cpi%29%5Cnabla_%7B%5Ctheta%7DQ_%7B0%7D%5E%5Cpi%28s_0%2Ca_0%29%5C%5C+%26%3D%5Csum_%7Bs_0%2Ca_0%7DQ_%7B0%7D%5E%5Cpi%28s_0%2Ca_0%29%5Cnabla_%7B%5Ctheta%7DP%5E%5Cpi%28s_0%2Ca_0%7Cb_0%2C%5Cpi%29%2B%5Csum_%7Bs_0%2Ca_0%7DP%5E%5Cpi%28s_0%2Ca_0%7Cb_0%2C%5Cpi%29%5Cnabla_%7B%5Ctheta%7D%5B%5Csum_%7Bs_1%2Ca_1%7DP%28s_1%2Ca_1%7Cs_0%2Ca_0%2C%5Cpi%29Q_1%5E%5Cpi%28s_1%2Ca_1%29%5D%5C%5C+%5Cend%7Baligned%7D)

逐渐将 (2) 代入上式第二部分展开，可得到如下的规范形式

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%5Cfrac%7B%5Cpartial+V_0%7D%7B%5Cpartial+%5Ctheta%7D%26%3D%5Csum_t%5Csum_%7Bs_%7B1%3At%7D%2Ca_%7B1%3At%7D%7DQ_t%5E%5Cpi%28s_t%2Ca_t%29P%28s_%7Bt-1%7D%2Ca_%7Bt-1%7D%7Cb_0%2C%5Cpi%29%5Cnabla_%7B%5Ctheta%7DP%28s_t%2Ca_t%7Cs_%7Bt-1%7D%2Ca_%7Bt-1%7D%2C%5Cpi%29%5C%5C+%26%3D%5Csum_t%5Csum_%7Bs_%7B1%3At%7D%2Ca_%7B1%3At%7D%7DQ_t%5E%5Cpi%28s_t%2Ca_t%29P%28s_t%2Ca_t%7Cb_0%2C%5Cpi%29%5Cnabla_%7B%5Ctheta%7D%5Clog+P%28s_t%2Ca_t%7Cs_%7Bt-1%7D%2Ca_%7Bt-1%7D%2C%5Cpi%29%5C%5C+%26%3DE_%7Bs_t%2Ca_t%7Cb_0%2C%5Cpi%7D%5B%5Csum_t+Q_t%5E%5Cpi%28s_t%2Ca_t%29%5Cnabla_%7B%5Ctheta%7D%5Clog+P%28s_t%2Ca_t%7Cs_%7Bt-1%7D%2Ca_%7Bt-1%7D%2C%5Cpi%29%5D%5C%5C+%26%3DE_%7Bs_t%2Ca_t%7Cb_0%2C%5Cpi%7D%5B%5Csum_t+Q_t%5E%5Cpi%28s_t%2Ca_t%29%5Cnabla_%7B%5Ctheta%7D%5Clog+P%28a_t%7Cs_%7Bt%7D%2C%5Cpi%29%5D%5Cqquad+%284%29%5C%5C+%5Cend%7Baligned%7D)

![](https://www.zhihu.com/equation?tex=%5Csum_%7Bs_%7B1%3At%7D%2Ca_%7B1%3At%7D%7DP%28s_t%2Ca_t%7Cb_0%2C%5Cpi%29) 是通过条件概率 ![](https://www.zhihu.com/equation?tex=%5Csum_%7Bs_0%2Ca_0%7DP%28s_0%2Ca_0%7Cb_0%2C%5Cpi%29%5Csum_%7Bs_1%2Ca_1%7DP%28s_1%2Ca_1%7Cs_0%2Ca_0%2C%5Cpi%29%5Ccdots+%5Csum_%7Bs_t%2Ca_t%7DP%28s_t%2Ca_t%7Cs_%7Bt-1%7D%2Ca_%7Bt-1%7D%2C%5Cpi%29) 累乘得到的。

为了方便计算，对 ![](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Ctheta%7D%5Clog+P%28a_t%7Cs_%7Bt%7D%2C%5Cpi%29) 进行转化

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%5Cnabla_%7B%5Ctheta%7D%5Clog+P%28a_t%7Cs_%7Bt%7D%2C%5Cpi%29%26%3D%5Cnabla_%7B%5Ctheta%7D%5Clog%28%5Cprod_m+%5Cpi_t%5Em%28a_t%5Em%7Co%28s_t%5Em%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29%5C%5C+%26%3D%5Csum_m%5Cnabla_%7B%5Ctheta%7D%5Clog%28+%5Cpi_t%5Em%28a_t%5Em%7Co%28s_t%5Em%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29%5Cqquad+%285%29%5C%5C+%5Cend%7Baligned%7D)

如上， ![](https://www.zhihu.com/equation?tex=%5Cpi%5Em) 为属于每个智能体不同的策略。在同构的智能体系统中，所有智能体使用同一个策略，可以化为如下更简便形式

![](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Ctheta%7D%5Clog+P%28a_t%7Cs_%7Bt%7D%2C%5Cpi%29%3D%5Csum_%7Bi%5Cin+S%2Cj%5Cin+A+%7Dn_t%28i%2Cj%29%5Cnabla_%7B%5Ctheta%7D%5Cpi_t%28j%7Co%28i%2C%5Cmathbf%7Bn_%7Bs_%7Bt%7D%7D%7D%29%29%5Cqquad+%286%29)

将 (6) 代入 (4) 可得梯度公式(3)。

如上策略梯度公式 (3) 所示，在实际应用中还是无法很好地处理智能体数目很多的问题。主要在于：1\. 关于联合状态与联合动作 ![](https://www.zhihu.com/equation?tex=%28s_t%2Ca_t%29) 求期望，需要采样大量的轨迹，并且随着智能体数量呈指数增长，计算复杂度太高，后文转化为关于数量求期望。2\. 根据轨迹获得 ![](https://www.zhihu.com/equation?tex=Q_t%5E%5Cpi%28s_t%2Ca_t%29) 较困难，方差大，不易操作，后文通过 Critic 来估计。

### 3.3\. 值函数重构

为了解决上述两个问题，使算法更具实用性。考虑将值函数重构为如下形式

![](https://www.zhihu.com/equation?tex=Q_t%5E%5Cpi%28s_t%2Ca_t%29%5Capprox+f_%5Comega%28s_t%2Ca_t%29%3D%5Csum_%7Bm%3D1%7D%5Emf_%7B%5Comega%7D%5Em%28s_t%5Em%2Co%28s_t%5Em%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%2Ca_%7Bt%7D%5Em%29%5Cqquad+%287%29)

![](https://www.zhihu.com/equation?tex=f_%7B%5Comega%7D%5Em) 定义为智能体 ![](https://www.zhihu.com/equation?tex=m) 的局部观测，由于都使用 ![](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bn_%7Bs_t%7D%7D) 作为输入，因此局部观测之间存在一定的联系。这种解构方式具有两点好处：1\. 简化策略梯度的计算，使其效率更高。2\. 解构的方式更容易进行学习逼近。

并且这种估计对于策略梯度是无偏估计， ![](https://www.zhihu.com/equation?tex=f_%5Comega%28s_t%2Ca_t%29) 的构建过程如下：

假设通过线性函数逼近 ![](https://www.zhihu.com/equation?tex=Q_t%5E%5Cpi%28s_t%2Ca_t%29) ，则写为 ![](https://www.zhihu.com/equation?tex=f_%5Comega%28s_t%2Ca_t%29%3D%5Comega%5ET%5Cphi%28s_t%2Ca_t%29) ， ![](https://www.zhihu.com/equation?tex=%5Comega) 为参数。根据 (5), 特征向量 ![](https://www.zhihu.com/equation?tex=%5Cphi%28s_t%2Ca_t%29) 可以写为

![](https://www.zhihu.com/equation?tex=%5Cphi%28s_t%2Ca_t%29%3D%5Cnabla_%7B%5Ctheta%7D%5Clog+P%28a_t%7Cs_t%29%3D%5Csum_m+%5Cnabla_%7B%5Ctheta%7D%5Clog+%5Cpi_t%5Em%28a%5Em%7Co%28s_t%5Em%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29)

代入 ![](https://www.zhihu.com/equation?tex=f_%5Comega%28s_t%2Ca_t%29) 可得

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+f_%5Comega%28s_t%2Ca_t%29%26%3D%5Comega%5ET%5Cphi%28s_t%2Ca_t%29%3D%5Comega%5ET%5B%5Csum_m+%5Cnabla_%7B%5Ctheta%7D%5Clog+%5Cpi_t%5Em%28a%5Em%7Co%28s_t%5Em%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29%5D%5C%5C+%26%3D%5Csum_%7Bm%7D%5Comega%5ET%5Cnabla_%7B%5Ctheta%7D%5Clog+%5Cpi_t%5Em%28a%5Em%7Co%28s_t%5Em%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29%5C%5C+%26%3D%5Csum_m+f_%7B%5Comega%7D%5Em%28s_t%5Em%2Co%28s_t%5Em%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%2Ca_%7Bt%7D%5Em%29%5Cquad+%288%29+%5Cend%7Baligned%7D)

如上 ![](https://www.zhihu.com/equation?tex=f_%7B%5Comega%7D%5Em%28s_t%5Em%2Co%28s_t%5Em%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%2Ca_%7Bt%7D%5Em%29%3D%5Comega%5ET%5Cnabla_%7B%5Ctheta%7D%5Clog+%5Cpi_t%5Em%28a%5Em%7Co%28s_t%5Em%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29) 。

上述推导过程是针对每个智能体有自己异构的策略 ![](https://www.zhihu.com/equation?tex=%5Cpi%5Em) ，将 (7) 代入 (4) 可得

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%5Cnabla_%7B%5Ctheta%7DV_1%28%5Cpi%29%26%3D%5Csum_tE_%7Bs_t%2Ca_t%7Cb_0%2C%5Cpi%7D%5B+%5Cnabla_%7B%5Ctheta%7D%5Clog+P%28a_t%7Cs_%7Bt%7D%2C%5Cpi%29f_%5Comega%28s_t%2Ca_t%29%5D%5C%5C+%26%3D%5Csum_tE_%7Bs_t%2Ca_t%7Cb_0%2C%5Cpi%7D%5B+%5Cfrac%7B%5Cpartial+%5Clog%5Cprod_%7Bm%7D%5Cpi%5Em%28a_t%5Em%7Cs_t%5Em%2Co%28s_t%5Em%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29%7D%7B%5Cpartial+%5Ctheta%7D%28%5Csum_%7Bm%27%7D+f_%7B%5Comega%7D%5E%7Bm%27%7D%28s_t%5E%7Bm%27%7D%2Co%28s_t%5E%7Bm%27%7D%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%2Ca_%7Bt%7D%5E%7Bm%27%7D%29%29%5D%5C%5C+%26%3D%5Csum_tE_%7Bs_t%2Ca_t%7Cb_0%2C%5Cpi%7D%5B+%5Csum_%7Bm%7D%5Cnabla_%7B%5Ctheta%7D+%5Clog%5Cpi%5Em%28a_t%5Em%7Cs_t%5Em%2Co%28s_t%5Em%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29%28%5Csum_%7Bm%27%7D+f_%7B%5Comega%7D%5E%7Bm%27%7D%28s_t%5E%7Bm%27%7D%2Co%28s_t%5E%7Bm%27%7D%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%2Ca_%7Bt%7D%5E%7Bm%27%7D%29%29%5D%5Cqquad+%289%29%5C%5C+%5Cend%7Baligned%7D)

简化上式期望中间的乘项，针对智能体 ![](https://www.zhihu.com/equation?tex=m) 的策略梯度与其他智能体的奖励值无关，固定 ![](https://www.zhihu.com/equation?tex=m%2Ct) 可以得到

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%26E_%7Bs_t%2Ca_t%7D%5B+%5Cnabla_%7B%5Ctheta%7D+%5Clog%5Cpi%5Em%28a_t%5Em%7Cs_t%5Em%2Co%28s_t%5Em%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29%28%5Csum_%7Bm%27%5Cne+m%7D+f_%7B%5Comega%7D%5E%7Bm%27%7D%28s_t%5E%7Bm%27%7D%2Co%28s_t%5E%7Bm%27%7D%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%2Ca_%7Bt%7D%5E%7Bm%27%7D%29%29%5D%5C%5C+%26%3DE_%7Bs_t%7D%5BE_%7Ba_t%7Cs_t%7D%5B+%5Cnabla_%7B%5Ctheta%7D+%5Clog%5Cpi%5Em%28a_t%5Em%7Cs_t%5Em%2Co%28s_t%5Em%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29%28%5Csum_%7Bm%27%5Cne+m%7DE_%7Ba_t%5E%7Bm%27%7D%7Cs_t%7D+f_%7B%5Comega%7D%5E%7Bm%27%7D%28s_t%5E%7Bm%27%7D%2Co%28s_t%5E%7Bm%27%7D%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%2Ca_%7Bt%7D%5E%7Bm%27%7D%29%29%5D%5D%5C%5C+%26%3DE_%7Bs_t%7D%5BE_%7Ba_t%7Cs_t%7D%5B+%5Cnabla_%7B%5Ctheta%7D+%5Clog%5Cpi%5Em%28a_t%5Em%7Cs_t%5Em%2Co%28s_t%5Em%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29%5Ctimes+constant%5C_to%5C_a_t%5Em%5D%5D%5C%5C+%26%3DE_%7Bs_t%7D%5Bconstant+%5Ctimes++%5Cnabla_%7B%5Ctheta%7D+%5Clog+%5Cint_%7Ba_t%5Em%7D%5Cpi%5Em%28a_t%5Em%7Cs_t%5Em%2Co%28s_t%5Em%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29%5D%5C%5C+%26%3DE_%7Bs_t%7D%5Bconstant+%5Ctimes++%5Cnabla_%7B%5Ctheta%7D+%5Clog+1%5D%5C%5C+%26%3D0%5C%5C+%5Cend%7Baligned%7D)

利用上述简化 (9) 可得策略梯度为

![](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Ctheta%7DV_1%28%5Cpi%29%3D%5Csum_tE_%7Bs_t%2Ca_t%7Cb_0%2C%5Cpi%7D%5B+%5Csum_%7Bm%7D%5Cnabla_%7B%5Ctheta%7D+%5Clog%5Cpi%5Em%28a_t%5Em%7Cs_t%5Em%2Co%28s_t%5Em%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29+f_%7B%5Comega%7D%5E%7Bm%7D%28s_t%5E%7Bm%7D%2Co%28s_t%5E%7Bm%7D%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%2Ca_%7Bt%7D%5E%7Bm%7D%29%5D+%5Cqquad+%2810%29)

如上 (10) 所示，利用重构后的值函数，将策略梯度公式的计算进行了简化，并且每个智能体是异构的，有自己独立的 ![](https://www.zhihu.com/equation?tex=%5Cpi%5Em%2Cf_%7B%5Comega%7D%5Em) 。

如果所有智能体都是同构的，(8) 能够做出如下简化

![](https://www.zhihu.com/equation?tex=f_%7B%5Comega%7D%28s_t%2Ca_t%29%3D%5Csum_%7Bi%2Cj%7Dn_t%28i%2Cj%29f_%7B%5Comega%7D%28i%2Cj%2Co%28i%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29%5Cqquad+%2811%29)

直接将上述 (11) 公式代入 (3) 可以得到同构的策略梯度公式

![](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Ctheta%7DV_1%28%5Cpi%29%3D%5Csum_%7Bt%7DE_%7Bs_t%2Ca_t%7D%5B%5Csum_%7Bi%2Cj%7Dn_t%28i%2Cj%29%5Cnabla_%7B%5Ctheta%7D%5Clog+%5Cpi%28j%7Ci%2Co%28i%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29f_%7B%5Comega%7D%28i%2Cj%2Co%28i%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29%5D%5Cqquad+%2812%29)

### 3.4\. 基于数量的策略梯度改进

观察 (12) 能够看出，对 ![](https://www.zhihu.com/equation?tex=%28s_t%2Ca_t%29) 求期望在实际应用中仍然会因为智能体数量过多而带来采样复杂和计算困难的问题，如果能够将 (12) 改进为基于 ![](https://www.zhihu.com/equation?tex=%28s_t%2Ca_t%29) 的数量进行求期望，将会大大降低计算复杂度。因此做如下转化

![](https://www.zhihu.com/equation?tex=+%5Cnabla_%7B%5Ctheta%7DV_1%28%5Cpi%29%3DE_%7B%5Cmathbf%7Bn%7D_%7B1%3AT%7D%5Cin+%5COmega_%7B1%3AT%7D%7D%5B%5Csum_%7Bt%7D%5Csum_%7Bi%2Cj%7Dn_t%28i%2Cj%29%5Cnabla_%7B%5Ctheta%7D%5Clog+%5Cpi%28j%7Ci%2Co%28i%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29f_%7B%5Comega%7D%28i%2Cj%2Co%28i%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29%5D%5Cqquad+%2813%29)

如上，只需要通过对数量进行采样就能实现策略梯度的估计。

### 3.5\. 值函数更新方法

本小节想通过转化，使值函数的更新也只需要数量信息。通过观察 (11)，其变量与联合状态 - 联合动作 ![](https://www.zhihu.com/equation?tex=%28s_t%2Ca_t%29) 的元素组成数量有关。与 DQN 等算法相同，值函数 ![](https://www.zhihu.com/equation?tex=f_%5Comega) 的代价函数可以写为

![](https://www.zhihu.com/equation?tex=%5Cmin_%7B%5Comega%7D%5Csum_%7Bt%3D1%7D%5ET%28f_%5Comega%28%5Cmathbf%7Bn_%7Bs_t%7D%7D%29-R_t%29%5E2%5Cqquad+%2814%29)

其中 ![](https://www.zhihu.com/equation?tex=f_%7B%5Comega%7D%28%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%3D%5Csum_%7Bi%2Cj%7Dn_t%28i%2Cj%29f_%7B%5Comega%7D%28i%2Cj%2Co%28i%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29) ， ![](https://www.zhihu.com/equation?tex=R_t%3D%5Csum_%7Bt%3D1%7D%5ET%5Csum_%7Bi%2Cj%7Dn_t%28i%2Cj%29r_t%28i%2Cj%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29) 。在多智能体强化学习中，局部奖励值相对于全局奖励值来说具有一定的优势，能处理复杂的合作对抗问题，并且计算上更加灵活有效。因此，这篇文章利用局部的奖励函数来更新全局 Critic ![](https://www.zhihu.com/equation?tex=f_%5Comega) 。

定义智能体 ![](https://www.zhihu.com/equation?tex=m) 在时刻 ![](https://www.zhihu.com/equation?tex=t) 状态 ![](https://www.zhihu.com/equation?tex=s_t%5Em) 动作 ![](https://www.zhihu.com/equation?tex=a_t%5Em) 的局部值函数为

![](https://www.zhihu.com/equation?tex=V_t%5Em%28i%2Cj%29%3DE%5B%5Csum_%7Bt%27%3Dt%7D%5ETr_%7Bt%27%7D%5Em%7Cs_t%5Em%3Di%2Ca_t%5Em%3Dj%2C%5Cmathbf%7Bn_%7B1%3AT%7D%7D%5D)

这个值函数可以利用动态规划的方法更新，我感觉也可以在构建一个 DQN 进行拟合。利用局部值函数重构 ![](https://www.zhihu.com/equation?tex=R_t) 如下

![](https://www.zhihu.com/equation?tex=R_t%3D%5Csum_%7Bi%2Cj%7Dn_t%28i%2Cj%29V_t%5Em%28i%2Cj%29)

为了更便于大规模计算，确定代价函数 (14) 的上界

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%5Csum_%7Bt%3D1%7D%5ET%28f_%5Comega%28%5Cmathbf%7Bn_%7Bs_t%7D%7D%29-R_t%29%5E2%26%3D%5Csum_t%5B%5Csum_%7Bi%2Cj%7Dn_t%28i%2Cj%29f_%5Comega%28i%2Cj%2Co%28i%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29-%5Csum_%7Bi%2Cj%7Dn_t%28i%2Cj%29V_t%5Em%28i%2Cj%29%5D%5E2%5Cqquad+%2815%29%5C%5C+%26%3D%5Csum_t%5B%5Csum_%7Bi%2Cj%7Dn_t%28i%2Cj%29%28f_%5Comega%28i%2Cj%2Co%28i%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29-V_t%5Em%28i%2Cj%29%29%5D%5E2%5C%5C+%26%5Cle+M%5Csum_%7Bt%2Ci%2Cj%7Dn_t%28i%2Cj%29%5Bf_%5Comega%28i%2Cj%2Co%28i%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29-V_t%5Em%28i%2Cj%29%5D%5E2%5Cqquad+%2816%29+%5Cend%7Baligned%7D)

最后一步利用了柯西不等式。文章中说明代价函数 (16) 比(14)在大规模系统中具有更快的收敛速度。因为其逼近的局部的值函数， ![](https://www.zhihu.com/equation?tex=f_%5Comega%28i%2Cj%2Co%28i%2C%5Cmathbf%7Bn_%7Bs_t%7D%7D%29%29) 逼近 ![](https://www.zhihu.com/equation?tex=V_t%5Em%28i%2Cj%29) ，分工更具体细化，避免了震荡。

### 3.6\. 算法结构框图

![](https://pic4.zhimg.com/v2-a0c3116535bf5517e1164204c21348ff_b.jpg)![](https://pic4.zhimg.com/80/v2-a0c3116535bf5517e1164204c21348ff_hd.jpg)

fC：利用局部值函数，代价函数 (16) 更新 Critic，计算复杂度较低

C：利用全局值函数，代价函数 (15) 更新 Critic，计算复杂度较高

fA：利用 (11) 进行策略 Actor 的更新，若是异构智能体可以采用(10)，计算复杂度较低

A：利用 (9) 进行策略 Actor 的更新，计算复杂度较高

## 四、仿真结果

原文最后给出了两个仿真：(a). 大规模出租车区域调度问题的仿真，分别设置了 4000 跟 8000 个智能体。(b). 20 个智能体的路径规划问题，避免出现交通堵塞。

![](https://pic3.zhimg.com/v2-d84a9fc8bb3cb0ee0c42a90fb192e62a_b.jpg)![](https://pic3.zhimg.com/80/v2-d84a9fc8bb3cb0ee0c42a90fb192e62a_hd.jpg)

o0：策略只依赖局部状态 ![](https://www.zhihu.com/equation?tex=s_t%5Em) ，不依赖数量信息

o1：策略依赖局部状态信息 ![](https://www.zhihu.com/equation?tex=s_t%5Em) 和自身状态数量 ![](https://www.zhihu.com/equation?tex=n_t%28s_t%5Em%29)

oN：策略依赖局部状态信息 ![](https://www.zhihu.com/equation?tex=s_t%5Em) 和邻居状态数量 ![](https://www.zhihu.com/equation?tex=n_t%28i%27%29%5Cforall+i%27%5Cin+Nb%28i%29) 其中 ![](https://www.zhihu.com/equation?tex=i%3Ds_t%5Em)

EM 与 SMFU：其他多智能体算法

仿真效果可以看出，算法在 8000 个智能体的大规模应用场景中效果更突出，在 20 个智能体的问题中优势不明显。

写下你的评论...