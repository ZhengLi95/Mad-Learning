> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 https://zhuanlan.zhihu.com/p/53474965

## **一、引言**

在多智能体系统中，每个智能体通过与环境进行交互获取奖励值（reward）来学习改善自己的策略，从而获得该环境下最优策略的过程就多智能体强化学习。

在单智能体强化学习中，智能体所在的环境是稳定不变的，但是在多智能体强化学习中，环境是复杂的、动态的，因此给学习过程带来很大的困难。

1.  维度爆炸：在单体强化学习中，需要存储状态值函数或动作 - 状态值函数。在多体强化学习中，状态空间变大，联结动作空间（联结动作是指每个智能体当前动作组合而成的多智能体系统当前时刻的动作，联结动作 ![](https://www.zhihu.com/equation?tex=A_t%3D%5Ba_%7B1%2Ct%7D%2Ca_%7B2%2Ct%7D%2C%5Cdots%2Ca_%7Bn%2Ct%7D%5D%5ET) ， ![](https://www.zhihu.com/equation?tex=a_%7Bi%2Ct%7D) 指第 i 个智能体在时刻 t 选取的动作。）随智能体数量指数增长，因此多智能体系统维度非常大，计算复杂。
2.  目标奖励确定困难：多智能体系统中每个智能体的任务可能不同，但是彼此之间又相互耦合影响。奖励设计的优劣直接影响学习到的策略的好坏。
3.  不稳定性：在多智能体系统中，多个智能体是同时学习的。当同伴的策略改变时，每个智能体自身的最优策略也可能会变化，这将对算法的收敛性带来影响。
4.  探索 - 利用：探索不光要考虑自身对环境的探索，也要对同伴的策略变化进行探索，可能打破同伴策略的平衡状态。每个智能体的探索都可能对同伴智能体的策略产生影响，这将使算法很难稳定，学习速度慢。

在多智能体系统中智能体之间可能涉及到合作与竞争等关系，引入博弈的概念，将博弈论与强化学习相结合可以很好的处理这些问题。

## **二、博弈论基础**

在本节中主要介绍多智能体强化学习中需要用到的一些概念及定义，仅局限于多智能体强化学习算法的理解分析。包括矩阵博弈、静态博弈、阶段博弈、重复博弈和随机博弈等概念。

## **1\. 矩阵博弈**

一个矩阵博弈可以表示为 ![](https://www.zhihu.com/equation?tex=%28n%2CA_1%2CA_2%2C%5Cdots%2CA_n%2CR_1%2CR_2%2C%5Cdots%2CR_n%29) ，n 表示智能体数量， ![](https://www.zhihu.com/equation?tex=A_i) 是第 i 个智能体的动作集， ![](https://www.zhihu.com/equation?tex=R_i%3AA_1%5Ctimes%5Ccdots%5Ctimes+A_n%5Cto%5Cmathbb+R) 表示第 i 个智能体的奖励函数，从奖励函数可以看出每个智能体获得的奖励与多智能体系统的联结动作有关，联结动作空间为 ![](https://www.zhihu.com/equation?tex=A_1%5Ctimes%5Ccdots%5Ctimes+A_n) 。每个智能体的策略是一个关于其动作空间的概率分布，每个智能体的目标是最大化其获得的奖励值。

令 ![](https://www.zhihu.com/equation?tex=V_i%28%5Cpi_1%2C%5Ccdots%2C%5Cpi_i%2C%5Ccdots%2C%5Cpi_n%29) 表示智能体 i 在，联结策略 ![](https://www.zhihu.com/equation?tex=%28%5Cpi_1%2C%5Ccdots%2C%5Cpi_n%29) 下的期望奖励，即值函数。

## 定义 1：纳什均衡

在矩阵博弈中，如果联结策略 ![](https://www.zhihu.com/equation?tex=%28%5Cpi_1%5E%2A%2C%5Ccdots%2C%5Cpi_n%5E%2A%29) 满足

![](https://www.zhihu.com/equation?tex=V_i%28%5Cpi_1%5E%2A%2C%5Ccdots%2C%5Cpi_i%5E%2A%2C%5Ccdots%2C%5Cpi_n%5E%2A%29%5Cge+V_i%28%5Cpi_1%5E%2A%2C%5Ccdots%2C%5Cpi_i%2C%5Ccdots%2C%5Cpi_n%5E%2A%29%2C+%5Cforall%5Cpi_i%5Cin+%5CPi_i%2Ci%3D1%2C%5Ccdots%2Cn%5Cqquad+%281%29)

则为一个纳什均衡。

总体来说，纳什均衡就是一个所有智能体的联结策略。在纳什均衡处，对于所有智能体而言都不能在仅改变自身策略的情况下，来获得更大的奖励。

定义 ![](https://www.zhihu.com/equation?tex=Q_i%28a_1%2C%5Ccdots%2Ca_n%29) 表示在执行联结动作 ![](https://www.zhihu.com/equation?tex=%5Ba_1%2C%5Ccdots%2Ca_n%5D) 时，智能体 i 所能获得的期望奖励。令 ![](https://www.zhihu.com/equation?tex=%5Cpi_i%28a_i%29) 表示第 i 个智能体选取动作 ![](https://www.zhihu.com/equation?tex=a_i) 的概率。则纳什均衡的另一种定义方式如下 ![](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%5Csum_%7Ba_1%2C%5Ccdots%2Ca_n%5Cin+A_1%5Ctimes+%5Ccdots+%5Ctimes+A_n%7DQ_i%28a_1%2C%5Ccdots%2Ca_n%29%5Cpi_1%5E%2A%28a_1%29%5Ccdots%5Cpi_i%5E%2A%28a_i%29%5Ccdots%5Cpi_n%5E%2A%28a_n%29%5Cge+%5C%5C+%5Csum_%7Ba_1%2C%5Ccdots%2Ca_n%5Cin+A_1%5Ctimes+%5Ccdots+%5Ctimes+A_n%7DQ_i%28a_1%2C%5Ccdots%2Ca_n%29%5Cpi_1%5E%2A%28a_1%29%5Ccdots%5Cpi_i%28a_i%29%5Ccdots%5Cpi_n%5E%2A%28a_n%29%2C%5Cforall+%5Cpi_i%5Cin%5CPi_i%2Ci%3D1%2C%5Ccdots%2Cn+%5Cend%7Baligned%7D)

## 定义 2：严格纳什均衡

若 (1) 式严格大于，则 ![](https://www.zhihu.com/equation?tex=%28%5Cpi_1%5E%2A%2C%5Ccdots%2C%5Cpi_n%5E%2A%29) 为严格纳什均衡。

## 定义 3：完全混合策略

若一个策略对于智能体动作集中的所有动作的概率都大于 0，则这个策略为一个完全混合策略。

## 定义 4：纯策略

若智能体的策略对一个动作的概率分布为 1，对其余的动作的概率分布为 0，则这个策略为一个纯策略。

## **2\. 两个智能体的矩阵博弈中的纳什均衡**

本节介绍针对一个两智能体博弈问题的常规建模方式，并介绍几种常见的博弈形式。后面的很多多智能体强化学习算法都是以此为基础建立起来的，双智能体矩阵博弈对于多智能体强化学习类似于感知机对于神经网络。

在双智能体矩阵博弈中，我们可以设计一个矩阵，矩阵每一个元素的索引坐标表示一个联结动作 ![](https://www.zhihu.com/equation?tex=%5BA_1%3Dx%2CA_2%3Dy%5D) ，第 i 个智能体的奖励矩阵 ![](https://www.zhihu.com/equation?tex=R_i) 的元素 ![](https://www.zhihu.com/equation?tex=r_%7Bxy%7D) 就表示第一个智能体采用动作 x，第二个智能体采用动作 y 时第 i 个智能体获得的奖励。通常我们将第一个智能体定义为**行智能体**，第二个智能体定义为**列智能体**，行号表示第一个智能体选取的动作，列号表示第二个智能体选取的动作。则对于只有 2 个动作的智能体，其奖励矩阵分别可以写为

![](https://www.zhihu.com/equation?tex=R_1%3D%5Cbegin%7Bbmatrix%7D+r_%7B11%7D+%26+r_%7B12%7D+%5C%5C+r_%7B21%7D+%26+r_%7B22%7D+%5Cend%7Bbmatrix%7D%2C%5Cqquad+R_2%3D%5Cbegin%7Bbmatrix%7D+c_%7B11%7D+%26+c_%7B12%7D+%5C%5C+c_%7B21%7D+%26+c_%7B22%7D+%5Cend%7Bbmatrix%7D)

## 定义 5\. 零和博弈

零和博弈中，两个智能体是完全竞争对抗关系，则 ![](https://www.zhihu.com/equation?tex=R_1%3D-R_2) 。在零和博弈中只有一个纳什均衡值，即使可能有很多纳什均衡策略，但是期望的奖励是相同的。

## 定义 6\. 一般和博弈

一般和博弈是指任何类型的矩阵博弈，包括完全对抗博弈、完全合作博弈以及二者的混合博弈。在一般和博弈中可能存在多个纳什均衡点。

我们定义策略 ![](https://www.zhihu.com/equation?tex=%5Cpi_i+%3D+%28%5Cpi_i%28a_1%29%2C%5Ccdots%2C%5Cpi_i%28a_%7Bm_i%7D%29%29) 为智能体 i 的动作集中每个动作的概率集合， ![](https://www.zhihu.com/equation?tex=m_i) 表示可选的动作数量，则值函数 ![](https://www.zhihu.com/equation?tex=V_i) 可以表示为

![](https://www.zhihu.com/equation?tex=V_i%3D%5Cpi_1R_i%5Cpi_2%5ET)

纳什均衡策略 ![](https://www.zhihu.com/equation?tex=%28%5Cpi_1%5E%2A%2C%5Cpi_2%5E%2A%29) 可以表示为

![](https://www.zhihu.com/equation?tex=V_i%28%5Cpi_i%5E%2A%2C%5Cpi_%7B-i%7D%5E%2A%29%5Cge+V_i%28%5Cpi_i%2C%5Cpi_%7B-i%7D%5E%2A%29%2C%5Cforall+%5Cpi_i%5Cin+PD%28A_i%29)

![](https://www.zhihu.com/equation?tex=PD%28A_i%29) 表示第 ![](https://www.zhihu.com/equation?tex=i) 个智能体的策略空间， ![](https://www.zhihu.com/equation?tex=-i) 表示另一个智能体。

如上定义一个两智能体一般和博弈为

![](https://www.zhihu.com/equation?tex=R_1%3D%5Cbegin%7Bbmatrix%7D+r_%7B11%7D+%26+r_%7B12%7D+%5C%5C+r_%7B21%7D+%26+r_%7B22%7D+%5Cend%7Bbmatrix%7D%2C%5Cqquad+R_2%3D%5Cbegin%7Bbmatrix%7D+c_%7B11%7D+%26+c_%7B12%7D+%5C%5C+c_%7B21%7D+%26+c_%7B22%7D+%5Cend%7Bbmatrix%7D)

若满足

![](https://www.zhihu.com/equation?tex=r_%7Blf%7D%5Cgt+r_%7B-lf%7D%2Cc_%7Blf%7D%5Cgt+c_%7Bl-f%7D)

则`$l,f$`为纯策略严格纳什均衡， ![](https://www.zhihu.com/equation?tex=-l%2C-f) 表示除了 ![](https://www.zhihu.com/equation?tex=l%2Cf) 的另一个策略。

## **3\. 线性规划求解双智能体零和博弈**

求解双智能体零和博弈的公式如下

![](https://www.zhihu.com/equation?tex=%5Cmax_%7B%5Cpi_i%7D%5Cmin_%7Ba_%7B-i%7D%5Cin+A_%7B-i%7D%7D%5Csum_%7Ba_i%5Cin+A_i%7DQ_i%5E%2A%28a_i%2Ca_%7B-1%7D%29%5Cpi_i%28a_i%29)

上式的意义为，每个智能体最大化在与对手博弈中最差情况下的期望奖励值。

将博弈写为如下形式

![](https://www.zhihu.com/equation?tex=R_1%3D%5Cbegin%7Bbmatrix%7D+r_%7B11%7D+%26+r_%7B12%7D+%5C%5C+r_%7B21%7D+%26+r_%7B22%7D+%5Cend%7Bbmatrix%7D%2C%5Cqquad+R_2%3D-R_1)

定义 ![](https://www.zhihu.com/equation?tex=p_j%28j%3D1%2C2%29) 表示第一个智能体选择动作 ![](https://www.zhihu.com/equation?tex=j) 的概率， ![](https://www.zhihu.com/equation?tex=q_j%28j%3D1%2C2%29) 表示第二个智能体选择动作 ![](https://www.zhihu.com/equation?tex=j) 的概率。则对于第一个智能体，可以列写如下线性规划

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%26%5Cmax_%7Bp_1%2Cp_2%7D+V_1%5C%5C+r_%7B11%7Dp_1%26%2Br_%7B21%7Dp_2%5Cge+V_1%5C%5C+r_%7B12%7Dp_1%26%2Br_%7B22%7Dp_2%5Cge+V_1%5C%5C+p_1%26%2Bp_2%3D1%5C%5C+p_j%26%5Cge+0%2Cj%3D1%2C2+%5Cend%7Baligned%7D)

同理，可以列出第二个智能体的纳什策略的线性规划

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%26%5Cmax_%7Bq_1%2Cq_2%7D+V_2%5C%5C+-r_%7B11%7Dq_1%26-r_%7B12%7Dq_2%5Cge+V_2%5C%5C+-r_%7B21%7Dq_1%26-r_%7B22%7Dq_2%5Cge+V_2%5C%5C+q_1%26%2Bq_2%3D1%5C%5C+q_j%26%5Cge+0%2Cj%3D1%2C2+%5Cend%7Baligned%7D)

求解上式就可得到纳什均衡策略。

## **4\. 几个博弈概念**

马尔可夫决策过程包含一个智能体与多个状态。矩阵博弈包括多个智能体与一个状态。**随机博弈** (stochastic game / Markov game) 是马尔可夫决策过程与矩阵博弈的结合，具有多个智能体与多个状态，即多智能体强化学习。为更好地理解，引入如下定义

**静态博弈**：static/stateless game 是指没有状态 s，不存在动力学使状态能够转移的博弈。例如一个矩阵博弈。

**阶段博弈**：stage game，是随机博弈的组成成分，状态 s 是固定的，相当于一个状态固定的静态博弈，随机博弈中的 Q 值函数就是该阶段博弈的奖励函数。若干状态的阶段博弈组成一个随机博弈。

**重复博弈**：智能体重复访问同一个状态的阶段博弈，并且在访问同一个状态的阶段博弈的过程中收集其他智能体的信息与奖励值，并学习更好的 Q 值函数与策略。

多智能体强化学习就是一个随机博弈，将每一个状态的阶段博弈的纳什策略组合起来成为一个智能体在动态环境中的策略。并不断与环境交互来更新每一个状态的阶段博弈中的 Q 值函数（博弈奖励）。

对于一个随机博弈可以写为 ![](https://www.zhihu.com/equation?tex=%28n%2CS%2CA_1%2C%5Ccdots%2CA_n%2CTr%2C%5Cgamma%2CR_1%2C%5Ccdots%2CR_n%29) ，其中 n 表示智能体数量，S 表示状态空间， ![](https://www.zhihu.com/equation?tex=A_i) 表示第 i 个智能体的动作空间， ![](https://www.zhihu.com/equation?tex=Tr%3AS%5Ctimes+A_1%5Ctimes%5Ccdots%5Ctimes+A_n%5Ctimes+S%5Cto+%5B0%2C1%5D) 表示状态转移概率， ![](https://www.zhihu.com/equation?tex=R_i%3As%5Ctimes+A_1%5Ctimes%5Ccdots%5Ctimes+A_n%5Ctimes+S%5Cto%5Cmathbb+%7BR%7D) 表示第 i 个智能体在当前状态与联结动作下获得的回报值， ![](https://www.zhihu.com/equation?tex=%5Cgamma) 表示累积奖励折扣系数。随机博弈也具有马尔科夫性，下一个状态与奖励只与当前状态与当前的联结动作有关。

对于一个多智能体强化学习过程，就是找到每一个状态的纳什均衡策略，然后将这些策略联合起来。 ![](https://www.zhihu.com/equation?tex=%5Cpi_i%3AS%5Cto+A_i) 就是一个智能体 i 的策略，在每个状态选出最优的纳什策略。多智能体强化学习最优策略（随机博弈的纳什均衡策略）可以写为 ![](https://www.zhihu.com/equation?tex=%28%5Cpi_1%5E%2A%2C%5Ccdots%2C%5Cpi_n%5E%2A%29) ，且 ![](https://www.zhihu.com/equation?tex=%5Cforall+s%5Cin+S%2Ci%3D1%2C%5Ccdots%2Cn) 满足

![](https://www.zhihu.com/equation?tex=V_i%28s%2C%5Cpi_1%5E%2A%2C%5Ccdots%2C%5Cpi_i%5E%2A%2C%5Ccdots%2C%5Cpi_n%5E%2A%29%5Cge+V_i%28s%2C%5Cpi_1%5E%2A%2C%5Ccdots%2C%5Cpi_i%2C%5Ccdots%2C%5Cpi_n%5E%2A%29%2C%5Cforall+%5Cpi_i%5Cin+%5CPi_i)

![](https://www.zhihu.com/equation?tex=V_i%28s%2C%5Cpi_1%5E%2A%2C%5Ccdots%2C%5Cpi_i%5E%2A%2C%5Ccdots%2C%5Cpi_n%5E%2A%29) 为 ![](https://www.zhihu.com/equation?tex=%5Cgamma) 折扣累积状态值函数，用 ![](https://www.zhihu.com/equation?tex=V_i%5E%2A%28s%29) 简记上式。用 ![](https://www.zhihu.com/equation?tex=Q_i%5E%2A%28s%2Ca_1%2C%5Ccdots%2Ca_n%29) 表示动作状态 ![](https://www.zhihu.com/equation?tex=%5Cgamma) 折扣累积值函数，在每个固定状态 s 的阶段博弈中，就是利用 ![](https://www.zhihu.com/equation?tex=Q_i%5E%2A) 作为博弈的奖励求解纳什均衡策略的。根据强化学习中的 Bellman 公式，可得

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+V_i%5E%2A%28s%29%3D%5Csum_%7Ba_1%2C%5Ccdots%2Ca_n%5Cin+A_1%5Ctimes%5Ccdots%5Ctimes+A_n%7DQ_i%5E%2A%28s%2Ca_1%2C%5Ccdots%2Ca_n%29%5Cpi_1%5E%2A%28s%2Ca_1%29%5Ccdots%5Cpi_n%5E%2A%28s%2Ca_n%29%5C%5C+Q_i%5E%2A%28s%2Ca_1%2C%5Ccdots%2Ca_n%29%3D%5Csum_%7Bs%27%5Cin+S%7DTr%28s%2Ca_1%2C%5Ccdots%2Ca_n%2Cs%27%29%5BR_i%28s%2Ca_1%2C%5Ccdots%2Ca_n%2Cs%27%29%2B%5Cgamma+V_i%5E%2A%28s%27%29%5D+%5Cend%7Baligned%7D)

MARL（多智能体强化学习）的纳什策略可以改写为

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%5Csum_%7Ba_1%2C%5Ccdots%2Ca_n%5Cin+A_1%5Ctimes%5Ccdots%5Ctimes+A_n%7DQ_i%5E%2A%28s%2Ca_1%2C%5Ccdots%2Ca_n%29%5Cpi_1%5E%2A%28s%2Ca_1%29%5Ccdots%5Cpi_i%5E%2A%28s%2Ca_i%29%5Ccdots%5Cpi_n%5E%2A%28s%2Ca_n%29%5Cge%5C%5C+%5Csum_%7Ba_1%2C%5Ccdots%2Ca_n%5Cin+A_1%5Ctimes%5Ccdots%5Ctimes+A_n%7DQ_i%5E%2A%28s%2Ca_1%2C%5Ccdots%2Ca_n%29%5Cpi_1%5E%2A%28s%2Ca_1%29%5Ccdots%5Cpi_i%28s%2Ca_i%29%5Ccdots%5Cpi_n%5E%2A%28s%2Ca_n%29+%5Cend%7Baligned%7D)

根据每个智能体的奖励函数可以对随机博弈进行分类。若智能体的奖励函数相同，则称为完全合作博弈或团队博弈。若智能体的奖励函数逆号，则称为完全竞争博弈或零和博弈。为了求解随机博弈，需要求解每个状态 s 的阶段博弈，每个阶段博弈的奖励值就是 ![](https://www.zhihu.com/equation?tex=Q_i%28s%2C%5Ccdot%29) 。

## **三、随机博弈示例**

定义一个 2*2 的网格博弈，两个智能体分别表示为 ![](https://www.zhihu.com/equation?tex=P_1) , ![](https://www.zhihu.com/equation?tex=P_2) ，1 的初始位置在左下角，2 的初始位置在右上角，每一个智能体都想以最快的方式达到 G 标志的地方。从初始位置开始，每个智能体都有两个动作可以选择。只要有一个智能体达到 G 则游戏结束，达到 G 的智能体获得奖励 10，奖励折扣率为 0.9。虚线表示栏杆，智能体穿过栏杆的概率为 0.5。该随机博弈一共包含 7 个状态。这个博弈的纳什均衡策略是，每个智能体到达邻居位置而不穿过栏杆。

![](https://pic1.zhimg.com/v2-0d7eddce558cbe695e0ef42e2e6459c4_b.jpg)![](https://pic1.zhimg.com/80/v2-0d7eddce558cbe695e0ef42e2e6459c4_hd.jpg)

根据前文公式，我们可以得到如下状态值函数

![](https://www.zhihu.com/equation?tex=V_i%5E%2A%28s_1%29%3DR_i%28s_1%2Cright%2Cleft%2Cs_7%29%2B%5Cgamma+V_i%5E%2A%28s_7%29%3D10%2B0.9%2A0%3D10)

由此我们可以得到动作状态值函数

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+Q_1%5E%2A%28s_1%2Cup%2Cdown%29%26%3D%5Csum_%7Bs%27%3Ds_1%5Csim+s_4%7DTr%28s_1%2Cup%2Cdown%2Cs%27%29%5BR_1%28s_1%2Cup%2Cdown%2Cs%27%29%2B%5Cgamma+V_1%5E%2A%28s%27%29%5D%5C%5C+%26%3D0.25%280%2B0.9V_1%5E%2A%28s_1%29%29%2B0.25%280%2B0.9V_1%5E%2A%28s_2%29%29%2B0.25%280%2B0.9V_1%5E%2A%28s_3%29%29%2B0.25%280%2B0.9V_1%5E%2A%28s_4%29%29%5C%5C+%26%3D7.25+%5Cend%7Baligned%7D)

![](https://pic4.zhimg.com/v2-1e6073342c9555a3bd8b2b2a93820e4b_b.jpg)![](https://pic4.zhimg.com/80/v2-1e6073342c9555a3bd8b2b2a93820e4b_hd.jpg)

求解上述矩阵博弈就可得到多智能体强化学习的策略。

**相关连接：**

[多智能体强化学习入门（一）——基础知识与博弈]

[ECKai：多智能体强化学习入门（一）——基础知识与博弈​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/53474965)

[多智能体强化学习入门（二）——基础算法（MiniMax-Q，NashQ，FFQ，WoLF-PHC）](

[ECKai：多智能体强化学习入门（二）——基础算法（MiniMax-Q，NashQ，FFQ，WoLF-PHC）​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/53563792)

)

[多智能体强化学习入门（三）——矩阵博弈中的分布式学习算法](

[ECKai：多智能体强化学习入门（三）——矩阵博弈中的分布式学习算法​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/53622102)

)

[多智能体强化学习入门（四）——MADDPG 算法](

[ECKai：多智能体强化学习入门（四）——MADDPG 算法​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/53811876)

)

[多智能体强化学习入门（五）——QMIX 算法分析](

[ECKai：多智能体强化学习入门（五）——QMIX 算法分析​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/55003734)

)

附录：

[附：强化学习——DRQN 分析详解](

[ECKai：附：强化学习——DRQN 分析详解​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/54898904)


