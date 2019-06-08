> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 https://zhuanlan.zhihu.com/p/53811876

## 一、引言

本章介绍 OpenAI 2017 发表在 NIPS 上的一篇文章，《Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments》。主要是将 AC 算法进行了一系列改进，使其能够适用于传统 RL 算法无法处理的复杂多智能体场景。

传统 RL 算法面临的一个主要问题是由于每个智能体都是在不断学习改进其策略，因此从每一个智能体的角度看，环境是一个动态不稳定的，这不符合传统 RL 收敛条件。并且在一定程度上，无法通过仅仅改变智能体自身的策略来适应动态不稳定的环境。由于环境的不稳定，将无法直接使用之前的经验回放等 DQN 的关键技巧。policy gradient 算法会由于智能体数量的变多使得本就有的方差大的问题加剧。

MADDPG 算法具有以下三点特征： 1\. 通过学习得到的最优策略，在应用时只利用局部信息就能给出最优动作。 2\. 不需要知道环境的动力学模型以及特殊的通信需求。 3\. 该算法不仅能用于合作环境，也能用于竞争环境。

MADDPG 算法具有以下三点技巧：

1.  集中式训练，分布式执行：训练时采用集中式学习训练 critic 与 actor，使用时 actor 只用知道局部信息就能运行。critic 需要其他智能体的策略信息，本文给了一种估计其他智能体策略的方法，能够只用知道其他智能体的观测与动作。
2.  改进了经验回放记录的数据。为了能够适用于动态环境，每一条信息由 ![](https://www.zhihu.com/equation?tex=%28x%2Cx%27%2C+a_q%2C%5Ccdots%2Ca_n%2Cr_1%2C%5Ccdots%2Cr_n%29) 组成， ![](https://www.zhihu.com/equation?tex=x%3D%28o_1%2C%5Ccdots%2Co_n%29) 表示每个智能体的观测。
3.  利用策略集合效果优化（policy ensemble）：对每个智能体学习多个策略，改进时利用所有策略的整体效果进行优化。以提高算法的稳定性以及鲁棒性。

其实 MADDPG 本质上还是一个 DPG 算法，针对每个智能体训练一个需要全局信息的 Critic 以及一个需要局部信息的 Actor，并且允许每个智能体有自己的奖励函数（reward function），因此可以用于合作任务或对抗任务。并且由于脱胎于 DPG 算法，因此动作空间可以是连续的。

## 二、背景知识

## 1\. DQN

DQN 的思想就是设计一个 ![](https://www.zhihu.com/equation?tex=Q%28s%2Ca%7C%5Ctheta%29) 不断逼近真实的 ![](https://www.zhihu.com/equation?tex=Q%28s%2Ca%29) 函数。其中主要用到了两个技巧：1\. 经验回放。2\. 目标网络。该技巧主要用来打破数据之间联系，因为神经网络对数据的假设是独立同分布，而 MDP 过程的数据前后有关联。打破数据的联系可以更好地拟合 ![](https://www.zhihu.com/equation?tex=Q%28s%2Ca%29) 函数。其代价函数为：

![](https://www.zhihu.com/equation?tex=L%28%5Ctheta%29+%3D+E_%7Bs%2Ca%2Cr%2Cs%27%7D%5B%28Q%28s%2Ca%7C%5Ctheta%29-y%29%5E2%5D%2C%5Cqquad+%5Crm%7Bwhere%7D%5C+y%3Dr%2B%5Cgamma+max_%7Ba%27%7D%5Coverline+Q%28s%27%2Ca%27%7C%5Coverline+%5Ctheta%29)

其中 ![](https://www.zhihu.com/equation?tex=%5Coverline+Q%28s%27%2Ca%27%7C%5Coverline+%5Ctheta%29) 表示目标网络，其参数更新与 ![](https://www.zhihu.com/equation?tex=%5Ctheta) 不同步（滞后）。具体可以参看[值函数强化学习 - DQN、DDQN 和 Dueling DQN 算法公式推导分析](https://link.zhihu.com/?target=https%3A//blog.csdn.net/weixin_37895339/article/details/84822668)。

## 2\. SPG（stochastic policy gradient）

SPG 算法不采用拟合 Q 函数的方式，而是直接优化累积回报来获得使回报最大的策略。假定参数化的策略为 ![](https://www.zhihu.com/equation?tex=%5Cpi_%5Ctheta%28a%7Cs%29) ，累积回报为 ![](https://www.zhihu.com/equation?tex=J%28%5Ctheta%29%3DE_%7Bs%5Csim+%5Crho%5E%7B%5Cpi%7D%2Ca%5Csim+%5Cpi_%5Ctheta%7D%5B%5Csum_%7Bt%3D0%7D%5E%7B%5Cinfty%7D%5Cgamma%5Et+r_t%5D) 。为了使 ![](https://www.zhihu.com/equation?tex=J%28%5Ctheta%29) 最大化，直接对策略参数求导得到策略更新梯度：

![](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Ctheta%7D+J%28%5Ctheta%29%3DE_%7Bs%5Csim+%5Crho%5E%7B%5Cpi%7D%2Ca%5Csim+%5Cpi_%5Ctheta%7D%5B%5Cnabla_%7B%5Ctheta%7D%5Clog%5Cpi_%5Ctheta%28a%7Cs%29Q%5E%5Cpi%28s%2Ca%29%5D)

AC 算法也可以由此推出，如果按照 DQN 的方法拟合一个 ![](https://www.zhihu.com/equation?tex=Q%28s%2Ca%7C%5Ctheta%29) 函数，则这个参数化的 ![](https://www.zhihu.com/equation?tex=Q%28s%2Ca%7C%5Ctheta%29) 函数被称为 Critic， ![](https://www.zhihu.com/equation?tex=%5Cpi_%5Ctheta%28a%7Cs%29) 被称为 Actor。具体可以参看，[随机策略梯度算法（stochastic-policy-gradient）](https://link.zhihu.com/?target=https%3A//blog.csdn.net/weixin_37895339/article/details/84792752)。

## 3\. DPG

上述两种算法都是针对随机策略， ![](https://www.zhihu.com/equation?tex=%5Cpi_%5Ctheta%28a%7Cs%29) 是一个在状态 s 对于各个动作 a 的条件概率分布。DPG 针对确定性策略， ![](https://www.zhihu.com/equation?tex=%5Cmu_%5Ctheta%28s%29%3AS%5Cto+A) 是一个状态空间到动作空间的映射。其思想与 SPG 相同，得到策略梯度公式为

![](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Ctheta%7D+J%28%5Ctheta%29%3DE_%7Bs%5Csim+%5Cbeta%7D%5B%5Cnabla_%7B%5Ctheta%7D%5Cmu_%5Ctheta%28s%29%5Cnabla_a+Q%5E%5Cmu%28s%2Ca%29%7C_%7Ba%3D%5Cmu_%5Ctheta%28s%29%7D%5D)

DPG 可以是使用 AC 的方法来估计一个 Q 函数，DDPG 就是借用了 DQN 经验回放与目标网络的技巧，具体可以参看，[确定性策略强化学习 - DPG&DDPG 算法推导及分析](https://link.zhihu.com/?target=https%3A//blog.csdn.net/weixin_37895339/article/details/84881872)。

## 三、MADDPG

下面我们一次介绍 MADDPG 技巧。

## 1\. 多智能体 AC 设计

MADDPG 集中式的学习，分布式的应用。因此我们允许使用一些额外的信息（全局信息）进行学习，只要在应用的时候使用局部信息进行决策就行。这点就是 Q-learning 的一个不足之处，Q-learning 在学习与应用时必须采用相同的信息。所以这里 MADDPG 对传统的 AC 算法进行了一个改进，Critic 扩展为可以利用其他智能体的策略进行学习，这点的进一步改进就是每个智能体对其他智能体的策略进行一个函数逼近。

我们用 ![](https://www.zhihu.com/equation?tex=%5Ctheta%3D%5B%5Ctheta_1%2C%5Ccdots%2C%5Ctheta_n%5D) 表示 n 个智能体策略的参数， ![](https://www.zhihu.com/equation?tex=%5Cpi%3D%5B%5Cpi_1%2C%5Ccdot%2C%5Cpi_n%5D) 表示 n 个智能体的策略。针对第 i 个智能体的累积期望奖励 ![](https://www.zhihu.com/equation?tex=J%28%5Ctheta_i%29%3DE_%7Bs%5Csim+%5Crho%5E%7B%5Cpi%7D%2Ca_i%5Csim+%5Cpi_%7B%5Ctheta_i%7D%7D%5B%5Csum_%7Bt%3D0%7D%5E%7B%5Cinfty%7D%5Cgamma%5Et+r_%7Bi%2Ct%7D%5D) ，针对随机策略，求策略梯度为：

![](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Ctheta_i%7DJ%28%5Ctheta_i%29%3DE_%7Bs%5Csim+%5Crho%5E%5Cpi%2Ca_i%5Csim+%5Cpi_i%7D%5B%5Cnabla_%7B%5Ctheta_i%7D%5Clog%5Cpi_i%28a_i%7Co_i%29Q_i%5E%7B%5Cpi%7D%28x%2Ca_1%2C%5Ccdots%2Ca_n%29%5D)

其中 ![](https://www.zhihu.com/equation?tex=o_i) 表示第 i 个智能体的观测， ![](https://www.zhihu.com/equation?tex=x%3D%5Bo_1%2C%5Ccdots%2Co_n%5D) 表示观测向量，即状态。 ![](https://www.zhihu.com/equation?tex=Q_i%5E%7B%5Cpi%7D%28x%2Ca_1%2C%5Ccdots%2Ca_n%29) 表示第 i 个智能体集中式的状态 - 动作函数。由于是每个智能体独立学习自己的 ![](https://www.zhihu.com/equation?tex=Q_i%5E%5Cpi) 函数，因此每个智能体可以有不同的奖励函数（reward function），因此可以完成合作或竞争任务。

上述为随机策略梯度算法，下面我们拓展到确定性策略 ![](https://www.zhihu.com/equation?tex=%5Cmu_%7B%5Ctheta_i%7D) ，梯度公式为

![](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Ctheta_i%7DJ%28%5Cmu_i%29%3DE_%7Bx%2Ca%5Csim+D%7D%5B%5Cnabla_%7B%5Ctheta_i%7D%5Cmu_i%28a_i%7Co_i%29%5Cnabla_%7Ba_i%7DQ_i%5E%5Cmu%28x%2Ca_1%2C%5Ccdots%2Ca_n%29%7C_%7Ba_i%3D%5Cmu_i%28o_i%29%7D%5D)

由以上两个梯度公式可以看出该算法与 SPG 与 DPG 十分类似，就像是将单体直接扩展到多体。但其实 ![](https://www.zhihu.com/equation?tex=Q_i%5E%5Cmu) 是一个非常厉害的技巧，针对每个智能体建立值函数，极大的解决了传统 RL 算法在 Multi-agent 领域的不足。D 是一个经验存储（experience replay buffer），元素组成为 ![](https://www.zhihu.com/equation?tex=%28x%2Cx%27%2Ca_1%2C%5Ccdots%2Ca_n%2Cr_1%2C%5Ccdots%2Cr_n%29) 。集中式的 critic 的更新方法借鉴了 DQN 中 TD 与目标网络思想

![](https://www.zhihu.com/equation?tex=L%28%5Ctheta_i%29%3DE_%7Bx%2Ca%2Cr%2Cx%27%7D%5B%28Q_i%5E%5Cmu%28x%2Ca_1%2C%5Ccdots%2Ca_n%29-y%29%5E2%5D%2C%5Cqquad+%5Crm%7Bwhere%7D%5C+y%3Dr_i%2B%5Cgamma+%5Coverline+Q_i%5E%7B%5Cmu%27%7D%28x%27%2Ca_1%27%2C%5Ccdots%2Ca_n%27%29%7C_%7Ba_j%27%3D%5Cmu_j%27%28o_j%29%7D%5Cqquad+%281%29)

![](https://www.zhihu.com/equation?tex=%5Coverline+Q_i%5E%7B%5Cmu%27%7D) 表示目标网络， ![](https://www.zhihu.com/equation?tex=%5Cmu%27%3D%5B%5Cmu_1%27%2C%5Ccdots%2C%5Cmu_n%27%5D) 为目标策略具有滞后更新的参数 ![](https://www.zhihu.com/equation?tex=%5Ctheta_j%27) 。其他智能体的策略可以采用拟合逼近的方式得到，而不需要通信交互。

如上可以看出 critic 借用了全局信息学习，actor 只是用了局部观测信息。MADDPG 的一个启发就是，如果我们知道所有的智能体的动作，那么环境就是稳定的，就算策略在不断更新环境也是恒定的，因为模型动力学使稳定的 ![](https://www.zhihu.com/equation?tex=P%28s%27%7Cs%2Ca_1%2C%5Ccdots%2Ca_n%2C%5Cpi_1%2C%5Ccdots%2C%5Cpi_n%29%3DP%28s%27%7Cs%2Ca_1%2C%5Ccdots%2Ca_n%29%3DP%28s%27%7Cs%2Ca_1%2C%5Ccdots%2Ca_n%2C%5Cpi_1%27%2C%5Ccdots%2C%5Cpi_n%27%29) 。

## 2\. 估计其他智能体策略

在 (1) 式中，我们用到了其他智能体的策略，这需要不断的通信来获取，但是也可以放宽这个条件，通过对其他智能体的策略进行估计来实现。每个智能体维护 n-1 个策略逼近函数 ![](https://www.zhihu.com/equation?tex=%5Chat+%5Cmu_%7B%5Cphi_i%5Ej%7D) 表示第 i 个智能体对第 j 个智能体策略 ![](https://www.zhihu.com/equation?tex=%5Cmu_j) 的函数逼近。其逼近代价为对数代价函数，并且加上策略的熵，其代价函数可以写为

![](https://www.zhihu.com/equation?tex=L%28%5Cphi_i%5Ej%29%3D-E_%7Bo_j%2Ca_j%7D%5B%5Clog+%5Chat+%5Cmu_%7B%5Cphi_i%5Ej%7D%28a_j%7Co_j%29%2B%5Clambda+H%28%5Chat+%5Cmu_%7B+%5Cphi_i%5Ej%7D%29%5D)

只要最小化上述代价函数，就能得到其他智能体策略的逼近。因此可以替换 (1) 式中的 y。

![](https://www.zhihu.com/equation?tex=y%3Dr_i%2B%5Cgamma+%5Coverline+Q_i%5E%7B%5Cmu%27%7D%28x%27%2C%5Chat+%5Cmu_%7B%5Cphi_i%5Ej%7D%27%5E1%28o_1%29%2C%5Ccdots%2C%5Chat+%5Cmu_%7B%5Cphi_i%5Ej%7D%27%5En%28o_n%29%29)

在更新 ![](https://www.zhihu.com/equation?tex=Q_i%5E%5Cmu) 之前，利用经验回放的一个采样 batch 更新 ![](https://www.zhihu.com/equation?tex=%5Chat+%5Cmu_%7B%5Cphi_i%5Ej%7D) 。

## 3\. 策略集合优化（policies ensemble）

这个技巧也是本文的一个亮点。多智能体强化学习一个顽固的问题是由于每个智能体的策略都在更新迭代导致环境针对一个特定的智能体是动态不稳定的。这种情况在竞争任务下尤其严重，经常会出现一个智能体针对其竞争对手过拟合出一个强策略。但是这个强策略是非常脆弱的，也是我们希望得到的，因为随着竞争对手策略的更新改变，这个强策略很难去适应新的对手策略。

为了能更好的应对上述情况，MADDPG 提出了一种策略集合的思想，第 i 个智能体的策略 ![](https://www.zhihu.com/equation?tex=%5Cmu_i) 由一个具有 K 个子策略的集合构成，在每一个训练 episode 中只是用一个子策略 ![](https://www.zhihu.com/equation?tex=%5Cmu_%7B%5Ctheta%5E%7B%28k%29%7D_i%7D) （简写为 ![](https://www.zhihu.com/equation?tex=%5Cmu_i%5E%7B%28k%29%7D) ）。对每一个智能体，我们最大化其策略集合的整体奖励 ![](https://www.zhihu.com/equation?tex=J_e%28%5Cmu_i%29%3DE_%7Bk%5Csim+%7B%5Crm+unif%281%2CK%29%7D%2Cs%5Csim+%5Crho%5E%5Cmu%2Ca%5Csim%5Cmu_i%5E%7B%28k%29%7D%7D%5B%5Csum_%7Bt%3D0%7D%5E%5Cinfty+%5Cgamma%5Et+r_%7Bi%2Ct%7D%5D) 。并且我们为每一个子策略 k 构建一个记忆存储 ![](https://www.zhihu.com/equation?tex=D_i%5E%7B%28k%29%7D) 。我们优化策略集合的整体效果，因此针对每一个子策略的更新梯度为

![](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Ctheta_i%5E%7B%28k%29%7D%7DJ_e%28%5Cmu_i%29%3D%5Cfrac%7B1%7D%7BK%7DE_%7Bx%2Ca%5Csim+D_i%5E%7B%28k%29%7D%7D%5B%5Cnabla_%7B%5Ctheta_i%5E%7B%28k%29%7D%7D%5Cmu_i%5E%7B%28k%29%7D%28a_i%7Co_i%29%5Cnabla_%7Ba_i%7DQ%5E%7B%5Cmu_i%7D%28x%2Ca_1%2C%5Ccdots%2Ca_n%29%7C_%7Ba_i%3D%5Cmu_i%5E%7B%28k%29%7D%28o_i%29%7D%5D)

以上就是 MADDPG 所有的内容，仿真效果也很好的证明了 MADDPG 在多智能体系统中的有效性。 [MADDPG](https://link.zhihu.com/?target=https%3A//github.com/openai/maddpg) 是 OpenAI 给的仿真代码。

