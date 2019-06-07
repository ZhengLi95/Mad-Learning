> 
>
> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 https://zhuanlan.zhihu.com/p/53563792

## 一、引言

上一篇文章介绍，一个随机博弈可以看成是一个多智能体强化学习过程。其实这两个概念不能完全等价，随机博弈中假定每个状态的奖励矩阵是已知的，不需要学习。而多智能体强化学习则是通过与环境的不断交互来学习每个状态的奖励值函数，再通过这些奖励值函数来学习得到最优纳什策略。通常情况下，模型的转移概率以及奖励函数为止，因此需要利用到 Q-learning 中的方法来不断逼近状态值函数或动作 - 状态值函数。

在多智能体强化学习算法中，两个主要的技术指标为合理性与收敛性。

**合理性**（rationality）是指在对手使用一个恒定策略的情况下，当前智能体能够学习并收敛到一个相对于对手策略的最优策略。

**收敛性**（convergence）是指在其他智能体也使用学习算法时，当前智能体能够学习并收敛到一个稳定的策略。通常情况下，收敛性针对系统中的所有的智能体使用相同的学习算法。

针对应用来分，多智能体强化学习算法可分为零和博弈算法与一般和博弈算法。

本文主要介绍四种多智能体强化学习算法，主要介绍每种算法的应用特性与应用公式，具体的收敛性证明后面每个算法单开一章讲。

## 二、Minimax-Q 算法

Minimax-Q 算法应用于**两个玩家的零和随机博弈**中。Minimax-Q 中的 Minimax 指的是使用上一篇文章中的 minimax 方法构建线性规划来求解每个特定状态 s 的阶段博弈的纳什均衡策略。Q 指的是借用 Q-learning 中的 TD 方法来迭代学习状态值函数或动作 - 状态值函数。

在两玩家零和随机博弈中，给定一个状态 s，则定义第 i 个智能体的状态值函数为

![](https://www.zhihu.com/equation?tex=V_i%5E%2A%28s%29%3D%5Cmax_%7B%5Cpi_i%28s%2C%5Ccdot%29%7D%5Cmin_%7Ba_%7B-i%7D%5Cin+A_%7B-i%7D%7D%5Csum_%7Ba_i%5Cin+A_i%7DQ_i%5E%2A%28s%2Ca_i%2Ca_%7B-i%7D%29%5Cpi_i%28s%2Ca_i%29%2Ci%3D1%2C2)

![](https://www.zhihu.com/equation?tex=-i) 表示智能体 ![](https://www.zhihu.com/equation?tex=i) 的对手。 ![](https://www.zhihu.com/equation?tex=Q_i%5E%2A%28s%2Ca_i%2Ca_%7B-i%7D%29) 为联结动作状态值函数，若 ![](https://www.zhihu.com/equation?tex=Q_i%5E%2A%28%5Ccdot%29) 已知，我们可以直接用线性规划求解出状态 s 处的纳什均衡策略。但是在多智能体强化学习中， ![](https://www.zhihu.com/equation?tex=Q_i%5E%2A%28%5Ccdot%29) 是未知的，所以借用 Q-learning 中优秀的 TD 来更新逼近真实的 ![](https://www.zhihu.com/equation?tex=Q_i%28s%2Ca_i%2Ca_%7B-1%7D%29) 值。整个算法流程如下

![](https://pic2.zhimg.com/v2-e907b291d3ec2e6dd62c96a86b4fd171_b.jpg)

![](https://pic2.zhimg.com/v2-e907b291d3ec2e6dd62c96a86b4fd171_r.jpg)

理想情况，如果算法能够对每一个状态 - 动作对访问无限次，那么该算法能够收敛到纳什均衡策略。但是在上述算法中存在几个缺点：

1.  在第 5 步中需要不断求解一个线性规划，这将造成学习速度的降低，增加计算时间。
2.  为了求解第 5 步，智能体 ![](https://www.zhihu.com/equation?tex=i) 需要知道所有智能体的动作空间，这个在分布式系统中将无法满足。
3.  只满足收敛性，不满足合理性。Minimax-Q 算法能够找到多智能体强化学习的纳什均衡策略，但是假设对手使用的不是纳什均衡策略，而是一个较差的策略，则当前智能体并不能根据对手的策略学习到一个更优的策略。该算法无法让智能体根据对手的策略来调节优化自己的策略，而只能找到随机博弈的纳什均衡策略。这是由于 Minimax-Q 算法是一个对手独立算法（opponent-independent algorithm），不论对手策略是怎么样的，都收敛到该博弈的纳什均衡策略。就算对手采用一个非常弱的策略，当前智能体也不能学习到一个比纳什均衡策略更好的策略。

## 三、Nash Q-Learning 算法

Nash Q-Learning 算法是将 Minimax-Q 算法从零和博弈扩展到**多人一般和博弈**的算法。在 Minimax-Q 算法中需要通过 Minimax 线性规划求解阶段博弈的纳什均衡点，拓展到 Nash Q-Learning 算法就是使用二次规划求解纳什均衡点，具体求解方法后面单独开一章讲解。Nash Q-Learning 算法在合作性均衡或对抗性均衡的环境中能够收敛到纳什均衡点，其收敛性条件是，在每一个状态 s 的阶段博弈中，都能够找到一个全局最优点或者鞍点，只有满足这个条件，Nash Q-Learning 算法才能够收敛。与 Minimax-Q 算法相同，Nash Q-Learning 算法求解二次规划的过程也非常耗时，降低了算法的学习速度。其算法流程如下：

![](https://pic4.zhimg.com/v2-5d50dc1f2ad874c22f10d5e796f64347_b.jpg)

![](https://pic4.zhimg.com/80/v2-5d50dc1f2ad874c22f10d5e796f64347_hd.jpg)

该算法需要观测其他所有智能体的动作 ![](https://www.zhihu.com/equation?tex=a_i) 与奖励值 ![](https://www.zhihu.com/equation?tex=r_i) 。并且与 Minimax-Q 算法一样，只满足收敛性，不满足合理性。只能收敛到纳什均衡策略，不能根据其他智能体的策略来优化调剂自身的策略。

## 四、Friend-or-Foe Q-Learning 算法

Friend-or-Foe Q-Learning（FFQ）算法也是从 Minimax-Q 算法拓展而来。为了能够处理一般和博弈，FFQ 算法对一个智能体 i，将其他所有智能体分为两组，一组为 i 的 friend 帮助 i 一起最大化其奖励回报，另一组为 i 的 foe 对抗 i 并降低 i 的奖励回报，因此对每个智能体而言都有两组。这样一个 **n 智能体的一般和博弈**就转化为了一个两智能体的零和博弈。其纳什均衡策略求解方法如下所示

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+V_i%28s%29%3D%5Cmax_%7B%5Cpi_1%28s%2C%5Ccdot%29%2C%5Ccdots%2C%5Cpi_%7Bn_1%7D%28s%2C%5Ccdot%29%7D%5Cmin_%7Bo_1%2C%5Ccdots%2C+o_%7Bn_2%7D%5Cin+O_1%5Ctimes%5Ccdots%5Ctimes+O_%7Bn_2%7D%7D%5Csum_%7Ba_1%2C%5Ccdots%2Ca_%7Bn_1%7D%5Cin+A_1%5Ctimes%5Ccdots%5Ctimes+A_%7Bn_1%7D%7D%5C%5C+Q_i%28s%2Ca_1%2C%5Ccdots%2Ca_%7Bn_1%7D%2Co_1%2C%5Ccdots%2C+o_%7Bn_2%7D%29%5Cpi_1%28s%2Ca_1%29%2C%5Ccdots%2C%5Cpi_%7Bn_1%7D%28s%2Ca_%7Bn_1%7D%29+%5Cend%7Baligned%7D)

算法流程如下：

![](https://pic1.zhimg.com/v2-2f4cff250568ca25853f1bc8e23e3b54_b.jpg)

![](https://pic1.zhimg.com/80/v2-2f4cff250568ca25853f1bc8e23e3b54_hd.jpg)

有一种利用 Minimax-Q 算法进行多人博弈方法为，两队零和博弈，将所有智能体分成两个小组进行零和博弈。两队零和博弈中每一组有一个 leader 才控制这一队智能体的所有策略，获取的奖励值也是这一个小组的整体奖励值。

FFQ 算法没有 team learder，每个人选择自己动作学习自己的策略获得自己的奖励值，但是为了更新 ![](https://www.zhihu.com/equation?tex=Q) 值，每个智能体需要在每一步观测其他所有 friend 与 foe 的执行动作。

FFQ 与 Minimax-Q 算法一样都需要利用线性规划，因此算法整体学习速度会变慢。

## 五、WoLF Policy Hill-Climbing 算法

上述的三种方法都需要在学习过程中维护 Q 函数，假设动作空间 ![](https://www.zhihu.com/equation?tex=A_i) 与状态空间 ![](https://www.zhihu.com/equation?tex=S) 都是离散，假设每个智能体的动作空间相同，则对于每一个智能体都需要有一个 ![](https://www.zhihu.com/equation?tex=%7CS%7C%5Ccdot%7CA%7C%5En) 大小的空间来存储 Q 值，因此上述三种方法所需空间非常大。

为了解决上述问题，我们期望每个智能体只用知道自己的动作来维护 Q 值函数，这样空间就降到了 ![](https://www.zhihu.com/equation?tex=%7CS%7C%5Ccdot%7CA%7C) 。WoLF-PHC 就是这样的算法，每个智能体只用保存自己的动作来完成学习任务。WoLF-PHC 是将 “Win or Learn Fast” 规则与 policy hill-climbing 算法结合。

**WolF** 是指，当智能体做的比期望值好的时候小心缓慢的调整参数，当智能体做的比期望值差的时候，加快步伐调整参数。

**PHC** 是一种单智能体在稳定环境下的一种学习算法。该算法的核心就是通常强化学习的思想，增大能够得到最大累积期望的动作的选取概率。该算法具有合理性，能够收敛到最优策略。其算法流程如下

![](https://pic4.zhimg.com/v2-8420ec197cb0516076725645a1359ab3_b.jpg)

![](https://pic4.zhimg.com/80/v2-8420ec197cb0516076725645a1359ab3_hd.jpg)

为了将 PHC 应用于动态环境中，将 WoLF 与 PHC 算法结合，使得智能体获得的奖励在比预期差时，能够快速调整适应其他智能体策略变化，当比预期好时谨慎学习，给其他智能体适应策略变化的时间。并且 WoLF-PHC 算法能够收敛到纳什均衡策略，并且具备合理性，当其他智能体采用某个固定策略使，其也能收敛到一个目前状况下的最优策略而不是收敛到一个可能效果不好的纳什均衡策略处。在 WoLF-PHC 算法中，使用一个可变的学习速率 ![](https://www.zhihu.com/equation?tex=%5Cdelta) 来实现 WoLF 效果，当策略效果较差时使用 ![](https://www.zhihu.com/equation?tex=%5Cdelta_l) ，策略效果较好时使用 ![](https://www.zhihu.com/equation?tex=%5Cdelta_w) ，并且满足 ![](https://www.zhihu.com/equation?tex=%5Cdelta_l%5Cgt%5Cdelta_w) 。还有一个优势是，WoLF-PHC 算法不用观测其他智能体的策略、动作及奖励值，需要更少的空间去记录 Q 值，并且 WoLF-PHC 算法是通过 PHC 算法进行学习改进策略的，所以不需要使用线性规划或者二次规划求解纳什均衡，算法速度得到了提高。虽然 WoLF-PHC 算法在实际应用中取得了非常好的效果，并且能够收敛到最优策略。但是其收敛性在理论上一直没有得到证明。其算法流程如下所示：

![](https://pic2.zhimg.com/v2-a549b9cfd895898e7a4cc74d7432ce55_b.jpg)

![](https://pic2.zhimg.com/80/v2-a549b9cfd895898e7a4cc74d7432ce55_hd.jpg)

最后：还有很多基础算法后序慢慢补，并且经典的收敛性证明后序也会补上。

**相关连接：**

[ECKai：多智能体强化学习入门（一）——基础知识与博弈​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/53474965)[ECKai：多智能体强化学习入门（二）——基础算法（MiniMax-Q，NashQ，FFQ，WoLF-PHC）​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/53563792)[ECKai：多智能体强化学习入门（三）——矩阵博弈中的分布式学习算法​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/53622102)[ECKai：多智能体强化学习入门（四）——MADDPG 算法​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/53811876)[ECKai：多智能体强化学习入门（五）——QMIX 算法分析​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/55003734)

附录：

[ECKai：附：强化学习——DRQN 分析详解​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/54898904)
写下你的评论...

多智能体强化学习学到的策略是纳什均衡策略吗

有的算法在特定环境下可以，有可能还要考虑同伴的策略。这个问题的答案需要考虑到具体的算法，环境，对手的算法等因素

那强化学习和博弈是个什么关系呢

因为多 Agent 考虑的就是一个的博弈的游戏，单纯的单 Agent 的 RL 和博弈没什么关系。

可否提供一下对应论文的链接呢？

![](https://pic2.zhimg.com/v2-90359a720808ff45062287127cfa1039_r.gif)

您好，可以参考这篇综述性文章。

[1] Lucian Buşoniu, Robert Babuška, Schutter B D . Multi-agent Reinforcement Learning: An Overview[M] Innovations in Multi-Agent Systems and Applications - 1\. Springer Berlin Heidelberg, 2010.

感谢！

可以讲一下相关均衡吗