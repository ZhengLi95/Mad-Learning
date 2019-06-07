> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 https://zhuanlan.zhihu.com/p/56049023

## 一、引言

[Mean Field Multi-Agent Reinforcement Learning（MFMARL）](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1802.05438v4) 是伦敦大学学院（UCL）计算机科学系教授汪军提出的一个多智能体强化学习算法。主要致力于极大规模的多智能体强化学习问题，解决大规模智能体之间的交互及计算困难。由于多智能体强化学习问题不仅有环境交互问题，还有智能体之间的动态影响，因此为了得到最优策略，每个智能体都需要考察其他智能体的动作及状态得到联合动作值函数。由于状态空间跟动作空间随着智能体数量的增多而迅速扩大，这给计算以及探索带来了非常大的困难。

MFMARL 算法借用了平均场论（Mean Field Theory，MFT）的思想，其对多智能体系统给出了一个近似假设：对某个智能体，其他所有智能体对其产生的作用可以用一个均值替代。这样就就将一个智能体与其邻居智能体之间的相互作用简化为两个智能体之间的相互作用（该智能体与其所有邻居的均值）。这样极大地简化了智能体数量带来的模型空间的增大。应用平均场论后，学习在两个智能体之间是相互促进的：单个智能体的最优策略的学习是基于智能体群体的动态；同时，集体的动态也根据个体的策略进行更新。

下面具体介绍算法细节，其结合平均场论得出两个主要算法 MF-Q 与 MF-AC，是对 Q-learning 以及 AC 算法的改进，并且在理论上给出了收敛性证明，能够收敛到纳什均衡点。该算法可以应用于竞争环境或合作环境，并且每个智能体不知道环境的模型以及奖励模型，但是能够观察邻居智能体的动作及奖励。每个智能体有自己的奖励值。

## 二、预备知识

为了帮助更好的理解 MFMARL 算法，这里简要回顾之前文章的内容。

## 1\. 随机博弈

随机博弈的具体内容可以看[多智能体强化学习入门（一）——基础知识与博弈](https://zhuanlan.zhihu.com/p/53474965)。 将一个 n 智能体的随机博弈过程写为元组 ![](https://www.zhihu.com/equation?tex=%28S%2CA_1%2C%5Ccdots%2CA_n%2Cr_1%2C%5Ccdots%2Cr_n%2Cp%2C%5Cgamma%29) ，其中 ![](https://www.zhihu.com/equation?tex=S) 为全局的状态信息， ![](https://www.zhihu.com/equation?tex=A_i) 为智能体 ![](https://www.zhihu.com/equation?tex=i) 的动作空间， ![](https://www.zhihu.com/equation?tex=r_i) 为智能体 i 的奖励值， ![](https://www.zhihu.com/equation?tex=p) 为环境的状态转移概率， ![](https://www.zhihu.com/equation?tex=%5Cgamma) 为折扣系数。智能体 ![](https://www.zhihu.com/equation?tex=i) 的策略定义为 ![](https://www.zhihu.com/equation?tex=%5Cpi_i%3AS%5Cto+%5COmega%28A_i%29) ，从状态到智能体动作空间概率分布的映射， ![](https://www.zhihu.com/equation?tex=%5COmega%28A_i%29) 表示智能体 i 动作空间概率分布的所有可能集合。 ![](https://www.zhihu.com/equation?tex=%5Cpi%3D%5B%5Cpi_1%2C%5Ccdots%2C%5Cpi_n%5D) 表示所有智能体的联合策略。通常假定 ![](https://www.zhihu.com/equation?tex=%5Cpi) 是时不变的。假定初始状态 ![](https://www.zhihu.com/equation?tex=s) ，第 ![](https://www.zhihu.com/equation?tex=j) 个智能体在策略 ![](https://www.zhihu.com/equation?tex=%5Cpi) 下的累积折扣值函数为

![](https://www.zhihu.com/equation?tex=v_%5Cpi%5Ej%28s%29%3Dv%5Ej%28s%3B%5Cpi%29%3D%5Csum_%7Bt%3D0%7D%5E%5Cinfty%5Cgamma%5EtE_%7B%5Cpi%7D%5Br_t%5Ej%7Cs_0%3Ds%2C%5Cpi%5D%5Cqquad+%281%29)

根据 (1) 式，结合 Bellman 方程可以将动作值函数 ![](https://www.zhihu.com/equation?tex=Q_%5Cpi%5Ej%3AS%5Ctimes+A_1%5Ctimes+%5Ccdots%5Ctimes+A_n%5Cto+R) 写为

![](https://www.zhihu.com/equation?tex=Q_%5Cpi%5Ej%28s%2Ca%29%3Dr_j%28s%2Ca%29%2B%5Cgamma+E_%7Bs%27%5Csim+p%7D%5Bv_%5Cpi%5Ej%28s%27%29%5D%5Cqquad+%282%29)

同时值函数 ![](https://www.zhihu.com/equation?tex=v_%5Cpi%5Ej%28s%29) 也可以写为

![](https://www.zhihu.com/equation?tex=v_%5Cpi%5Ej%28s%29%3DE_%7Ba%5Csim+%5Cpi%28s%29%7D%5BQ_%5Cpi%5Ej%28s%2Ca%29%5D)

从 (2) 中可以看出，每次动作值函数考虑的是整个联合状态空间 ![](https://www.zhihu.com/equation?tex=s) 及联合动作空间 ![](https://www.zhihu.com/equation?tex=a%3D%5Ba_1%2C%5Ccdots%2Ca_n%5D) 。算法假设每个智能体不知道环境的动力学模型以及其他智能体的奖励函数，但是能够通过观测或者交互得到其他智能体的奖励值以及上一步的动作。

## 2\. Nash-Q 算法

Nash-Q 算法的具体内容可以看[多智能体强化学习入门（二）——基础算法（MiniMax-Q，NashQ，FFQ，WoLF-PHC）](https://zhuanlan.zhihu.com/p/53563792)中的内容。 在随机博弈的每一个阶段博弈中，即每一个状态 s 处的博弈中，我们为了找到一个纳什平衡点来使每一个智能体的 ![](https://www.zhihu.com/equation?tex=v_%5Cpi%5Ej%28s%29) 在联合策略 ![](https://www.zhihu.com/equation?tex=%5Cpi) 下取得最优值。在状态 s 处，纳什均衡策略可以表示为 ![](https://www.zhihu.com/equation?tex=%5Cpi%5E%2A%3D%5B%5Cpi_1%5E%2A%2C%5Ccdots%2C%5Cpi_n%5E%2A%5D) ，使得对于任意 ![](https://www.zhihu.com/equation?tex=j%5Cin%5C%7B1%2C2%2C%5Ccdots%2Cn%5C%7D) 值函数满足

![](https://www.zhihu.com/equation?tex=v_j%28s%3B%5Cpi%5E%2A%29%3Dv_j%28s%3B%5Cpi_j%5E%2A%2C%5Cpi_%7B-j%7D%5E%2A%29%5Cge+v_j%28s%3B%5Cpi_j%2C%5Cpi_%7B-j%7D%5E%2A%29)

其中 ![](https://www.zhihu.com/equation?tex=%5Cpi_%7B-j%7D%3D%5B%5Cpi_1%2C%5Ccdots%2C%5Cpi_%7Bj-1%7D%2C%5Cpi_%7Bj%2B1%7D%2C%5Ccdots%2C%5Cpi_n%5D) 表示除了 j 的联合策略。在纳什平衡点处，每个智能体的策略是在其他智能体策略下的最优策略。Nash-Q 的核心就是在每个状态 s 处的阶段博弈中找到纳什均衡，这样可以保证策略在特定情况下收敛。具体内容见[多智能体强化学习入门（二）——基础算法（MiniMax-Q，NashQ，FFQ，WoLF-PHC）](https://zhuanlan.zhihu.com/p/53563792)。其算法核心公式为

![](https://www.zhihu.com/equation?tex=%5Cmathcal+H%5E%7BNash%7DQ%28s%2Ca%29%3DE_%7Bs%27%5Csim+p%7D%5Br%28s%2Ca%29%2B%5Cgamma+v%5E%7BNash%7D%28s%27%29%5D%5Cqquad+%283%29)

其中 ![](https://www.zhihu.com/equation?tex=Q%3D%5BQ_1%2C%5Ccdots%2CQ_n%5D) ， ![](https://www.zhihu.com/equation?tex=r%3D%5Br_1%2C%5Ccdots%2Cr_n%5D) ，最终通过上式迭代，Q 函数将会收敛到纳什均衡点。

## 三、Mean Field MARL

MFMARL 算法主要解决的是集中式多智能体强化学习中，联合动作 ![](https://www.zhihu.com/equation?tex=a) 的维度随智能体数量 n 的增多极速扩大的情况。因为每个智能体是同时根据联合策略估计自身的值函数，因此当联合动作空间很大时，学习效率及学习效果非常差。为了解决这个问题，算法将值函数 ![](https://www.zhihu.com/equation?tex=Q_%5Cpi%5Ej%28s%2Ca%29) 转化为只包含邻居之间相互作用的形式

![](https://www.zhihu.com/equation?tex=Q_j%28s%2Ca%29%3D%5Cfrac%7B1%7D%7BN_j%7D%5Csum_%7Bk%5Cin+N%28j%29%7DQ_j%28s%2Ca_j%2Ca_k%29%5Cqquad+%284%29)

其中 ![](https://www.zhihu.com/equation?tex=N%28j%29) 表示智能体 j 邻居智能体的标签集， ![](https://www.zhihu.com/equation?tex=N_j%3D%7CN%28j%29%7C) 表示邻居节点的个数。上式 (4) 对智能体之间的交互作用进行了一个近似，降低了表示智能体交互的复杂度，并且保留了部分主要的交互作用（近似保留邻居之间的交互，去掉了非邻居之间的交互）。虽然对联合动作 ![](https://www.zhihu.com/equation?tex=a) 做了近似化简，但是状态信息 ![](https://www.zhihu.com/equation?tex=s) 依然是一个全局信息。

## 1\. Mean Field 近似

下面就是将平均场论（Mean Field Theory，MFT）的思想引入式 (4) 中。该算法假定所有智能体都是同构的，其动作空间相同，并且动作空间是离散的。每个智能体的动作采用 one-hot 编码方式， 如智能体 j 的动作 ![](https://www.zhihu.com/equation?tex=a_j%3D%5Ba_j%5E1%2Ca_j%5E2%2C%5Ccdots%2Ca_j%5ED%5D) 表示共有 D 个动作的动作空间每个动作的值，若选取动作 ![](https://www.zhihu.com/equation?tex=i) ，则 ![](https://www.zhihu.com/equation?tex=a_j%5Ei%3D1) ，其余为 0。定义 ![](https://www.zhihu.com/equation?tex=%5Coverline+a_j) 为智能体 ![](https://www.zhihu.com/equation?tex=j) 邻居 ![](https://www.zhihu.com/equation?tex=N%28j%29) 的平均动作，其邻居 ![](https://www.zhihu.com/equation?tex=k) 的 one-hot 编码动作 ![](https://www.zhihu.com/equation?tex=a_k) 可以表示为 ![](https://www.zhihu.com/equation?tex=%5Coverline+a_j) 与一个波动 ![](https://www.zhihu.com/equation?tex=%5Cdelta+a_%7Bj%2Ck%7D) 的形式

![](https://www.zhihu.com/equation?tex=a_k%3Da_j%2B%5Cdelta+a_%7Bj%2Ck%7D%2C+%5Cqquad+%7B%5Crm+where%7D%5C+%5Coverline+a_j%3D%5Cfrac%7B1%7D%7BN_j%7D%5Csum_k+a_k)

利用泰勒公式展开 (4)，可作出如下近似改写

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+Q_j%28s%2Ca%29%26%3D%5Cfrac%7B1%7D%7BN_j%7D%5Csum_%7Bk%5Cin+N%28j%29%7DQ_j%28s%2Ca_j%2Ca_k%29%5C%5C+%26%3D%5Cfrac%7B1%7D%7BN_j%7D%5Csum_k%5BQ_j%28s%2Ca_j%2C%5Coverline+a_j%29%2B%5Cnabla_%7B%5Cwidetilde+a_%7Bj%2Ck%7D%7DQ_j%28s%2Ca_j%2C%5Coverline+a_j%29%5Cdelta+a_%7Bj%2Ck%7D%2B%5Cfrac%7B1%7D%7B2%7D%5Cdelta+a_%7Bj%2Ck%7D%5Cnabla_%7B%5Cwidetilde+a_%7Bj%2Ck%7D%7D%5E2Q_j%28s%2Ca_j%2C%5Coverline+a_j%29%5Cdelta+a_%7Bj%2Ck%7D%5D%5C%5C+%26%3DQ_j%28s%2Ca_j%2C%5Coverline+a_j%29%2B%5Cnabla_%7B%5Cwidetilde+a_%7Bj%2Ck%7D%7DQ_j%28s%2Ca_j%2C%5Coverline+a_j%29%5B%5Cfrac%7B1%7D%7BN_j%7D%5Csum_k%5Cdelta+a_%7Bj%2Ck%7D%5D%2B%5Cfrac%7B1%7D%7B2N_j%7D%5Csum_k%5B%5Cdelta+a_%7Bj%2Ck%7D%5Cnabla_%7B%5Cwidetilde+a_%7Bj%2Ck%7D%7D%5E2Q_j%28s%2Ca_j%2C%5Coverline+a_j%29%5Cdelta+a_%7Bj%2Ck%7D%5D%5Cqquad+%285%29%5C%5C+%26%3DQ_j%28s%2Ca_j%2C%5Coverline+a_j%29%2B%5Cfrac%7B1%7D%7B2N_j%7D%5Csum_k+R_%7Bs%2Cj%7D%28a_k%29%5C%5C+%26%5Capprox+Q_j%28s%2Ca_j%2C%5Coverline+a_j%29%5Cqquad+%286%29+%5Cend%7Baligned%7D)

式 (7) 中第二项求和为 0，可以直接化简掉，第三项为泰勒展开中的二阶项 ![](https://www.zhihu.com/equation?tex=R_%7Bs%2Cj%7D%28a_k%29) 是一个随机变量。具有如下性质：若值函数 ![](https://www.zhihu.com/equation?tex=Q_j%28s%2Ca_j%2Ca_k%29) 是一个 M-smooth 函数（M 阶导数连续），则 ![](https://www.zhihu.com/equation?tex=R_%7Bs%2Cj%7D%28a_k%29%5Cin+%5B-2M%2C-2M%5D) ，该性质在原文中给出了证明。

根据式 (6)，将式(4) 两两作用求和的形式进一步化简为中心智能体 j 与一个虚拟智能体 ![](https://www.zhihu.com/equation?tex=%5Coverline+a_j) 的相互作用，虚拟智能体是智能体 j 所有邻居作用效果的平均。因此得到 MF-Q 函数 ![](https://www.zhihu.com/equation?tex=Q_j%28s%2Ca_j%2C%5Coverline+a_j%29) 。假设有一段经验 ![](https://www.zhihu.com/equation?tex=%5Bs%2C%5C%7Ba_j%5C%7D%2C%5C%7Br_j%5C%7D%2Cs%27%5D) ，MF-Q 可以通过下式循环更新

![](https://www.zhihu.com/equation?tex=Q_%7Bj%2Ct%2B1%7D%28s%2Ca_j%2C%5Coverline+a_j%29%3D%281-%5Calpha%29Q_%7Bj%2Ct%7D%28s%2Ca_j%2C%5Coverline+a_j%29%2B%5Calpha%5Br_j%2B%5Cgamma+v_%7Bj%2Ct%7D%28s%27%29%5D%5Cqquad+%287%29)

对于为什么不取 max 而是选取 MF-v 函数的情况，其一是因为取 ![](https://www.zhihu.com/equation?tex=%5Cmax_%7Ba_j%2C%5Coverline+a_j%7D) 需要邻居智能体策略的配合，中心智能体不能直接改变邻居智能体的策略；其二取 ![](https://www.zhihu.com/equation?tex=%5Cmax_%7Ba_j%7D) 贪心获取动作，如果每个智能体都贪心获取动作则会因为环境的动态不稳定性而造成算法最终无法收敛。MF-v 函数 ![](https://www.zhihu.com/equation?tex=v_%7Bj%2Ct%7D%28s%27%29) 可以定义为如下形式

![](https://www.zhihu.com/equation?tex=v_%7Bj%2Ct%7D%28s%27%29%3D%5Csum_%7Ba_j%7D%5Cpi_%7Bj%2Ct%7D%28a_j%7Cs%2C%5Coverline+a_j%29E_%7B%5Coverline+a_j%28a_%7B-j%7D%5Csim%5Cpi_%7B-j%2Ct%7D%29%7D%5BQ_%7Bj%2Ct%7D%28s%27%2Ca_j%2C%5Coverline+a_j%29%5D%5Cqquad+%288%29)

在每一时刻的阶段博弈中， ![](https://www.zhihu.com/equation?tex=%5Coverline+a_j) 是通过上一时刻邻居 ![](https://www.zhihu.com/equation?tex=k) 的策略 ![](https://www.zhihu.com/equation?tex=%5Cpi_%7Bk%2Ct%7D) 得出的，其策略参数中的 ![](https://www.zhihu.com/equation?tex=%5Coverline+a_%7Bk-%7D) 也是使用的上一时刻的平均动作，更新过程如下

![](https://www.zhihu.com/equation?tex=%5Coverline+a_j%3D%5Cfrac%7B1%7D%7BN_j%7D%5Csum_k+a_k%2C+%5Cqquad+%7B%5Crm+where%7D%5C+a_k%5Csim+%5Cpi_%7Bk%2Ct%7D%28%5Ccdot%7Cs%2C%5Coverline+a_%7Bk-%7D%29)

通过上式可以计算出邻居平均动作 ![](https://www.zhihu.com/equation?tex=%5Coverline+a_j) ，然后通过玻尔兹曼分布得到新的策略如下形式

![](https://www.zhihu.com/equation?tex=%5Cpi_%7Bj%2Ct%7D%28a_j%7Cs%2C%5Coverline+a_j%29%3D%5Cfrac%7B%5Cexp%28-%5Cbeta+Q_%7Bj%2Ct%7D%28s%2Ca_j%2C%5Coverline+a_j%29%29%7D%7B%5Csum_%7Baj%5Cin+A_j%7D%5Cexp%28-%5Cbeta+Q_%7Bj%2Ct%7D%28s%2Ca_%7Bj%27%7D%2C%5Coverline+a_j%29%29%7D)

通过上两式不断迭代更新，能够提升策略的效果而获得较大的累积回报值。原文中证明 ![](https://www.zhihu.com/equation?tex=%5Coverline+a_j) 能够收敛到唯一的平衡点，并推得策略 ![](https://www.zhihu.com/equation?tex=%5Cpi_j) 收敛到纳什均衡策略。为了与 Nash-Q 算法 (3) 对应，MF-Q 算法给出下式

![](https://www.zhihu.com/equation?tex=%5Cmathcal+H%5E%7BMF%7D+Q%28s%2Ca%29%3DE_%7Bs%27%5Csim+p%7D%5Br%28s%2Ca%29%2B%5Cgamma+v%5E%7BMF%7D%28s%27%29%5D)

最终 MF-Q 的值函数将会收敛到 Nash-Q 的值函数。

## 2\. 算法设计

原文中将 (7) 通过参数化用神经网络进行拟合逼近，与 [DQN](https://link.zhihu.com/?target=https%3A//blog.csdn.net/weixin_37895339/article/details/84822668) 以及 [AC](https://link.zhihu.com/?target=https%3A//blog.csdn.net/weixin_37895339/article/details/74612572) 算法结合，分别给出了 MF-Q 与 MF-AC 算法。

## a. MF-Q

通过神经网络拟合 (7) 式中的 Q 函数，使用如下所示的代价函数，与 DQN 中相同，为了减小拟合的 Q 值与真实 Q 值之间的误差

![](https://www.zhihu.com/equation?tex=L%28%5Cphi_j%29%3D%28y_j-Q_%7B%5Cphi_j%7D%28s%2Ca_j%2C%5Coverline+a_j%29%29%5E2)

其中 ![](https://www.zhihu.com/equation?tex=y_j%3Dr_j%2B%5Cgamma+v_%7B%5Coverline+%5Cphi_j%7D%5E%7BMF%7D%28s%27%29) 为目标 MF 值函数， ![](https://www.zhihu.com/equation?tex=%5Coverline+%5Cphi_j) 对应 DQN 中的目标网络参数。对上式求导可以得出参数梯度方向如下

![](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Cphi_j%7DL%28%5Cphi_j%29%3D%28y_j-Q_%7B%5Cphi_j%7D%28s%2Ca_j%2C%5Coverline+a_j%29%29%5Cnabla_%7B%5Cphi_j%7DQ_%7B%5Cphi_j%7D%28s%2Ca_j%2C%5Coverline+a_j%29%5Cqquad+%289%29)

通过梯度下降法可以得到参数更新公式。

![](https://pic4.zhimg.com/v2-adff0c5ccd84498b96e50818b85117c7_b.jpg)![](https://pic4.zhimg.com/80/v2-adff0c5ccd84498b96e50818b85117c7_hd.jpg)

## b. MF-AC

类似 [DPG](https://link.zhihu.com/?target=https%3A//blog.csdn.net/weixin_37895339/article/details/84881872)，使用神经网络拟合一个策略替换玻尔兹曼机通过 Q 的到的策略，则得到了 MF-AC 算法。

类似 [SPG](https://link.zhihu.com/?target=https%3A//blog.csdn.net/weixin_37895339/article/details/84792752)，actor 策略梯度公式可以写为

![](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Ctheta_j%7D+L%28%5Ctheta_j%29%3D%5Cnabla_%7B%5Ctheta_j%7D%5Clog%5Cpi_%7B%5Ctheta_j%7D%28s%29Q_%7B%5Cphi_j%7D%28s%2Ca_j%2C%5Coverline+a_j%29%7C_%7Ba%3D%5Cpi_%7B%5Ctheta_j%7D%28s%29%7D)

critic 类似于 MF-Q 的更新方式 (9)

![](https://pic3.zhimg.com/v2-e2ded84b7c18435a2caea90477f22276_b.jpg)![](https://pic3.zhimg.com/80/v2-e2ded84b7c18435a2caea90477f22276_hd.jpg)

文章给出了[仿真代码](https://link.zhihu.com/?target=https%3A//github.com/mlii/mfrl)，原文[地址](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1802.05438.pdf)。

## 总结

MFMARL 算法主要解决的是联合动作 ![](https://www.zhihu.com/equation?tex=a%3D%5Ba_1%2C%5Ccdots%2Ca_n%5D) 的维度随智能体数量增多的扩张问题，将 ![](https://www.zhihu.com/equation?tex=a) 的维度缩减为 ![](https://www.zhihu.com/equation?tex=%5Ba_j%2C%5Coverline+a_j%5D) 。但是各个智能体的策略还是需要直到全局的状态信息 ![](https://www.zhihu.com/equation?tex=s) ，不算是一个分布化的算法，并且依赖于通信获取邻居智能体的动作 ![](https://www.zhihu.com/equation?tex=a_k) 。

虽然不是完全分布式的，但是该算法是一个解决大规模数量智能体强化学习的一个非常有效的算法，并且理论证明十分严格。

**相关连接：**

[ECKai：多智能体强化学习入门（六）——MFMARL 算法（Mean Field Multi-Agent RL）​zhuanlan.zhihu.com![](https://pic1.zhimg.com/v2-8cd255b42471ab6e227d5eaeb8b489ac_180x120.jpg)](https://zhuanlan.zhihu.com/p/56049023)[ECKai：多智能体强化学习入门（一）——基础知识与博弈​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/53474965)[ECKai：多智能体强化学习入门（二）——基础算法（MiniMax-Q，NashQ，FFQ，WoLF-PHC）​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/53563792)[ECKai：多智能体强化学习入门（三）——矩阵博弈中的分布式学习算法​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/53622102)[ECKai：多智能体强化学习入门（四）——MADDPG 算法​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/53811876)[ECKai：多智能体强化学习入门（五）——QMIX 算法分析​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/55003734)

附录：

[ECKai：附：强化学习——DRQN 分析详解​zhuanlan.zhihu.com![](https://pic3.zhimg.com/v2-5286358fcfe6318821edecc74bd3febe_180x120.jpg)](https://zhuanlan.zhihu.com/p/54898904)
写下你的评论...

万物皆平均

谢谢分享