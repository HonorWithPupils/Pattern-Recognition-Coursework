# AdaBoost

对于一个二分类问题数据集
$$
T = \lbrace (x_1,y_1),...,(x_N,y_N) \rbrace
$$
其中 $x_i\in \mathcal X \subseteq \mathbb R^m, y_i\in\mathcal Y=\lbrace-1,+1\rbrace$ 。AdaBoost 使用以下算法，从训练数据中学习一系列弱分类器，并将这些弱分类器线性组合称为一个强分类器。

## 算法

输入：训练集数据；弱学习器 $G_m$ 

输出：一个强学习器 $G$

#### 1. 初始化训练数据的权重向量

$$
D_1=(\overbrace{\frac{1}{N},...,\frac{1}{N}}^N)
$$

#### 2. for m=1,...,M

a）使用具有权重 $D_m$ 的数据集训练弱学习器，得到
$$
G_m(x):\mathcal X\to \lbrace-1,+1\rbrace
$$
b）计算 $G_m(x)$ 在训练集上的分类误差率，
$$
e_m=\sum_{n=1}^N w_{mn}I(G_m(x_n)\neq y_n)
$$
c）更新权重向量，
$$
w_{m+1,n} = \frac{w_{mn}}{Z_m} \exp(-\alpha_my_n G_m(x_n)) \\ \\
Z_m = \sum_{n=1}^N w_{mn}\exp(-\alpha_my_n G_m(x_n)) \\ \\
\alpha_m = \frac 1 2 \ln \frac{1-e_m}{e_m}
$$
其中 $\alpha_m$ 是最重要的参数，它不但表示在最终分类器中，每一个弱分类器所占的比重，也表示每个弱分类器对权重更新的程度。

#### 3. 构建最终学习器

$$
G(x)=sign(\sum_{m=1}^M\alpha_mG_m(x))
$$

## 误差上界

$$
e(T) = \frac 1 N \sum_{n=1}^N I(G(x_n)\neq y_n) \le \prod_{m=1}^M Z_m
$$

证明：

首先易得，
$$
\begin{align}
e(T) &= \frac 1 N \sum_{n=1}^N I(G(x_n)\neq y_n) \\ \\ 
&\le \frac 1 N \sum_{n=1}^N \exp\left[-y_n\sum_{m=1}^M\alpha_mG_m(x_n)\right] \\ \\
& = \sum_{n=1}^N w_{1n}\prod_{m=1}^M \exp
\end{align}
$$
由 ${Z_m}w_{m+1,n} = w_{mn} \exp(-\alpha_my_n G_m(x_n))$ ，继续推导，
$$
\begin{align}
e(T) 
&\le \frac 1 N \sum_{n=1}^N \exp\left[-y_n\sum_{m=1}^M\alpha_mG_m(x_n)\right] \\ \\
& = \sum_{n=1}^N w_{1n}\prod_{m=1}^M \exp(-\alpha_my_nG_m(x_n)) \\ \\
& = \sum_{n=1}^N w_{1n}\exp(-\alpha_1y_nG_1(x_n)) \prod_{m=2}^M \exp(-\alpha_my_nG_m(x_n)) \\ \\
& = \sum_{n=1}^N Z_1w_{2n} \prod_{m=2}^M \exp(-\alpha_my_nG_m(x_n)) \\ \\
& = Z_1\sum_{n=1}^N w_{2n}\exp(-\alpha_2y_nG_2(x_n)) \prod_{m=3}^M \exp(-\alpha_my_nG_m(x_n)) \\ \\
& = \ \cdots \\ \\
& = \prod_{m=1}^M Z_m
\end{align}
$$

## $\alpha_m$ 的选择

我们现在已经知道了误差上界为 $\prod_{m=1}^M Z_m$ ，因此应该寻找 $\alpha_m$ 使得 $Z_m$ 最小，
$$
\def\part{\partial}
\begin{align}
\frac{\part Z_m}{\part \alpha_m}
&= \frac{\part \sum_{n=1}^N w_{mn}\exp(-\alpha_my_n G_m(x_n))}{\part \alpha_m} \\ \\
&= -\sum_{n=1}^Nw_{mn}y_n G_m(x_n)\exp(-\alpha_my_n G_m(x_n)) \\ \\
& = -\exp(-\alpha_m)\sum_{y_i=G_m(x_i)}w_{mn}+\exp(\alpha_m)\sum_{y_i\neq G_m(x_i)}w_{mn} \\ \\
& = -\exp(-\alpha_m)(1-e_m)+\exp(\alpha_m)e_m = 0 \\ \\
&\ \ \Rightarrow \alpha_m=\frac 1 2 \ln(\frac{1-e_m}{e_m})
\end{align}
$$
 确定完 $\alpha_m$ 之后，我们可以进一步界定误差上界，
$$
\begin{align}
Z_m &= \exp(-\alpha_m) \sum_{y_i=G_m(x_i)} w_{mn} + \exp(\alpha_m) \sum_{y_i\neq G_m(x_i)} w_{mn} \\ \\
&= (1-e_m)\sqrt{\frac{e_m}{1-e_m}}+e_m\sqrt{\frac{1-e_m}{e_m}} \\ \\
&=2\sqrt{(1-e_m)e_m} \le 1
\end{align}
$$
因此，只要保证每个弱分类器只要比随机猜测略好（$e_m<0.5$），就能保证最终的强分类器的误差上界被不断算小。