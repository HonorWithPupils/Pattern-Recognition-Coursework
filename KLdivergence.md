# KL 散度

KL 散度 ( KL divergence，Kullback-Leibler divergence ) 是描述两个概率 $Q(x)$ 和 $P(x)$ 相似度的一种度量，记作 $D(Q||P)$ 。对离散随机变量，KL 散度定义为
$$
D(Q||P)=\sum_iQ(i)\log\frac {Q(i)}{P(i)}
$$
易于证明 $D(Q||P)\ge0$ ，利用 Jensen 不等式可得，
$$
\begin{aligned}
-D(Q||P) &=-\sum_iQ(i)\log\frac {Q(i)}{P(i)} \\ \\
&\le  \log\sum_iQ(i)\frac {P(i)}{Q(i)} \\ \\
&= \log1=0
\end{aligned}
$$

$$
\Rightarrow D(Q||P)\ge0
$$

然而 由于 $D(Q||P)\neq D(P||Q)$ ，也不满足三角不等式，所以不是严格意义上的距离