# Least Square 最小二乘法

最小二乘法的基本原理如下，针对线性回归模型 $y=\bm w^T\bm x$，寻找参数 $w$ 使得残差平方和最小，
$$
\def\part{\partial}
\def\grad{\nabla}
\def\bm{\boldsymbol}
\min_\bm w f(\bm w)=||X\bm w-\bm y||^2
$$

## 直接求解

$$
\begin{align}
\frac{\part f(\bm w)}{\part \bm w}&=\frac{\part(X\bm w-\bm y)^T(X\bm w-\bm y)}{\part \bm w} \\ \\
&= 2X^T(X\bm w-\bm y) = 0 \\ \\
\Rightarrow \ & \bm w=(X^TX)^{-1}X\bm y
\end{align}
$$

又因为二阶导数半正定，所以这个解是极值点。
$$
\frac{\part^2f(\bm w)}{\part\bm w^2} = 2X^TX\succeq0
$$
注意 $X^TX$ 存在逆，这要求 $X$ 是列无关的，这个要求在梯度下降法的证明中也是需要的，若这个要求不符合，意味着存在不知唯一一个 $w$ 满足残差平方和最小。

由于当数据规模极其巨大的时候，计算 $X^TX$ 是不现实的，所以通常情况下会使用迭代法，而考虑到 $X^TX$ 是一个半正定矩阵，最小二乘问题只需要凸优化方法即可。

## 梯度下降法

$$
\bm w_{i+1} = \bm w_i -\eta\grad f(\bm w_i)
$$

其中 $\eta$ 为学习率，是一个可供选择的常数或者变量。

### 引理 L-smooth 条件

若函数 $f(x)$ 对于任意两个$x,y$，存在一个常数 $L>0$ 使得，
$$
||\grad f(x)-\grad f(y)||\le L||x-y||
$$
则称 $f(x)$ 符合 L-Lipschitz 条件。

证明：
$$
\begin{align}
||\grad f(\bm w_1)-\grad f(\bm w_2)|| &= ||2X^T(X\bm w_1-\bm y)-2X^T(X\bm w_2-\bm y)|| \\ \\
& = 2||X^TX(\bm w_1-\bm w_2)|| \\ \\
& \le  ||X^TX||\cdot 2||\bm w_1-\bm w_2||
\end{align}
$$
另 $L=2||X^TX||>0$，所以函数 $f(\bm w)$ 符合  L-smooth 条件。

### 引理 L-smooth 等价形式

$$
|f(x)-f(y)-\grad f(y)^T(x-y)|\le \frac{L}{2}||x-y||^2
$$

证明：

构造一个插值函数 $g(t) = f(y+t(x-y))$，对 $t$ 求导，
$$
g'(t)=\grad f(y+t(x-y))^T(x-y)
$$
可以把函数值之差转变为积分，
$$
f(x)-f(y)=g(1)-g(0) =\int_0^1\grad f(y+t(x-y))^T(x-y)dt
$$
将 上式代入等式左侧，
$$
\begin{align}
\text{left} &= |\int_0^1\grad f(y+t(x-y))^T(x-y)dt-\grad f(y)^T(x-y)| \\ \\
& = |\int_0^1[\grad f(y+t(x-y))^T(x-y) - \grad f(y)^T(x-y)]dt| \\ \\
& \le \int_0^1|[\grad f(y+t(x-y))-\grad f(y)]^T(x-y)|dt \\ \\
& \le \int_0^1 \sqrt{||\grad f(y+t(x-y))-\grad f(y)||^2||x-y||^2} dt \\ \\
& \le \int_0^1 \sqrt{L||t(x-y)||^2||x-y||^2} dt \\ \\
& =L||x-y||^2\int_0^1tdt=\frac{L}{2}||x-y||^2
\end{align}
$$
其中第3行使用了和的绝对值小于等于绝对值的和，第4行使用了柯西施瓦茨不等式 $a^Tb\le\sqrt{||a||^2||b||^2}$ ，第5行代入了 L-smooth 条件。

删去绝对值，可得到，
$$
f(x)\le f(y)+\grad f(y)^T(x-y)+\frac{L}{2}||x-y||^2
$$

### 证明 收敛性（固定学习率）

接下来证明梯度下降法在最小二乘问题上的收敛性，注意这个证明具有一般性（符合 L-smooth 条件的凸函数），所以会略显冗长。
$$
\begin{align}
f(\bm w_{i+1}) &\le f(\bm w_i) +\grad f(\bm w_i)^T(\bm w_{i+1}-\bm w)+\frac{L}{2}||\bm w_{i+1}-\bm w||^2 \\ \\
& = f(\bm w_i) +\grad f(\bm w_i)^T(-\eta\grad f(\bm w_i))+\frac{L}{2}||-\eta\grad f(\bm w_i)||^2 \\ \\ 
& = f(\bm w_i)-(1-\frac{L\eta}{2})\eta||\grad f(\bm w_i)||^2
\end{align}
$$
选择 $\eta L\le 1$ ，则 $1-\frac{L\eta}{2}\ge\frac{1}{2}$ ，所以得到：
$$
f(\bm w_{i+1})\le f(\bm w_i) - \frac 1 2\eta||\grad f(\bm w_i)||^2
$$
观察上面的公式，我们会发现 $f(\bm w)$ 每次迭代，都会让 $f(\bm w)$ 变得更小，朝着更好的方向去前进，也就是单调性，并且我们已经知道存在最小值，所以梯度下降法收敛。接下来我们继续求证收敛速度。

假设最优解 $f(\bm w^*)$ 为最优解，那么根据泰勒一阶展开，以及 $f(\bm w)$ 是一个凸函数：
$$
f(\bm w_i)\le f(\bm w^*)+\grad f(\bm w_i)^T(\bm w_i-\bm w^*)
$$
代入上式子，
$$
\begin{align}
f(\bm w_{i+1}) &\le f(\bm w^*)+\grad f(\bm w_i)^T(\bm w_i-\bm w^*) - \frac 1 2\eta||\grad f(\bm w_i)||^2 \\ \\
f(\bm w_{i+1}) - f(\bm w^*) &\le \frac{1}{2\eta}\left( 2\eta\grad f(\bm w_i)^T(\bm w_i -\bm w^*)-\eta^2||\grad f(\bm w_i)||^2  \right) \\ \\
f(\bm w_{i+1}) - f(\bm w^*) &\le \frac{1}{2\eta}\left( ...-||\bm w_i -\bm w^*||^2+||\bm w_i -\bm w^*||^2  \right) \\ \\
f(\bm w_{i+1}) - f(\bm w^*) &\le \frac{1}{2\eta}\left( ||\bm w_i -\bm w^*||^2-||\bm w_i - \eta\grad f(\bm w_i) -\bm w^*||^2  \right) \\ \\
f(\bm w_{i+1}) - f(\bm w^*) &\le \frac{1}{2\eta}\left( ||\bm w_i -\bm w^*||^2-||\bm w_{i+1} -\bm w^*||^2  \right)
\end{align}
$$
将 $i=0,...,k-1$ 代入上式，得到
$$
\begin{align}
f(\bm w_{1}) - f(\bm w^*) &\le \frac{1}{2\eta}\left( ||\bm w_0 -\bm w^*||^2-||\bm w_1 -\bm w^*||^2  \right) \\ \\
f(\bm w_2) - f(\bm w^*) &\le \frac{1}{2\eta}\left( ||\bm w_1 -\bm w^*||^2-||\bm w_2 -\bm w^*||^2  \right) \\ \\
&\vdots \\ \\
f(\bm w_{k}) - f(\bm w^*) &\le \frac{1}{2\eta}\left( ||\bm w_{k-1} -\bm w^*||^2-||\bm w_{k} -\bm w^*||^2  \right) 
\end{align}
$$
将上式全部相加，得到
$$
\begin{align}
\sum_{i=1}^k[f(\bm w_{i}) - f(\bm w^*)] &= \frac{1}{2\eta}(\left( ||\bm w_0 -\bm w^*||^2-||\bm w_k -\bm w^*||^2  \right)) \\ \\
&\le \frac{1}{2\eta}||\bm w_0 -\bm w^*||^2 \\ \\
\sum_{i=1}^k[f(\bm w_k) - f(\bm w^*)] &\le \sum_{i=1}^k[f(\bm w_i) - f(\bm w^*)] \le \frac{1}{2\eta}||\bm w_0 -\bm w^*||^2 \\ \\
f(\bm w_k) &\le f(\bm w^*)+\frac{1}{2\eta k}||\bm w_0 -\bm w^*||^2
\end{align}
$$
证明完毕。随着 $k$ 越来越大，误差 $\epsilon=\frac{1}{2\eta k}||\bm w_0 -\bm w^*||^2$ 也越来越小，从上面得到的公式我们可以知道在一个符合 L-smooth 的凸函数上，梯度下降法的收敛步数为 $O(\frac{1}{\epsilon})$ ，是次线性收敛 ；若在证明中加入强凸的属性，则梯度下降法的收敛步数为 $O(log(\frac{1}{\epsilon}))$ ，是线性收敛。

## 牛顿法

由泰勒展开式得，
$$
f(\bm w_i +\bm d) = f(\bm w_i)+\grad f(\bm w_i)^T\bm d + \frac{1}{2}\bm d^T\grad^2f(\bm w_i)\bm d + o(||\bm d||^2)
$$
舍弃高阶项，并对 $\bm d$ 求导，
$$
\grad_{\bm d}f(\bm w_i +\bm d) =\grad f(\bm w_i)+\grad^2f(\bm w_i)\bm d=0 \\ \\
\Rightarrow \bm d = -\grad^2f(\bm w_i)^{-1}\grad f(\bm w_i)
$$
因此迭代方程为，
$$
\bm w_{i+1} = \bm w_i-\grad^2f(\bm w_i)^{-1}\grad f(\bm w_i)
$$

### 证明 收敛性

首先对于最小二乘问题，
$$
H(\bm w) = \grad^2f(\bm w) = 2X^TX=C
$$
在迭代方程两边同时减去最优点 $\bm w^*$ ，
$$
\begin{align}
\bm w_{i+1} - \bm w^* &= \bm w_i - \bm w^* -H^{-1}(\bm w_i)\grad f(\bm w_i) \\ \\
& = \bm w_i - \bm w^* -H^{-1}(\bm w_i)[\grad f(\bm w_i)-\grad f(\bm w^*)]\ \ (\grad f(\bm w^*)=\bm 0)
\end{align}
$$
构造插值函数 $g(t)=\grad f(\bm w_i+t(\bm w^*-\bm w_i))$，则 $g'(t)=H(\bm w_i+t(\bm w^*-\bm w_i))(\bm w^*-\bm w_i)$
$$
\begin{align}
\grad f(\bm w^*)-\grad f(\bm w_i) &= g(1)-g(0) \\ \\
&= \int_0^1g'(t)dt \\ \\
-\grad f(\bm w_i)&= \int_0^1H(\bm w_i+t(\bm w^*-\bm w_i))(\bm w^*-\bm w_i)dt
\end{align}
$$
故得，
$$
\begin{align}
\bm w_{i+1} - \bm w^* &= \bm w_i - \bm w^* + H^{-1}(\bm w_i)\int_0^1H(\bm w_i+t(\bm w^*-\bm w_i))(\bm w^*-\bm w_i)dt \\ \\
& = H^{-1}(\bm w_i)\int_0^1[H(\bm w_i+t(\bm w^*-\bm w_i))-H(\bm w_i)](\bm w^*-\bm w_i)dt \\ \\
&= 0
\end{align}
$$
因此对于最小二乘问题（线性），对于任意起点，牛顿法只需要一步就可以达到最优点。对于没有海森矩阵 $H(w)$ 为常数性质的问题，当海森矩阵有界且 Lipschitz 连续时，也可以通过一系列范数放缩，证明牛顿法是二次收敛的。

其实对于最小二乘法问题（线性）牛顿法和直接求解法是相同的。
$$
\begin{align}
\bm w_{i+1} &= \bm w_i-\grad^2f(\bm w_i)^{-1}\grad f(\bm w_i) \\ \\
&= \bm w_i - (2X^TX)^{-1}\cdot2X^T(X\bm w_i-\bm y) \\ \\
&= (X^TX)^{-1}X\bm y
\end{align}
$$
可见直接使用牛顿法，在最小二乘问题上，其实是意义不大的。
