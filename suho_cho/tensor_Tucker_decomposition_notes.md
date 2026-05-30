# Tucker Decomposition Notes

**핵심:** Matrix 에서의 PCA (SVD)를 텐서에서의 Tucker decomposition (HOSVD)로 확장.

---

## Singular Value Decomposition (SVD)

In general, matrix $A \in \mathbb{R}^{m \times n}$ can be decomposed:

$$A = U S V^T$$

where
- $U \in \mathbb{R}^{m \times m}$ : left singular vectors
- $S = \begin{bmatrix} \sigma_1 & & \\ & \sigma_2 & \\ & & \ddots \\ & & & \sigma_{\min(m,n)} \end{bmatrix} \in \mathbb{R}^{m \times n}$ : singular values
- $V \in \mathbb{R}^{n \times n}$ : right singular vectors

전개:

$$A^T A = V S^T U^T U S V^T = V (S^T S) V^T = \sigma^2 V V^T$$

$$\Rightarrow \quad A^T A\, V = \sigma^2 V$$

$$\Rightarrow \quad \sigma^2 \text{ is eigenvalue, } V \text{ is eigenvector of } A^T A$$

Likewise,

$$U \text{ is eigenvector of } A A^T$$

$U, V$ 는 각각 column wise orthonormal 이고 회전 또는 대칭 이동을 뜻함.
$S$ 는 diagonal 하고 크기 변환을 뜻함.

동일한 구조가 Tucker decomposition 에도 적용됨.

---

## Tucker Decomposition

The Tucker decomposition is a form of a higher-order PCA. It decomposes a tensor into a core tensor multiplied (or transformed) by a matrix along each mode.

In the three-way case, where $X \in \mathbb{R}^{I \times J \times K}$:

$$X \approx G \times_1 A \times_2 B \times_3 C = \sum_{p=1}^{P} \sum_{q=1}^{Q} \sum_{r=1}^{R} g_{pqr}\, a_p \circ b_q \circ c_r = [\![\,G\,;A,B,C\,]\!]$$

(여기서 $G \times_1 A = A \cdot G_{(1)}$)

where
- $A \in \mathbb{R}^{I \times P}$, $B \in \mathbb{R}^{J \times Q}$, $C \in \mathbb{R}^{K \times R}$ are the factor matrices
- $G \in \mathbb{R}^{P \times Q \times R}$ is called the core tensor
- '$\times_n$' is $n$-mode product (= $G$ Tensor 의 $n$-mode 에 Matrix $(A, B, C)$ 간 곱)

Elementwise,

$$x_{ijk} \approx \sum_{p=1}^{P} \sum_{q=1}^{Q} \sum_{r=1}^{R} g_{pqr}\, a_{ip}\, b_{jq}\, c_{kr} \quad \text{for } i = 1, \ldots, I,\ j = 1, \ldots, J,\ k = 1, \ldots, K$$

If $P, Q, R$ are smaller than $I, J, K$, the core tensor $G$ can be thought of as a **compressed version** of origin tensor ($X$).

In fact, CP can be viewed as a special case of Tucker where the core tensor is superdiagonal ($p = q = r$) and $P = Q = R$.

$$\sum_p \sum_q \sum_r g_{pqr}\, a_p \circ b_q \circ c_r \xrightarrow{\ p=q=r\ } \sum_r \lambda_r\, a_r \circ b_r \circ c_r$$

---

## The Flattening Forms of Tucker Decomposition

$$X_{(1)} \approx A\, G_{(1)} (C \otimes B)^T$$

$$X_{(2)} \approx B\, G_{(2)} (C \otimes A)^T$$

$$X_{(3)} \approx C\, G_{(3)} (B \otimes A)^T$$

---

## Tucker1 Decomposition

The Tucker1 decomposition sets two of the factor matrices to be the identity matrix.

e.g. $B = C = I$

$$\Rightarrow \quad X = G \times_1 A \times_2 I \times_3 I = G \times_1 A = [\![\,G\,;A,I,I\,]\!]$$

$$\therefore \quad X_{(1)} = A\, G_{(1)}$$

---

## Tucker2 Decomposition

Tucker2 decomposition of a third-order array sets one of the factor matrices to be the identity matrix.

e.g. $C = I$

$$\Rightarrow \quad X = G \times_1 A \times_2 B \times_3 I = G \times_1 A \times_2 B = [\![\,G\,;A,B,I\,]\!]$$

---

## The $n$-Rank

The $n$-Rank of $X$, denoted $\text{rank}_n(X)$, is the column rank of $X_{(n)}$. If we let $R_n = \text{rank}_n(X)$ for $n = 1, \ldots, N$, then we can say that $X$ is a rank-$(R_1, R_2, \ldots, R_N)$ tensor.

Trivially, $R_n \le I_n$ for all $n = 1, \ldots, N$.

---

## Higher Order SVD (HOSVD)

The basic idea is to find those components that best capture the variation in mode $n$, independent of the other modes. This is sometimes referred to as the "Tucker1" method.

When $R_n < \text{rank}_n(X)$ for one or more $n$, the decomposition is called the **truncated HOSVD**. In fact, the core tensor of the HOSVD is **all-orthogonal**, which has relevance to truncating the decomposition. The truncated HOSVD is not optimal in terms of giving the best fit as measured by the norm of the difference.

```
def HOSVD(X, R_1, ..., R_n):
    for n in range(N):
        A^(n+1) <- R_n leading left singular vectors of X_(n+1)
    G = X x_1 A^(1)T ... x_n A^(N)T
    return G, A^(1), ..., A^(N)
```

$R_n < \text{rank}_n(X)$ 일 때 손실 발생.

---

## Higher-Order Orthogonal Iteration (HOOI)

If we assume that $X$ is a tensor of size $I_1 \times I_2 \times \cdots \times I_N$, then the optimization problem that we wish to solve is

$$\min_{G, A^{(1)}, \ldots, A^{(N)}} \left\| X - [\![\,G\,;A^{(1)}, A^{(2)}, \ldots, A^{(N)}\,]\!] \right\|$$

where
- $G \in \mathbb{R}^{R_1 \times R_2 \times \cdots \times R_N}$
- $A^{(n)} \in \mathbb{R}^{I_n \times R_n}$ and columnwise orthogonal for $n = 1, \ldots, N$

전개:

$$
\begin{aligned}
\left\| X - [\![\,G\,;A^{(1)}, A^{(2)}, \ldots, A^{(N)}\,]\!] \right\|^2
&= \|X\|^2 - 2\langle X, [\![\,G\,;A^{(1)}, \ldots, A^{(N)}\,]\!] \rangle + \left\| [\![\,G\,;A^{(1)}, \ldots, A^{(N)}\,]\!] \right\|^2 \\
&= \|X\|^2 - 2\langle X \times_1 A^{(1)T} \times_2 A^{(2)T} \cdots \times_N A^{(N)T},\ G \rangle + \|G\|^2 \\
&\qquad (\because A^{(n)} \text{ is columnwise orthonormal}) \\
&= \|X\|^2 - 2\langle G, G \rangle + \|G\|^2 = \|X\|^2 - \|G\|^2 \\
&= \|X\|^2 - \left\| X \times_1 A^{(1)T} \times_2 A^{(2)T} \cdots \times_N A^{(N)T} \right\|^2
\end{aligned}
$$

Since $\|X\|$ is constant, we should maximize $\left\| X \times_1 A^{(1)T} \times_2 A^{(2)T} \cdots \times_N A^{(N)T} \right\|$

$$\Rightarrow \quad \max_{A^{(n)}} \left\| X \times_1 A^{(1)T} \times_2 A^{(2)T} \cdots \times_N A^{(N)T} \right\|$$

$$\Rightarrow \quad \max_{A^{(n)}} \left\| A^{(n)T} W \right\| \quad \text{where } W = X_{(n)} \left( A^{(N)} \otimes \cdots \otimes A^{(n+1)} \otimes A^{(n-1)} \otimes \cdots \otimes A^{(1)} \right)$$

The solution can be determined using SVD; simply set $A^{(n)}$ to be the $R_n$ leading left singular vectors of $W$.

---

## Tucker Decomposition: 수학적 요약

- **HOSVD:** 각 mode의 flattening Matrix에 SVD를 적용. 반복은 없고, 최적으로 근사를 보장 못함.
- **HOOI:** HOSVD의 값을 초기값으로 시작하여, 수렴할 때까지 반복 근사하는 알고리즘. 최적해는 아닐 수 있음.
