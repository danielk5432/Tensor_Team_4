# Tensor Decompositions and Applications
*T.G. Kolda, B.W. Bader*

---

## Inner Product of Two Same-Sized Tensors

For $X, Y \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$:

$$\langle X, Y \rangle = \sum_{i_1=1}^{I_1} \sum_{i_2=1}^{I_2} \cdots \sum_{i_N=1}^{I_N} x_{i_1 i_2 \cdots i_N} \, y_{i_1 i_2 \cdots i_N}$$

```python
def Inn_Pro(X, Y):
    I_1 = len(X)
    # ...
    I_N = len(X[0][0]...[0])  # N-1 times
    Sum = 0.0

    for i1 in range(I_1):
        for i2 in range(I_2):
            # ...
            for iN in range(I_N):
                Sum += X[i1, i2, ..., iN] * Y[i1, i2, ..., iN]

    return Sum
```

---

## Frobenius Norm

The norm of a tensor $X \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ is the square root of the sum of the squares of all its elements:

$$\|X\| = \sqrt{\sum_{i_1=1}^{I_1} \sum_{i_2=1}^{I_2} \cdots \sum_{i_N=1}^{I_N} x_{i_1 i_2 \cdots i_N}^2}$$

```python
def Fro_Norm(X):
    return np.sqrt(Inn_Pro(X, X))
```

---

## CP Decomposition

The CP decomposition factorizes a tensor into a sum of component rank-one tensors. For example, given a third-order tensor $X \in \mathbb{R}^{I \times J \times K}$:

$$X \approx \sum_{r=1}^{R} a_r \circ b_r \circ c_r$$

where $R$ is a positive integer, $a_r \in \mathbb{R}^I$, $b_r \in \mathbb{R}^J$, $c_r \in \mathbb{R}^K$.

Elementwise, it is written as:

$$x_{ijk} = \sum_{r=1}^{R} a_{ir} b_{jr} c_{kr} \quad (\text{for } i=1,\ldots,I,\ j=1,\ldots,J,\ k=1,\ldots,K)$$

---

## Khatri-Rao Product

$$A \odot B = [a_1 \otimes b_1 \quad a_2 \otimes b_2 \quad \cdots \quad a_R \otimes b_R]$$

where $A \in \mathbb{R}^{I \times R}$, $B \in \mathbb{R}^{J \times R}$ are matrices with column size $R$, and $A \odot B \in \mathbb{R}^{(IJ \times R)}$.

---

## Factor Matrices (Matricized Form)

The factor matrices refer to the combination of the vectors from the rank-one components ($A = [a_1\ a_2\ \cdots\ a_R]$, likewise for $B$ and $C$). Using CP decomposition's definitions, it may be written in matricized form:

$$X_{(1)} \approx A (C \odot B)^T \quad $$

$$X_{(2)} \approx B (C \odot A)^T$$

$$X_{(3)} \approx C (B \odot A)^T$$

### Example: $X \in \mathbb{R}^{3 \times 4 \times 2}$

$$X_1 = \begin{bmatrix} 1 & 4 & 7 & 10 \\ 2 & 5 & 8 & 11 \\ 3 & 6 & 9 & 12 \end{bmatrix}, \quad X_2 = \begin{bmatrix} 13 & 16 & 19 & 22 \\ 14 & 17 & 20 & 23 \\ 15 & 18 & 21 & 24 \end{bmatrix}$$

CP decomposition → $X'$: 2 rank-1 tensors

$$X' = a_1 \otimes b_1 \otimes c_1 + a_2 \otimes b_2 \otimes c_2$$

**Factor matrices:**

$$A = [a_1\ a_2] = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$$

$$B = [b_1\ b_2] = \begin{bmatrix} 1 & 2 \\ 0 & 1 \\ 2 & 0 \\ 1 & 1 \end{bmatrix}$$

$$C = [c_1\ c_2] = \begin{bmatrix} 1 & 1 \\ 0 & 2 \end{bmatrix}$$

$$C \odot B = [c_1 \otimes b_1 \quad c_2 \otimes b_2]$$

$$c_1 \otimes b_1 = \begin{bmatrix} 1 \times \begin{bmatrix}1\\0\\2\\1\end{bmatrix} \\ 0 \times \begin{bmatrix}1\\0\\2\\1\end{bmatrix} \end{bmatrix} = \begin{bmatrix}1\\0\\2\\1\\0\\0\\0\\0\end{bmatrix}$$

$$c_2 \otimes b_2 = \begin{bmatrix} 1 \times \begin{bmatrix}2\\1\\0\\1\end{bmatrix} \\ 2 \times \begin{bmatrix}2\\1\\0\\1\end{bmatrix} \end{bmatrix} = \begin{bmatrix}2\\1\\0\\1\\4\\2\\0\\2\end{bmatrix}$$

$$\Rightarrow C \odot B = \begin{bmatrix} 1 & 2 \\ 0 & 1 \\ 2 & 0 \\ 1 & 1 \\ 0 & 4 \\ 0 & 2 \\ 0 & 0 \\ 0 & 2 \end{bmatrix}$$

$$X_{(1)} = A(C \odot B)^T = \begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix} \begin{bmatrix}1&0&2&1&0&0&0&0\\2&1&0&1&4&2&0&2\end{bmatrix}$$

$$= \begin{bmatrix} 1&0&2&1&0&0&0&0 \\ 2&1&0&1&4&2&0&2 \\ 3&1&2&2&4&2&0&2 \end{bmatrix}$$

---

## Matricization (Unfolding or Flattening)

The mode-$n$ matricization of a tensor $X \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ is denoted by $X_{(n)}$, and arranges the mode-$n$ fibers to be the columns of the resulting matrix.

$$\text{Tensor } \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N} \rightarrow \text{Matrix } \mathbb{R}^{I_n \times (I_1 \times \cdots \times I_{n-1} \times I_{n+1} \times \cdots \times I_N)}$$

Index mapping:

$$(i_1, i_2, \ldots, i_N) \mapsto \left(i_n,\ 1 + \sum_{\substack{k=1 \\ k \neq n}}^{N} (i_k - 1) J_k \right)$$

where $J_k = \prod_{\substack{m=1 \\ m \neq n}}^{k-1} I_m$.

### Example: $X \in \mathbb{R}^{3 \times 4 \times 2}$

$$X_1 = \begin{bmatrix}1&4&7&10\\2&5&8&11\\3&6&9&12\end{bmatrix}, \quad X_2 = \begin{bmatrix}13&16&19&22\\14&17&20&23\\15&18&21&24\end{bmatrix}$$

$$X_{(1)} = \begin{bmatrix}1&4&7&10&13&16&19&22\\2&5&8&11&14&17&20&23\\3&6&9&12&15&18&21&24\end{bmatrix}$$

Verification: $x_{2,3,2} = X_{(1), 2, 7}$

Since $j = 1 + (i_2 - 1)J_2 + (i_3 - 1)J_3 = 1 + (3-1)\times1 + (2-1)\times4 = 1 + 2 + 4 = 7$ ✓

```python
def Find_Flattening_index(Origin_Ten, Origin_ind, mode):
    mode -= 1
    Flat_ind = [Origin_ind[mode], 0]
    I = []

    dummy = Origin_Ten.copy()
    for _ in range(len(origin_ind)):
        I.append(len(dummy))
        dummy = dummy[0]

    j = 1
    for k in range(len(origin_ind)):
        if (k == mode):
            pass

        J = 1
        for m in range(k):
            if (m == k) or (m == mode):
                pass
            J *= I[m]

        j += J * (Origin_ind[k] - 1)

    Flat_ind[1] = j
    return Flat_ind
```

---

## Alternating Least Squares (ALS) Method

Let $X \in \mathbb{R}^{I \times J \times K}$ be a third-order tensor. The goal is to compute a CP decomposition with $R$ components that best approximates $X$, i.e., to find:

$$\min_{X'} \|X - X'\| \quad \text{with} \quad X' = \sum_{r=1}^{R} \lambda_r a_r \circ b_r \circ c_r = [\![\lambda; A, B, C]\!]$$

where $A \in \mathbb{R}^{I \times R}$, $B \in \mathbb{R}^{J \times R}$, $C \in \mathbb{R}^{K \times R}$.

Having fixed all but one matrix, the problem reduces to a linear least-squares problem. For example, suppose that $B$ and $C$ are fixed. Then, we can rewrite the above minimization problem in matrix form as:

$$\min_{A'} \|X_{(1)} - A'(C \odot B)^T\|_F$$

where $A' = A \cdot \text{diag}(\lambda)$. The optimal solution is then given by:

$$X_{(1)} - A'(C \odot B)^T \approx 0$$

$$\Rightarrow A' = X_{(1)} \left[(C \odot B)^T\right]^\dagger = X_{(1)} \left[(C \odot B)^\dagger\right]^T$$

Since $M^\dagger = (M^T M)^\dagger M^T$:

$$\Rightarrow (C \odot B)^\dagger = \left[(C \odot B)^T (C \odot B)\right]^\dagger (C \odot B)^T$$

Since $\left[(C \odot B)^T (C \odot B)\right]^\dagger = (C^T C * B^T B)^\dagger$ (Hadamard product):

$$\therefore A' = X_{(1)} (C \odot B)(C^T C * B^T B)^\dagger$$

Finally, we normalize the columns of $A'$ to get $A$; in other words, let $\lambda_r = \|a_r'\|$ and $a_r = a_r' / \lambda_r$.

For $r = 1, \ldots, R$.

The factor matrices can be initialized in any way, such as randomly or by setting:

$$A^{(n)} = R \text{ leading left singular vectors of } X_{(n)} \text{ for } n = 1, \ldots, N$$

```
procedure CP-ALS(X, R)
    initialize A^(n) ∈ R^{I_n × R} for n = 1, ..., N
    repeat
        for n = 1, ..., N do
            V ← A^(1)ᵀA^(1) * ... * A^(n-1)ᵀA^(n-1) * A^(n+1)ᵀA^(n+1) * ... * A^(N)ᵀA^(N)
            A^(n) ← X_(n)(A^(N) ⊙ ... ⊙ A^(n+1) ⊙ A^(n-1) ⊙ ... ⊙ A^(1))V†
            normalize columns of A^(n) (storing norms as λ)
        end for
    until fit ceases to improve or maximum iterations exhausted
    return λ, A^(1), A^(2), ..., A^(N)
end procedure
```

---

## Other CP Algorithms

**(ASD)**: Performs optimization without flattening. Its accuracy is mathematically inferior to that of ALS

**(dGN)** Damped Gauss-Newton method: Mathematical Methodology.

**(PMF3)**: Uses dGN, suitable for real data with noise. Provides sparse correction to ensure stability. Optimize all at once without fixing the components. As the solution is approached, the convergence rate increases exponentially. High memory and computational costs are incurred.

---

## dGN-based PMF3

- The key feature of the PMF3 algorithm is that the log-penalty function ensures that no component becomes negative. Therefore, PMF3 is restricted to the complex plane.
- Excessive use: PMF3 uses a Hessian-based algorithm for large-scale matrix operations.
- Computation time on $C^3 \otimes C^3 \otimes C^3$ is not too long.
