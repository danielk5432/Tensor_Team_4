# CP-ALS Rank-2 Approximation — 실행 가이드

## 문제 설명

임의의 텐서 T ∈ C³⊗C³⊗C³에 대해 Frobenius norm 기준으로 가장 가까운 rank-2 텐서 T'을 구한다.

```
T' = a₁⊗b₁⊗c₁ + a₂⊗b₂⊗c₂   (T' ∈ σ₂⁰(Seg(C³×C³×C³)))

minimize  ||T - T'||_F  =  sqrt(Σ |T_{ijk} - T'_{ijk}|²)
```

알고리즘: **CP-ALS (Alternating Least Squares)**  
각 factor matrix를 번갈아 고정하고 최소제곱법으로 업데이트, 수렴할 때까지 반복.

---

## 파일 구조

```
donghyun_kim/
├── cp_als.py       # 핵심 알고리즘 (CP-ALS)
├── main.py         # 기본 데모 (4가지 케이스)
├── test_suite.py   # 체계적 테스트 (3가지 카테고리)
└── INSTRUCTION.md  # 이 파일
```

---

## 실행 방법

> **주의:** Python 3.13 경로로 실행해야 패키지가 인식됨.

```
C:\Users\danie\AppData\Local\Programs\Python\Python313\python.exe <파일명>
```

### 1. 기본 데모

```
python main.py           # complex 텐서 (기본)
python main.py --real    # real 텐서만
python main.py --seed 7  # 랜덤 시드 지정
```

### 2. 테스트 스위트

```
python test_suite.py                    # 기본 (real, 10 trials)
python test_suite.py --trials 20        # 20번 반복
python test_suite.py --complex          # complex 텐서
python test_suite.py --restarts 20      # ALS 재시작 횟수 늘리기 (정확도↑, 속도↓)
python test_suite.py --noise-levels 0.01 0.1 0.5 2.0   # 노이즈 레벨 지정
```

### 3. 내 코드에서 직접 사용

```python
import numpy as np
from cp_als import rank2_approximation

# 임의의 3x3x3 텐서
T = np.random.randn(3, 3, 3)

# rank-2 근사
T_prime, factors, abs_err, rel_err = rank2_approximation(T)
A, B, C = factors   # 각각 shape (3, 2)

print(f"||T - T'||_F = {abs_err:.6f}")
print(f"상대오차 = {rel_err*100:.2f}%")
# T' = A[:,0]⊗B[:,0]⊗C[:,0] + A[:,1]⊗B[:,1]⊗C[:,1]
```

---

## 테스트 결과 요약

> **오차 정의** — 모든 테스트에서 사용하는 상대오차:
> ```
> 상대오차 (%) = ||T - T'||_F / ||T||_F × 100
> ```
> - `||T - T'||_F` : 근사 잔차의 Frobenius norm (얼마나 틀렸는가)
> - `||T||_F`      : 원본 텐서의 Frobenius norm (기준값, 정규화용)

---

### Test 1 — Exact Rank-2 (Sanity Check)
T를 정확히 rank-2로 생성 → ALS가 완벽하게 복원해야 함

| | Real | Complex |
|---|---|---|
| 통과율 | 10/10 | 10/10 |
| 평균 상대오차 | ~1e-6 % | ~1e-5 % |

**결론:** ALS가 exact rank-2 텐서를 수치 오차(부동소수점 한계) 수준으로 정확하게 복원함.

---

### Test 2 — Noisy Rank-2 (Stability)
T = (rank-2 텐서) + ε × (unit noise)

| noise ε | 평균 상대오차 (real) | 비고 |
|---|---|---|
| 0.01 | 0.19% | ε(1%)보다 훨씬 작음 |
| 0.05 | 0.95% | ε에 비례하여 증가 |
| 0.10 | 1.89% | |
| 0.30 | 5.67% | |
| 0.50 | 9.30% | |
| 1.00 | 17.01% | |

**결론:** 상대오차가 노이즈 크기(ε)보다 항상 작고 비례 증가 → 알고리즘이 안정적.

---

### Test 3 — Random Tensor (Generic Input)
임의의 3×3×3 텐서를 rank-2로 근사했을 때의 상대오차.

| | Real | Complex |
|---|---|---|
| 평균 상대오차 | 45.5% | 50.3% |
| 범위 | 31.8% ~ 58.9% | 40.2% ~ 59.1% |

**오차가 40~60%인 이유:**

3×3×3 텐서의 전체 공간은 27차원(=3×3×3)이다.  
rank-2 텐서 T' = a₁⊗b₁⊗c₁ + a₂⊗b₂⊗c₂는 파라미터 수가 최대 2×(3+3+3)=18개이지만,
scaling 자유도를 제거하면 실질적으로 **~14~16차원** 부분공간에 갇혀 있다.

즉, 27차원 공간 중 rank-2가 도달할 수 없는 차원이 ~11~13개 존재한다.  
랜덤 텐서는 이 27차원 공간에서 균등하게 분포하므로, rank-2 부분공간으로의 정사영
(projection)이 전체의 절반 수준밖에 안 되는 것이 자연스럽다.

이 오차는 알고리즘 실패가 아니라 **rank-2 표현의 수학적 한계**다.  
오차를 줄이려면 rank를 높여야 한다 (rank-3이면 ~20~25%, rank-4면 ~5% 미만 예상).

---

## 알고리즘 설명

### CP Decomposition

텐서 T를 rank-R 형태로 분해:
```
T' = Σᵣ aᵣ⊗bᵣ⊗cᵣ    (r = 1, ..., R)
```
Factor matrices: A = [a₁|a₂] ∈ C³ˣ², B = [b₁|b₂] ∈ C³ˣ², C = [c₁|c₂] ∈ C³ˣ²

### ALS 업데이트 (핵심)

Mode-n unfolding T₍ₙ₎를 이용해 각 factor를 순서대로 업데이트:

```
A ← T₍₀₎ · KR(B,C)* · conj(gram)⁻¹
B ← T₍₁₎ · KR(A,C)* · conj(gram)⁻¹
C ← T₍₂₎ · KR(A,B)* · conj(gram)⁻¹
```

- KR(B,C): Khatri-Rao product (column-wise Kronecker)
- gram = (AᴴA) ⊙ (BᴴB) ⊙ (CᴴC) (Hadamard product)
- 복소 텐서에서는 켤레복소수(*)가 필요 (실수면 자동으로 동일)

**수렴 조건:**  
1 iteration = A → B → C 순으로 한 번씩 업데이트.  
매 iteration마다 아래 값을 계산:
```
rel_err = ||T - T'||_F / ||T||_F
```
연속된 두 iteration 사이의 변화가 threshold 미만이면 수렴으로 판정하고 중단:
```
|rel_err_prev - rel_err_curr| < tol (기본값 1e-8)
```
수렴하지 않으면 `n_iter_max`(기본 2000)회에서 강제 종료.  
`n_restarts`(기본 10)번 랜덤 초기화를 반복하고, 그 중 `rel_err`가 가장 낮은 결과를 반환.

### 파라미터 권장값

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `n_restarts` | 10 | 랜덤 초기화 반복 횟수 (많을수록 global minimum 찾기 쉬움) |
| `n_iter_max` | 2000 | 최대 반복 수 |
| `tol` | 1e-8 | 수렴 판정 기준 |
