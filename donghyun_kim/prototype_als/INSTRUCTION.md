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

### Test 1 — Exact Rank-2 (Sanity Check)
T를 정확히 rank-2로 생성 → ALS가 완벽하게 복원해야 함

| | Real | Complex |
|---|---|---|
| 통과율 | 10/10 | 10/10 |
| 평균 상대오차 | ~1e-8 | ~1e-7 |

**결론:** ALS가 exact rank-2 텐서를 수치 오차 수준으로 정확하게 복원함.

---

### Test 2 — Noisy Rank-2 (Stability)
T = (rank-2 텐서) + ε × (unit noise)

| noise ε | 평균 상대오차 (real) | 비고 |
|---|---|---|
| 0.01 | 0.0019 | ε의 약 1/5 |
| 0.05 | 0.0095 | ε에 비례 |
| 0.10 | 0.0189 | 오차 ≪ ε |
| 0.30 | 0.0567 | 안정적 |
| 0.50 | 0.0930 | |
| 1.00 | 0.1701 | |

**결론:** 오차가 노이즈 레벨보다 항상 작고, 노이즈에 비례하여 증가 → 알고리즘이 안정적.

---

### Test 3 — Random Tensor (Generic Input)
임의의 3×3×3 텐서 (rank 최대 5 이상)

| | Real | Complex |
|---|---|---|
| 평균 상대오차 | 45.5% | 50.3% |
| 범위 | 31.8% ~ 58.9% | 40.2% ~ 59.1% |

**결론:** Generic tensor를 rank-2로 근사하면 40~60% 오차가 정상. (rank-2는 전체 공간의 일부만 표현 가능)

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

수렴 조건: `|err_prev - err_curr| < 1e-8` 또는 최대 반복 도달

### 파라미터 권장값

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `n_restarts` | 10 | 랜덤 초기화 반복 횟수 (많을수록 global minimum 찾기 쉬움) |
| `n_iter_max` | 2000 | 최대 반복 수 |
| `tol` | 1e-8 | 수렴 판정 기준 |
