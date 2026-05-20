# # Day 1: 이것만 확실히 작동시키기
# import tensorly as tl
# from tensorly.decomposition import tucker
# import numpy as np

# # 1. Synthetic tensor generation
# X = np.random.randn(50, 50, 50)  # or load image

# # 2. TensorLy Tucker (Frobenius baseline)
# core, factors = tucker(X, rank=[10, 10, 10])
# X_hat = tl.tucker_to_tensor((core, factors))

# # 3. Basic metrics
# rel_error = np.linalg.norm(X - X_hat) / np.linalg.norm(X)

# # Day 2: Manual HOSVD + HOOI
# # Day 3: Rank sweep experiment  
# # Day 4: IRLS for L1 (핵심 구현)
# # Day 5-6: 실험 + 시각화
# # Day 7: 보고서

import tensorly as tl
from tensorly.decomposition import tucker
import numpy as np

# 1. Synthetic tensor
X = np.random.randn(50, 50, 50)
print(f"Tensor shape: {X.shape}, Total elements: {X.size}")

# 2. Tucker decomposition
core, factors = tucker(X, rank=[10, 10, 10])
X_hat = tl.tucker_to_tensor((core, factors))

print(f"Core shape: {core.shape}")
print(f"Factor shapes: {[f.shape for f in factors]}")

# 3. Metrics
rel_error = np.linalg.norm(X - X_hat) / np.linalg.norm(X)
max_error  = np.max(np.abs(X - X_hat))
n_original   = X.size
n_compressed = core.size + sum(f.size for f in factors)
compression  = n_original / n_compressed

print(f"\n=== Metrics ===")
print(f"Relative Error (F-norm): {rel_error:.4f}")
print(f"Max Error (L-inf):       {max_error:.4f}")
print(f"Compression Ratio:       {compression:.2f}x")
print(f"  Original params: {n_original}")
print(f"  Compressed params: {n_compressed}")