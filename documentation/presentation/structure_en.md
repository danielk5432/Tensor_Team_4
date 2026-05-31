# Low-Rank Tensor Approximation and Image Inpainting

## 1. Introduction

### Motivation

* Many real-world datasets naturally appear as tensors rather than matrices.
* Examples include RGB images, videos, hyperspectral images, and scientific measurements.
* Low-rank tensor decomposition provides compact representations while preserving essential structures.

### Project Objective

Given a tensor

T ∈ ℂ³ ⊗ ℂ³ ⊗ ℂ³

find a low-rank approximation minimizing reconstruction error.

The original project problem focuses on rank-2 approximation using CP decomposition, while this work further investigates tensor decomposition techniques and their application to image inpainting.

### Contributions

* Implementation of CP-ALS decomposition
* Implementation of Tucker decomposition
* Mathematical validation of low-rank approximation
* Analysis of convergence and robustness
* Application to image restoration and inpainting

---

## 2. Low-Rank Tensor Decomposition

### CP Decomposition

Canonical Polyadic (CP) decomposition approximates a tensor as a sum of rank-one tensors

X ≈ Σᵣ aᵣ ⊗ bᵣ ⊗ cᵣ

Characteristics:

* Compact representation
* Direct interpretation through tensor rank
* Typically optimized using Alternating Least Squares (ALS)

### Tucker Decomposition

Tucker decomposition represents a tensor using a smaller core tensor and factor matrices

X ≈ G ×₁ A ×₂ B ×₃ C

Characteristics:

* Multi-linear rank representation
* More flexible than CP decomposition
* Commonly optimized through HOSVD/HOOI

### Performance Metrics

The following metrics are used throughout the study:

* Relative Reconstruction Error
* Frobenius Norm Error
* Peak Signal-to-Noise Ratio (PSNR)
* Structural Similarity Index (SSIM)
* Runtime and Convergence Iterations

---

## 3. Mathematical Analysis

### CP-ALS Formulation

The approximation problem is formulated as

min ||X - X̂||²_F

where X̂ is represented by a rank-R CP model.

ALS iteratively updates one factor matrix while fixing the remaining factors.

Update sequence:

1. Update A while fixing B and C
2. Update B while fixing A and C
3. Update C while fixing A and B

### Mathematical Validation

Experiments include:

* Exact Rank-2 Tensor Recovery
* Noisy Rank-2 Tensor Recovery
* Generic Random Tensor Approximation

These experiments verify the correctness of the implementation and the effectiveness of the optimization procedure.

### Performance Analysis

Analysis focuses on:

* Convergence behavior
* Initialization sensitivity
* Noise robustness
* Reconstruction accuracy

The resulting convergence curves and sensitivity studies reveal the stability characteristics of CP-ALS.

---

## 4. Applications: Image Inpainting

### Experimental Environment

Dataset:

* Lena
* Baboon
* Peppers
* House
* Boats
* Airplane

Image representation:

RGB Image → H × W × 3 Tensor

Artificial masks are applied to simulate missing pixels.

### Ground Truth and Implementations

Two tensor decomposition approaches are evaluated:

#### CP-based Inpainting

Pipeline:

Original Image
→ Masked Image
→ CP-ALS Reconstruction
→ Restored Image

#### Tucker-based Inpainting

Pipeline:

Original Image
→ Masked Image
→ Tucker Reconstruction
→ Restored Image

### Performance Analysis

Evaluation metrics:

* Relative Error
* PSNR
* SSIM

Experiments include:

* Rank sweep analysis
* Noise robustness analysis
* Visual reconstruction comparison

Results demonstrate the influence of tensor rank selection and decomposition method on reconstruction quality.

---

## 5. Conclusion

### Summary

* Implemented low-rank tensor decomposition algorithms from scratch.
* Investigated CP and Tucker tensor models.
* Verified numerical correctness through mathematical validation experiments.
* Analyzed convergence, sensitivity, and robustness properties.
* Successfully applied tensor decomposition to image inpainting.

### Key Findings

* CP decomposition provides compact low-rank representations.
* Tucker decomposition generally offers greater reconstruction flexibility.
* Low-rank tensor models effectively recover missing image information.
* Tensor decomposition serves as a practical framework for image restoration tasks.

### Future Work

* Adaptive rank selection
* Tensor completion under structured missing patterns
* Robust tensor decomposition under heavy noise
* Extension to video and hyperspectral datasets