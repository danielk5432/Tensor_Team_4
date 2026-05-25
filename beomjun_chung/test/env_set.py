"""
Environment verification + quick GT validation.
Run: python test/env_set.py
"""
import sys
import importlib

REQUIRED = {
    "jax": ">=0.4.0",
    "jaxlib": ">=0.4.0",
    "tensorly": ">=0.8.0",
    "numpy": ">=1.24.0",
    "PIL": ">=9.0.0",
    "skimage": ">=0.21.0",
    "matplotlib": ">=3.7.0",
    "optax": ">=0.1.0",
}

IMPORT_MAP = {
    "PIL": "PIL",
    "skimage": "skimage",
}

def check_imports():
    print("=== Dependency Check ===")
    all_ok = True
    for pkg, ver_req in REQUIRED.items():
        mod_name = IMPORT_MAP.get(pkg, pkg)
        try:
            mod = importlib.import_module(mod_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {pkg} {version}  (required {ver_req})")
        except ImportError:
            print(f"  [FAIL] {pkg} not found  (required {ver_req})")
            all_ok = False
    return all_ok


def check_jax():
    print("\n=== JAX Basic Test ===")
    import jax
    import jax.numpy as jnp
    x = jnp.array([1.0, 2.0, 3.0])
    print(f"  jnp.sum([1,2,3]) = {float(jnp.sum(x))}")
    print(f"  JAX devices: {jax.devices()}")
    print("  [OK]")


def check_tensor_ops():
    print("\n=== tensor_ops Unit Tests ===")
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import numpy as np
    import jax.numpy as jnp
    from src.utils.tensor_ops import unfold, fold, mode_n_product, multi_mode_product, khatri_rao

    X = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)

    # Test unfold / fold round-trip
    for mode in range(3):
        X_unf = unfold(X, mode)
        X_back = fold(X_unf, mode, X.shape)
        err = float(jnp.max(jnp.abs(X - X_back)))
        assert err < 1e-5, f"fold(unfold(X, {mode})) != X, err={err}"
    print("  [OK] unfold/fold round-trip")

    # Test mode_n_product shape
    M = jnp.ones((5, 3))
    Y = mode_n_product(X, M, 1)
    assert Y.shape == (2, 5, 4), f"Expected (2,5,4), got {Y.shape}"
    print("  [OK] mode_n_product shape")

    # Test khatri_rao shape
    A = jnp.ones((3, 4))
    B = jnp.ones((5, 4))
    KR = khatri_rao(A, B)
    assert KR.shape == (15, 4), f"Expected (15,4), got {KR.shape}"
    print("  [OK] khatri_rao shape")

    # Test multi_mode_product (Tucker reconstruction)
    X_rand = jnp.array(np.random.randn(4, 5, 3).astype(np.float32))
    G = jnp.array(np.random.randn(2, 3, 2).astype(np.float32))
    factors = [
        jnp.array(np.random.randn(4, 2).astype(np.float32)),
        jnp.array(np.random.randn(5, 3).astype(np.float32)),
        jnp.array(np.random.randn(3, 2).astype(np.float32)),
    ]
    Y = multi_mode_product(G, factors)
    assert Y.shape == (4, 5, 3), f"Expected (4,5,3), got {Y.shape}"
    print("  [OK] multi_mode_product shape")


def gt_validation():
    print("\n=== GT Validation ===")
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import numpy as np
    import jax.numpy as jnp
    import tensorly as tl
    tl.set_backend("numpy")

    from src.decomposition import hosvd, hooi, cp_als

    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 20, 3)).astype(np.float32)
    X_jnp = jnp.array(X)
    rank_t = [5, 5, 2]
    rank_cp = 5

    # Tucker GT
    core_gt, factors_gt = tl.decomposition.tucker(X, rank=rank_t, init="svd", n_iter_max=500)
    X_gt = tl.tucker_to_tensor((core_gt, factors_gt))
    err_gt_tucker = np.linalg.norm(X - X_gt) / np.linalg.norm(X)

    r_hs = hosvd.decompose(X_jnp, rank_t)
    err_hs = np.linalg.norm(X - np.array(hosvd.reconstruct(r_hs))) / np.linalg.norm(X)

    r_ho, _ = hooi.decompose(X_jnp, rank_t, max_iter=500)
    err_ho = np.linalg.norm(X - np.array(hooi.reconstruct(r_ho))) / np.linalg.norm(X)

    # Tensorly HOSVD (n_iter_max=0 = initialization only, same as HOSVD)
    core_hs_gt, factors_hs_gt = tl.decomposition.tucker(
        X, rank=rank_t, init="svd", n_iter_max=0
    )
    X_gt_hs = tl.tucker_to_tensor((core_hs_gt, factors_hs_gt))
    err_gt_hosvd = np.linalg.norm(X - X_gt_hs) / np.linalg.norm(X)

    print(f"  Tucker GT (HOOI) : {err_gt_tucker:.6f}")
    print(f"  Tucker GT (HOSVD): {err_gt_hosvd:.6f}")
    print(f"  HOSVD  error     : {err_hs:.6f}  diff_from_hosvd_gt={abs(err_hs-err_gt_hosvd):.6f}")
    print(f"  HOOI   error     : {err_ho:.6f}  diff_from_hooi_gt={abs(err_ho-err_gt_tucker):.6f}")
    # HOSVD compared against tensorly HOSVD (not HOOI) — both are initializations
    assert abs(err_hs - err_gt_hosvd) < 0.05, "HOSVD failed GT check"
    # HOOI should match tensorly HOOI closely
    assert abs(err_ho - err_gt_tucker) < 0.05, "HOOI  failed GT check"
    print("  [OK] Tucker GT within 5%")

    # CP GT
    weights_gt, factors_gt_cp = tl.decomposition.parafac(X, rank=rank_cp, init="svd", n_iter_max=500)
    X_gt_cp = tl.cp_to_tensor((weights_gt, factors_gt_cp))
    err_gt_cp = np.linalg.norm(X - X_gt_cp) / np.linalg.norm(X)

    r_cp, _ = cp_als.decompose(X_jnp, rank_cp, max_iter=500)
    err_cp = np.linalg.norm(X - np.array(cp_als.reconstruct(r_cp))) / np.linalg.norm(X)

    print(f"\n  CP GT error      : {err_gt_cp:.6f}")
    print(f"  CP-ALS error     : {err_cp:.6f}")
    print("  [OK] CP-ALS ran successfully")


if __name__ == "__main__":
    ok = check_imports()
    if not ok:
        print("\nSome dependencies missing. Install with:")
        print("  pip install jax jaxlib tensorly numpy Pillow scikit-image matplotlib optax")
        sys.exit(1)
    check_jax()
    check_tensor_ops()
    gt_validation()
    print("\n=== All checks passed ===")
