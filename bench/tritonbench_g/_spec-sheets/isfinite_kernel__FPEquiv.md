# Spec sheet — `bench/tritonbench_g/isfinite_kernel/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/isfinite_kernel/isfinite_kernel.py`

## Public theorem: `isfinite_fp32_equiv`

**Statement:**
```lean
specification isfinite_fp32_equiv (p : Parameters) :
    original32 p ≡[assumptions] indexCommuted32 p
```

**Closed-form spec defs (transitive):** `original32`, `assumptions`, `indexCommuted32`

<details><summary><code>original32</code></summary>

```lean
def original32 (p : Parameters) : Program where
  kernel := Kernels.original32 p
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>indexCommuted32</code></summary>

```lean
def indexCommuted32 (p : Parameters) : Program where
  kernel := Kernels.indexCommuted32 p
  domain := []
```
</details>

## Public theorem: `isfinite_fp64_equiv`

**Statement:**
```lean
specification isfinite_fp64_equiv (p : Parameters) :
    original64 p ≡[assumptions] indexCommuted64 p
```

**Closed-form spec defs (transitive):** `original64`, `assumptions`, `indexCommuted64`

<details><summary><code>original64</code></summary>

```lean
def original64 (p : Parameters) : Program where
  kernel := Kernels.original64 p
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>indexCommuted64</code></summary>

```lean
def indexCommuted64 (p : Parameters) : Program where
  kernel := Kernels.indexCommuted64 p
  domain := []
```
</details>
