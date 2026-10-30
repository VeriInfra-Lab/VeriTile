# Spec sheet — `bench/tritonbench_g/reversed_cumsum_scalar/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/reversed_cumsum_scalar/reversed_cumsum_scalar.py`

## Public theorem: `reversed_cumsum_scalar_equiv`

**Statement:**
```lean
specification reversed_cumsum_scalar_equiv (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) :
    original dtype out_dtype s o T BT ≡[assumptions] indexCommuted dtype out_dtype s o T BT
```

**Closed-form spec defs (transitive):** `original`, `assumptions`, `indexCommuted`

<details><summary><code>original</code></summary>

```lean
def original (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) : Program where
  kernel := Kernels.original dtype out_dtype s o T BT
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) : Program where
  kernel := Kernels.indexCommuted dtype out_dtype s o T BT
  domain := []
```
</details>
