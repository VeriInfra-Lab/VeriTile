# Spec sheet — `bench/tritonbench_g/reversed_cumsum/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/reversed_cumsum/reversed_cumsum.py`

## Public theorem: `reversed_cumsum_compare_right_equiv`

**Statement:**
```lean
specification reversed_cumsum_compare_right_equiv (dtype out_dtype : FloatDType)
    (s o : RegionName) (sh st sd T S BT BS : Nat) :
    original dtype out_dtype s o sh st sd T S BT BS ≡[assumptions]
      candidate dtype out_dtype s o sh st sd T S BT BS
```

**Closed-form spec defs (transitive):** `original`, `assumptions`, `candidate`

<details><summary><code>original</code></summary>

```lean
def original (dtype out_dtype : FloatDType) (s o : RegionName)
    (sh st sd T S BT BS : Nat) : Program where
  kernel := TypedSource.original dtype out_dtype s o sh st sd T S BT BS
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>candidate</code></summary>

```lean
def candidate (dtype out_dtype : FloatDType) (s o : RegionName)
    (sh st sd T S BT BS : Nat) : Program where
  kernel := TypedSource.compareRight dtype out_dtype s o sh st sd T S BT BS
  domain := []
```
</details>
