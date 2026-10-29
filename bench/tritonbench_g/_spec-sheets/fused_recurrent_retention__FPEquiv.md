# Spec sheet — `bench/tritonbench_g/fused_recurrent_retention/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/fused_recurrent_retention/fused_recurrent_retention.py`

## Public theorem: `fused_recurrent_retention_forward_equiv`

**Statement:**
```lean
specification fused_recurrent_retention_forward_equiv (dtype out_dtype : FloatDType)
    (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool) :
    forward Bool.false dtype out_dtype p r hasSeed storeFinal ≡[assumptions]
      forward Bool.true dtype out_dtype p r hasSeed storeFinal
```

**Closed-form spec defs (transitive):** `forward`, `assumptions`

<details><summary><code>forward</code></summary>

```lean
def forward (swapped : Bool) (dtype out_dtype : FloatDType) (p : Layout)
    (r : ForwardRegions) (hasSeed storeFinal : Bool) : Program where
  kernel := Full.Program.forward swapped dtype out_dtype p r hasSeed storeFinal
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

## Public theorem: `fused_recurrent_retention_backward_equiv`

**Statement:**
```lean
specification fused_recurrent_retention_backward_equiv (dtype out_dtype : FloatDType)
    (p : Layout) (r : BackwardRegions) (hasSeed : Bool) :
    backward Bool.false dtype out_dtype p r hasSeed ≡[assumptions]
      backward Bool.true dtype out_dtype p r hasSeed
```

**Assumptions / layout contracts:**
- `hasSeed : Bool`

**Closed-form spec defs (transitive):** `backward`, `assumptions`

<details><summary><code>backward</code></summary>

```lean
def backward (swapped : Bool) (dtype out_dtype : FloatDType) (p : Layout)
    (r : BackwardRegions) (hasSeed : Bool) : Program where
  kernel := Full.Program.backward swapped dtype out_dtype p r hasSeed
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>
