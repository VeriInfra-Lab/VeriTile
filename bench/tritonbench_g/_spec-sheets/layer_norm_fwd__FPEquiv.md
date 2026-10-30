# Spec sheet — `bench/tritonbench_g/layer_norm_fwd/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/layer_norm_fwd/layer_norm_fwd.py`

## Public theorem: `layer_norm_fwd_fp32_equiv`

**Statement:**
```lean
specification layer_norm_fwd_fp32_equiv (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    original f r stride M N eps p block hBlock ≡[assumptions]
      candidate f r stride M N eps p block hBlock
```

**Assumptions / layout contracts:**
- `hBlock : Random.ArangeSizeDomain block`

**Closed-form spec defs (transitive):** `original`, `assumptions`, `candidate`

<details><summary><code>original</code></summary>

```lean
def original (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) : Program where
  kernel := LayerNormFwd.original f r stride M N eps p block hBlock
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
def candidate (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) : Program where
  kernel := LayerNormFwd.candidate f r stride M N eps p block hBlock
  domain := []
```
</details>
