# Spec sheet — `bench/tritonbench_g/layer_norm_fwd/Correct.lean`

**Python source:** `bench/tritonbench_g/layer_norm_fwd/layer_norm_fwd.py`

## Public theorem: `layer_norm_fwd_original_real`

**Statement:**
```lean
specification layer_norm_fwd_original_real (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    Spec.Real ((∀ initial : BlockState, N ≤ block → Domain.ReadWriteSeparated f r →
      ∃ final, exec (original f r stride M N eps p block hBlock).toAlgKernel initial = some final ∧
        Contract.Post f r stride M N eps p initial final) ∧
      (∀ initial : BlockState, N ≤ block → Domain.ReadWriteSeparated f r →
        ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)
          (original f r stride M N eps p block hBlock) initial
          (RealSurface.writes f r stride M N initial.pid)
          (RealSurface.expected f r stride M N eps p initial)) ∧
      RealIO.Implements (original f r stride M N eps p block hBlock).toAlgKernel
        f r stride M N eps p block)
```

**Assumptions / layout contracts:**
- `hBlock : Random.ArangeSizeDomain block`

**Closed-form spec defs (transitive):** `original`

<details><summary><code>original</code></summary>

```lean
def original (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) : ComputeKernel :=
  (LayerNormFwd.original f r stride M N eps p block hBlock).eraseDType
```
</details>

## Public theorem: `layer_norm_fwd_candidate_real`

**Statement:**
```lean
specification layer_norm_fwd_candidate_real (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    Spec.Real ((∀ initial : BlockState, N ≤ block → Domain.ReadWriteSeparated f r →
      ∃ final, exec (indexCommuted f r stride M N eps p block hBlock).toAlgKernel initial = some final ∧
        Contract.Post f r stride M N eps p initial final) ∧
      (∀ initial : BlockState, N ≤ block → Domain.ReadWriteSeparated f r →
        ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)
          (indexCommuted f r stride M N eps p block hBlock) initial
          (RealSurface.writes f r stride M N initial.pid)
          (RealSurface.expected f r stride M N eps p initial)) ∧
      RealIO.Implements (indexCommuted f r stride M N eps p block hBlock).toAlgKernel
        f r stride M N eps p block)
```

**Assumptions / layout contracts:**
- `hBlock : Random.ArangeSizeDomain block`

**Closed-form spec defs (transitive):** `indexCommuted`

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) : ComputeKernel :=
  (LayerNormFwd.candidate f r stride M N eps p block hBlock).eraseDType
```
</details>
