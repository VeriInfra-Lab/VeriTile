# Spec sheet — `bench/tritonbench_g/reversed_cumsum_scalar/Correct.lean`

**Python source:** `bench/tritonbench_g/reversed_cumsum_scalar/reversed_cumsum_scalar.py`

## Public theorem: `reversed_cumsum_scalar_correctness`

**Statement:**
```lean
specification reversed_cumsum_scalar_correctness (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) :
    Spec.Real (Contract (original dtype out_dtype s o T BT) T BT s o)
```

**Closed-form spec defs (transitive):** `Contract`, `original`

<details><summary><code>Contract</code></summary>

```lean
def Contract (kernel : ComputeKernel) (T BT : Nat) (s o : RegionName) : Prop :=
  (∀ initial : BlockState, s ≠ o → 0 < BT →
    ∃ final, exec kernel.toAlgKernel initial = some final ∧
      RealExecution.Post T BT s o initial final) ∧
  (∀ initial : BlockState, s ≠ o → 0 < BT →
    ComputeCorrect.RealizesFrame_without_Rounding kernel initial
      (RealSurface.writes T (initial.pids 0) o) (RealSurface.expectedCell T BT s initial)) ∧
  (∀ (A : FlatAlloc), A.Disjoint → (∀ r, r ∉ A.regions → A.extent r = 0) →
    ∀ initial : BlockState, s ≠ o → 0 < BT →
      RealSafety.Bounds T (initial.pids 0) s o A.extent →
      initial.undef = (fun _ _ => 0) →
      ∃ final, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState initial) = some final ∧
        RealIO.Post A T s o initial final)
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) : ComputeKernel :=
  (Kernels.original dtype out_dtype s o T BT).eraseDType
```
</details>

## Public theorem: `index_commuted_correctness`

**Statement:**
```lean
specification index_commuted_correctness (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) :
    Spec.Real (Contract (indexCommuted dtype out_dtype s o T BT) T BT s o)
```

**Closed-form spec defs (transitive):** `Contract`, `indexCommuted`

<details><summary><code>Contract</code></summary>

```lean
def Contract (kernel : ComputeKernel) (T BT : Nat) (s o : RegionName) : Prop :=
  (∀ initial : BlockState, s ≠ o → 0 < BT →
    ∃ final, exec kernel.toAlgKernel initial = some final ∧
      RealExecution.Post T BT s o initial final) ∧
  (∀ initial : BlockState, s ≠ o → 0 < BT →
    ComputeCorrect.RealizesFrame_without_Rounding kernel initial
      (RealSurface.writes T (initial.pids 0) o) (RealSurface.expectedCell T BT s initial)) ∧
  (∀ (A : FlatAlloc), A.Disjoint → (∀ r, r ∉ A.regions → A.extent r = 0) →
    ∀ initial : BlockState, s ≠ o → 0 < BT →
      RealSafety.Bounds T (initial.pids 0) s o A.extent →
      initial.undef = (fun _ _ => 0) →
      ∃ final, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState initial) = some final ∧
        RealIO.Post A T s o initial final)
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) : ComputeKernel :=
  (Kernels.indexCommuted dtype out_dtype s o T BT).eraseDType
```
</details>
