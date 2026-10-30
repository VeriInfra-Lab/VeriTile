# Spec sheet — `bench/tritonbench_g/reversed_cumsum/Correct.lean`

**Python source:** `bench/tritonbench_g/reversed_cumsum/reversed_cumsum.py`

## Public theorem: `reversed_cumsum_original_real`

**Statement:**
```lean
specification reversed_cumsum_original_real (dtype out_dtype : FloatDType)
    (p : Layout) (s o : RegionName) :
    Spec.Real (Implements (kernel Bool.false dtype out_dtype p s o) p s o)
```

**Closed-form spec defs (transitive):** `Implements`, `kernel`

<details><summary><code>Implements</code></summary>

```
/-- A complete per-program contract. The output is defined from initial input
and padding values, never from a prepared intermediate tile or a final readback.
The finite clause preserves every physical cell outside active writes. -/
```
```lean
def Implements (k : ComputeKernel) (p : Layout) (s o : RegionName) : Prop :=
  (∀ initial : BlockState, s ≠ o → 0 < p.BT → NoCollision p (RealInit.launch initial) →
    ∃ final, exec k.toAlgKernel initial = some final ∧ RealExecution.Post p s o initial final) ∧
  (∀ initial : BlockState, s ≠ o → 0 < p.BT → NoCollision p (RealInit.launch initial) →
    ComputeCorrect.RealizesFrame_without_Rounding k initial
      (RealSurface.writes p (RealInit.launch initial) o) (RealSurface.expectedCell p s initial)) ∧
  (∀ (A : FlatAlloc), A.Disjoint → (∀ r, r ∉ A.regions → A.extent r = 0) →
    ∀ initial : BlockState, s ≠ o → 0 < p.BT → NoCollision p (RealInit.launch initial) →
      RealSafety.Bounds p (RealInit.launch initial) s o A.extent →
      initial.undef = (fun _ _ => 0) →
      ∃ final, exec (A.flattenKernel k.toAlgKernel) (A.flattenState initial) = some final ∧
        RealIO.Post A p s o initial final)
```
</details>

<details><summary><code>kernel</code></summary>

```lean
def kernel (swapped : Bool) (dtype out_dtype : FloatDType)
    (p : Layout) (s o : RegionName) : ComputeKernel :=
  (RealSource.source swapped dtype out_dtype s o p.headStride p.timeStride p.channelStride
    p.T p.S p.BT p.BS).eraseDType
```
</details>

## Public theorem: `reversed_cumsum_candidate_real`

**Statement:**
```lean
specification reversed_cumsum_candidate_real (dtype out_dtype : FloatDType)
    (p : Layout) (s o : RegionName) :
    Spec.Real (Implements (kernel Bool.true dtype out_dtype p s o) p s o)
```

**Closed-form spec defs (transitive):** `Implements`, `kernel`

<details><summary><code>Implements</code></summary>

```
/-- A complete per-program contract. The output is defined from initial input
and padding values, never from a prepared intermediate tile or a final readback.
The finite clause preserves every physical cell outside active writes. -/
```
```lean
def Implements (k : ComputeKernel) (p : Layout) (s o : RegionName) : Prop :=
  (∀ initial : BlockState, s ≠ o → 0 < p.BT → NoCollision p (RealInit.launch initial) →
    ∃ final, exec k.toAlgKernel initial = some final ∧ RealExecution.Post p s o initial final) ∧
  (∀ initial : BlockState, s ≠ o → 0 < p.BT → NoCollision p (RealInit.launch initial) →
    ComputeCorrect.RealizesFrame_without_Rounding k initial
      (RealSurface.writes p (RealInit.launch initial) o) (RealSurface.expectedCell p s initial)) ∧
  (∀ (A : FlatAlloc), A.Disjoint → (∀ r, r ∉ A.regions → A.extent r = 0) →
    ∀ initial : BlockState, s ≠ o → 0 < p.BT → NoCollision p (RealInit.launch initial) →
      RealSafety.Bounds p (RealInit.launch initial) s o A.extent →
      initial.undef = (fun _ _ => 0) →
      ∃ final, exec (A.flattenKernel k.toAlgKernel) (A.flattenState initial) = some final ∧
        RealIO.Post A p s o initial final)
```
</details>

<details><summary><code>kernel</code></summary>

```lean
def kernel (swapped : Bool) (dtype out_dtype : FloatDType)
    (p : Layout) (s o : RegionName) : ComputeKernel :=
  (RealSource.source swapped dtype out_dtype s o p.headStride p.timeStride p.channelStride
    p.T p.S p.BT p.BS).eraseDType
```
</details>
