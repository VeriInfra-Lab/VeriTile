# Spec sheet — `bench/tritonbench_g/fused_recurrent_retention/Correct.lean`

**Python source:** `bench/tritonbench_g/fused_recurrent_retention/fused_recurrent_retention.py`

## Public theorem: `fused_recurrent_retention_forward_original_real`

**Statement:**
```lean
specification fused_recurrent_retention_forward_original_real (p : Layout) (r : ForwardRegions)
    (hasSeed storeFinal : Bool) :
    Spec.Real (ForwardImplements (ForwardSource.kernel p r hasSeed storeFinal Bool.false)
      p r hasSeed storeFinal)
```

**Closed-form spec defs (transitive):** `ForwardImplements`

<details><summary><code>ForwardImplements</code></summary>

```lean
def ForwardImplements (kernel : ComputeKernel) (p : Layout) (r : ForwardRegions)
    (hasSeed storeFinal : Bool) : Prop :=
  (∀ initial : BlockState, r.q ≠ r.o → r.k ≠ r.o → r.v ≠ r.o →
    (storeFinal = Bool.true → r.o ≠ r.final) →
    ∃ final, exec kernel.toAlgKernel initial = some final ∧
      ForwardReal.Post p r hasSeed storeFinal initial final) ∧
  (∀ initial : BlockState, r.q ≠ r.o → r.k ≠ r.o → r.v ≠ r.o →
    (storeFinal = Bool.true → r.o ≠ r.final) →
    ComputeCorrect.RealizesFrame_without_Rounding kernel initial
      (ForwardSurface.writes p (launch initial) r storeFinal) (ForwardSurface.expected p r hasSeed initial)) ∧
  (∀ A : FlatAlloc, A.Disjoint → (∀ q, q ∉ A.regions → A.extent q = 0) →
    ∀ initial : BlockState, r.q ≠ r.o → r.k ≠ r.o → r.v ≠ r.o →
      (storeFinal = Bool.true → r.o ≠ r.final) →
      SafetyAccess.Bounds p (launch initial) r hasSeed storeFinal A.extent →
      ∃ final, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState initial) = some final ∧
        ForwardIO.Post A p r hasSeed storeFinal initial final)
```
</details>

## Public theorem: `fused_recurrent_retention_forward_candidate_real`

**Statement:**
```lean
specification fused_recurrent_retention_forward_candidate_real (p : Layout) (r : ForwardRegions)
    (hasSeed storeFinal : Bool) :
    Spec.Real (ForwardImplements (ForwardSource.kernel p r hasSeed storeFinal Bool.true)
      p r hasSeed storeFinal)
```

**Closed-form spec defs (transitive):** `ForwardImplements`

<details><summary><code>ForwardImplements</code></summary>

```lean
def ForwardImplements (kernel : ComputeKernel) (p : Layout) (r : ForwardRegions)
    (hasSeed storeFinal : Bool) : Prop :=
  (∀ initial : BlockState, r.q ≠ r.o → r.k ≠ r.o → r.v ≠ r.o →
    (storeFinal = Bool.true → r.o ≠ r.final) →
    ∃ final, exec kernel.toAlgKernel initial = some final ∧
      ForwardReal.Post p r hasSeed storeFinal initial final) ∧
  (∀ initial : BlockState, r.q ≠ r.o → r.k ≠ r.o → r.v ≠ r.o →
    (storeFinal = Bool.true → r.o ≠ r.final) →
    ComputeCorrect.RealizesFrame_without_Rounding kernel initial
      (ForwardSurface.writes p (launch initial) r storeFinal) (ForwardSurface.expected p r hasSeed initial)) ∧
  (∀ A : FlatAlloc, A.Disjoint → (∀ q, q ∉ A.regions → A.extent q = 0) →
    ∀ initial : BlockState, r.q ≠ r.o → r.k ≠ r.o → r.v ≠ r.o →
      (storeFinal = Bool.true → r.o ≠ r.final) →
      SafetyAccess.Bounds p (launch initial) r hasSeed storeFinal A.extent →
      ∃ final, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState initial) = some final ∧
        ForwardIO.Post A p r hasSeed storeFinal initial final)
```
</details>

## Public theorem: `fused_recurrent_retention_backward_original_real`

**Statement:**
```lean
specification fused_recurrent_retention_backward_original_real (p : Layout) (r : BackwardRegions)
    (hasSeed : Bool) :
    Spec.Real (BackwardImplements (BackwardSource.kernel p r hasSeed Bool.false)
      p r hasSeed)
```

**Assumptions / layout contracts:**
- `hasSeed : Bool`

**Closed-form spec defs (transitive):** `BackwardImplements`

<details><summary><code>BackwardImplements</code></summary>

```lean
def BackwardImplements (kernel : ComputeKernel) (p : Layout) (r : BackwardRegions)
    (hasSeed : Bool) : Prop :=
  (∀ initial : BlockState, BackwardReal.Disjoint r →
    ∃ final, exec kernel.toAlgKernel initial = some final ∧ BackwardReal.Post p r hasSeed initial final) ∧
  (∀ initial : BlockState, BackwardReal.Disjoint r →
    ComputeCorrect.RealizesFrame_without_Rounding kernel initial
      (BackwardSurface.writes p (launch initial) r) (BackwardSurface.expected p r hasSeed initial)) ∧
  (∀ A : FlatAlloc, A.Disjoint → (∀ q, q ∉ A.regions → A.extent q = 0) →
    ∀ initial : BlockState, BackwardReal.Disjoint r → BackwardSafety.Bounds p (launch initial) r hasSeed A.extent →
      ∃ final, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState initial) = some final ∧
        BackwardIO.Post A p r hasSeed initial final)
```
</details>

## Public theorem: `fused_recurrent_retention_backward_candidate_real`

**Statement:**
```lean
specification fused_recurrent_retention_backward_candidate_real (p : Layout) (r : BackwardRegions)
    (hasSeed : Bool) :
    Spec.Real (BackwardImplements (BackwardSource.kernel p r hasSeed Bool.true)
      p r hasSeed)
```

**Assumptions / layout contracts:**
- `hasSeed : Bool`

**Closed-form spec defs (transitive):** `BackwardImplements`

<details><summary><code>BackwardImplements</code></summary>

```lean
def BackwardImplements (kernel : ComputeKernel) (p : Layout) (r : BackwardRegions)
    (hasSeed : Bool) : Prop :=
  (∀ initial : BlockState, BackwardReal.Disjoint r →
    ∃ final, exec kernel.toAlgKernel initial = some final ∧ BackwardReal.Post p r hasSeed initial final) ∧
  (∀ initial : BlockState, BackwardReal.Disjoint r →
    ComputeCorrect.RealizesFrame_without_Rounding kernel initial
      (BackwardSurface.writes p (launch initial) r) (BackwardSurface.expected p r hasSeed initial)) ∧
  (∀ A : FlatAlloc, A.Disjoint → (∀ q, q ∉ A.regions → A.extent q = 0) →
    ∀ initial : BlockState, BackwardReal.Disjoint r → BackwardSafety.Bounds p (launch initial) r hasSeed A.extent →
      ∃ final, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState initial) = some final ∧
        BackwardIO.Post A p r hasSeed initial final)
```
</details>
