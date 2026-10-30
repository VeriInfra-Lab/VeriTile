# Spec sheet — `bench/tritonbench_g/isfinite_kernel/Correct.lean`

**Python source:** `bench/tritonbench_g/isfinite_kernel/isfinite_kernel.py`

## Public theorem: `original_fp32_correctness`

**Statement:**
```lean
specification original_fp32_correctness (p : Parameters) :
    Spec.Real (Contract32 (Kernels.original32 p) p)
```

**Closed-form spec defs (transitive):** `Contract32`, `original32`, `correct_target`

<details><summary><code>Contract32</code></summary>

```lean
def Contract32 (kernel : ComputeKernel) (p : Parameters) : Prop :=
  correct_target (fun rest => engine (backend32 rest)) finite32 kernel p
```
</details>

<details><summary><code>original32</code></summary>

```lean
abbrev original32 := Parametric.original32
```
</details>

<details><summary><code>correct_target</code></summary>

```
/-- Named typed correctness target: a successful complete execution and the
independent pointwise bit formula with every unwritten tagged cell preserved.
This target uses the caller's actual initial state and input realization. -/
```
```lean
def correct_target {α : Type} [Inhabited α] (model : Algebra α → Algebra α)
    (classify : α → Bool) (kernel : ComputeKernel) (p : Parameters) : Prop :=
  ∀ (rest : Algebra α) (bits : Nat → α) (initial : State α),
    p.input ≠ p.output → 0 < p.outputStride → InputBits p bits initial →
    ∃ final, FP.Structural.exec (model rest) kernel initial = some final ∧
      Post classify p bits initial final
```
</details>

## Public theorem: `original_fp64_correctness`

**Statement:**
```lean
specification original_fp64_correctness (p : Parameters) :
    Spec.Real (Contract64 (Kernels.original64 p) p)
```

**Closed-form spec defs (transitive):** `Contract64`, `original64`, `correct_target`

<details><summary><code>Contract64</code></summary>

```lean
def Contract64 (kernel : ComputeKernel) (p : Parameters) : Prop :=
  correct_target (fun rest => engine (backend64 rest)) finite64 kernel p
```
</details>

<details><summary><code>original64</code></summary>

```lean
abbrev original64 := Parametric.original64
```
</details>

<details><summary><code>correct_target</code></summary>

```
/-- Named typed correctness target: a successful complete execution and the
independent pointwise bit formula with every unwritten tagged cell preserved.
This target uses the caller's actual initial state and input realization. -/
```
```lean
def correct_target {α : Type} [Inhabited α] (model : Algebra α → Algebra α)
    (classify : α → Bool) (kernel : ComputeKernel) (p : Parameters) : Prop :=
  ∀ (rest : Algebra α) (bits : Nat → α) (initial : State α),
    p.input ≠ p.output → 0 < p.outputStride → InputBits p bits initial →
    ∃ final, FP.Structural.exec (model rest) kernel initial = some final ∧
      Post classify p bits initial final
```
</details>

## Public theorem: `index_commuted_fp32_correctness`

**Statement:**
```lean
specification index_commuted_fp32_correctness (p : Parameters) :
    Spec.Real (Contract32 (Kernels.indexCommuted32 p) p)
```

**Closed-form spec defs (transitive):** `Contract32`, `indexCommuted32`, `correct_target`, `candidate32`, `candidateSource32`

<details><summary><code>Contract32</code></summary>

```lean
def Contract32 (kernel : ComputeKernel) (p : Parameters) : Prop :=
  correct_target (fun rest => engine (backend32 rest)) finite32 kernel p
```
</details>

<details><summary><code>indexCommuted32</code></summary>

```lean
abbrev indexCommuted32 := Parametric.candidate32
```
</details>

<details><summary><code>correct_target</code></summary>

```
/-- Named typed correctness target: a successful complete execution and the
independent pointwise bit formula with every unwritten tagged cell preserved.
This target uses the caller's actual initial state and input realization. -/
```
```lean
def correct_target {α : Type} [Inhabited α] (model : Algebra α → Algebra α)
    (classify : α → Bool) (kernel : ComputeKernel) (p : Parameters) : Prop :=
  ∀ (rest : Algebra α) (bits : Nat → α) (initial : State α),
    p.input ≠ p.output → 0 < p.outputStride → InputBits p bits initial →
    ∃ final, FP.Structural.exec (model rest) kernel initial = some final ∧
      Post classify p bits initial final
```
</details>

<details><summary><code>candidate32</code></summary>

```lean
def candidate32 (p : Parameters) : ComputeKernel :=
  candidateSource32 p.input p.output p.inputStride p.inputOrder
    p.outputStride p.outputOrder p.size p.numTasks p.iterations p.block
    p.oneTile p.inputOrderZero p.outputOrderZero
```
</details>

<details><summary><code>candidateSource32</code></summary>

```lean
def candidateSource32 := isfiniteCandidate%[tl.float32]
```
</details>

## Public theorem: `index_commuted_fp64_correctness`

**Statement:**
```lean
specification index_commuted_fp64_correctness (p : Parameters) :
    Spec.Real (Contract64 (Kernels.indexCommuted64 p) p)
```

**Closed-form spec defs (transitive):** `Contract64`, `indexCommuted64`, `correct_target`, `candidate64`, `candidateSource64`

<details><summary><code>Contract64</code></summary>

```lean
def Contract64 (kernel : ComputeKernel) (p : Parameters) : Prop :=
  correct_target (fun rest => engine (backend64 rest)) finite64 kernel p
```
</details>

<details><summary><code>indexCommuted64</code></summary>

```lean
abbrev indexCommuted64 := Parametric.candidate64
```
</details>

<details><summary><code>correct_target</code></summary>

```
/-- Named typed correctness target: a successful complete execution and the
independent pointwise bit formula with every unwritten tagged cell preserved.
This target uses the caller's actual initial state and input realization. -/
```
```lean
def correct_target {α : Type} [Inhabited α] (model : Algebra α → Algebra α)
    (classify : α → Bool) (kernel : ComputeKernel) (p : Parameters) : Prop :=
  ∀ (rest : Algebra α) (bits : Nat → α) (initial : State α),
    p.input ≠ p.output → 0 < p.outputStride → InputBits p bits initial →
    ∃ final, FP.Structural.exec (model rest) kernel initial = some final ∧
      Post classify p bits initial final
```
</details>

<details><summary><code>candidate64</code></summary>

```lean
def candidate64 (p : Parameters) : ComputeKernel :=
  candidateSource64 p.input p.output p.inputStride p.inputOrder
    p.outputStride p.outputOrder p.size p.numTasks p.iterations p.block
    p.oneTile p.inputOrderZero p.outputOrderZero
```
</details>

<details><summary><code>candidateSource64</code></summary>

```lean
def candidateSource64 := isfiniteCandidate%[tl.float64]
```
</details>
