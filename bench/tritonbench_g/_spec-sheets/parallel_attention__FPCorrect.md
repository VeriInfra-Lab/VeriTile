# Spec sheet — `bench/tritonbench_g/parallel_attention/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/parallel_attention/parallel_attention.py`

## Public theorem: `forward_fp32`

<details><summary>docstring</summary>

```
/-- Both original forward loops, O and Z, and the exact memory frame. -/
```
</details>

**Statement:**
```lean
specification forward_fp32 {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s : State α) :
    ∃ final, FP.Structural.exec (withDot M D) (forwardKernel c) s = some final ∧
      final.mem = (forwardMemory M D c s).mem
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `backward_fp32`

<details><summary>docstring</summary>

```
/-- Both original backward DK/DV loops and the exact memory frame. -/
```
</details>

**Statement:**
```lean
specification backward_fp32 {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s : State α) :
    ∃ final, FP.Structural.exec (withDot M D) (backwardKernel c) s = some final ∧
      final.mem = (backwardOutputMemory M c s (backwardValues M D c s)).mem
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `forward_software_fp32`

<details><summary>docstring</summary>

```
/-- Full forward source instantiated with genuine fp32 scalar rounding,
ordered multiply/add dot and ordered sum. Scale=1 closes every used primitive
without depending on the supplied unused backend operations. -/
```
</details>

**Statement:**
```lean
specification forward_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (c : ForwardConfig)
    (s : State (FP.Value .fp32)) (hscale : c.scale=1) :
    ∃ final, FP.Structural.exec (withDot (sequentialFP32 rest cfg) (softwareDot cfg)) (forwardKernel c) s = some final ∧
      final.mem = (roundedOutputMemory c s (roundedForwardValues cfg c s)).mem
```

**Assumptions / layout contracts:**
- `hscale : c.scale=1`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `forward_nonempty_rounding`

<details><summary>docstring</summary>

```
/-- Nonempty two-loop interpreter witness: 1+2^-25 rounds back to 1.
This unit-tile witness is not a CUDA launch-validity claim. -/
```
</details>

**Statement:**
```lean
specification forward_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (withDot (sequentialFP32 rest .ieeeValue) (softwareDot .ieeeValue))
      (forwardKernel roundingConfig) roundingInitial = some final ∧
      final.mem "out" 1 = .mk .real one32 ∧
      final.mem "out" 0 = roundingInitial.mem "out" 0 ∧
      (1 + 1/33554432 : Rat) ≠ 1
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
