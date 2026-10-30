# Spec sheet — `bench/tritonbench_g/parallel_retention_attention/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/parallel_retention_attention/parallel_retention_attention.py`

## Public theorem: `forward_fp32`

<details><summary>docstring</summary>

```
/-- Complete original accepted forward O source scope at fp32 storage.
The precision-selected dot/exp2/log2 backend remains an explicit binding. -/
```
</details>

**Statement:**
```lean
specification forward_fp32 {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s : State α) :
    ∃ final, FP.Structural.exec (withDot M D) (forwardKernel c) s = some final ∧
      final.mem = (outputMemory M c s (forwardValues M D c s)).mem
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `backward_fp32`

<details><summary>docstring</summary>

```
/-- Complete original accepted backward DK/DV stage, including descending
and diagonal loops and the full memory frame. -/
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
