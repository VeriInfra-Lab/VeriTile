# Spec sheet — `bench/tritonbench_g/int8_matmul_kernel/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/int8_matmul_kernel/int8_matmul_kernel.py`

## Public theorem: `int8_matmul_word_correctness`

<details><summary>docstring</summary>

```
/-- Full original four-field packed GEMM with arbitrary inner-loop extent.
Actual signed-byte narrowing and 32-bit wrap are part of the value model;
there is no no-overflow premise and no mathematical-Int dot fallback. -/
```
</details>

**Statement:**
```lean
specification int8_matmul_word_correctness {α : Type} [Inhabited α]
    (rest : Algebra α) (p : Parameters) (s : State α)
    (hBlock : 0 < p.blockInner) (hK : p.inner = 4*(p.blockInner*p.steps))
    (hInputs : Inputs p s)
    (hw : ∀ i j, active p s i → active p s j → outputOffset p s i = outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (backend rest) (source p) s = some final ∧
      Post p s final ∧
      ∀ f k, FP.IntegerDot.valid (α := α) (aValue p s f k) ∧
        FP.IntegerDot.valid (α := α) (bValue p s f k)
```

**Assumptions / layout contracts:**
- `hBlock : 0 < p.blockInner`
- `hK : p.inner = 4*(p.blockInner*p.steps)`
- `hInputs : Inputs p s`
- `hw : ∀ i j, active p s i → active p s j → outputOffset p s i = outputOffset p s j → i = j`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
