# Spec sheet — `bench/tritonbench_g/matmul_dequant_int4/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/matmul_dequant_int4/matmul_dequant_int4.py`

## Public theorem: `matmul_dequant_int4_fp16_correctness`

<details><summary>docstring</summary>

```
/-- The original standalone dequantize stage, with genuine signed int32 input
words, fixed-width nibble semantics and concrete binary16 conversion/multiply.
The active address injectivity and input representation are ordinary input
conditions. The independent packed-bit result is derived from initial memory. -/
```
</details>

**Statement:**
```lean
specification matmul_dequant_int4_fp16_correctness [Inhabited Binary16.Value]
    (rest : Algebra Binary16.Value) (p : FPTyped.Parameters) (s : State Binary16.Value)
    (hInputs : FPTyped.PackedInputs p s)
    (hw : ∀ i j, FPTyped.active p s i → FPTyped.active p s j →
      FPTyped.outputOffset p s i = FPTyped.outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (FPTyped.software rest) (FPTyped.source p) s = some final ∧
      FPTyped.Post p s final (FPTyped.roundedValue p s) ∧
      ∀ i, FPTyped.active p s i → ∃ word zero : BitVec 32,
        s.mem p.weight (FPTyped.weightOffset p s i) = .mk .int word.toInt ∧
        s.mem p.zero (FPTyped.zeroOffset p s i) = .mk .int zero.toInt ∧
        final.mem p.output (FPTyped.outputOffset p s i) = .mk .fp16 (FPTyped.packedResult p s i word zero)
```

**Assumptions / layout contracts:**
- `hInputs : FPTyped.PackedInputs p s`
- `hw : ∀ i j, FPTyped.active p s i → FPTyped.active p s j →
      FPTyped.outputOffset p s i = FPTyped.outputOffset p s j → i = j`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
