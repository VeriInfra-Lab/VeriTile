# Spec sheet — `bench/tritonbench_g/matmul_dequantize_int4/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/matmul_dequantize_int4/matmul_dequantize_int4.py`

## Public theorem: `matmul_dequantize_int4_fp16_correctness`

<details><summary>docstring</summary>

```
/-! Both original NO_GROUPS branches and all K iterations. Binary16 products
and subtraction retain their separate source rounding points; D is an explicit
typed fp16×fp16→fp32 dot. Compiler/GPU dot and contraction binding is separate. -/
```
</details>

**Statement:**
```lean
specification matmul_dequantize_int4_fp16_correctness
    (rest : Algebra Word) (D : FPTyped.Dot Word) (p : FPTyped.Parameters) (s : State Word)
    (hInputs : FPTyped.PackedInputs p s)
    (hw : ∀ i j, FPTyped.active p s i → FPTyped.active p s j →
      FPTyped.outputOffset p s i = FPTyped.outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (FPTyped.backend (FPSoftware.scalar rest) D)
      (FPTyped.source p) s = some final ∧
      FPTyped.Post p s final (FPSoftware.machineOutput D p s) ∧
      ∀ step, step < FPTyped.steps p → ∀ i : FPTyped.BIndex p, ∃ word zero : BitVec 32,
        s.mem p.weight (FPTyped.bOffset p s step i) = .mk .int word.toInt ∧
        s.mem p.zero (FPTyped.zeroOffset p s step (i.2.1, PUnit.unit)) = .mk .int zero.toInt ∧
        FPTyped.weightNibble p s step i = ((word >>> (i.1.val % 8 * 4)) &&& (15 : BitVec 32)).toInt ∧
        FPTyped.zeroNibble p s step (i.2.1, PUnit.unit) =
          ((zero >>> (FPTyped.inputColumn p s i.2.1 % 8 * 4)) &&& (15 : BitVec 32)).toInt
```

**Assumptions / layout contracts:**
- `hInputs : FPTyped.PackedInputs p s`
- `hw : ∀ i j, FPTyped.active p s i → FPTyped.active p s j →
      FPTyped.outputOffset p s i = FPTyped.outputOffset p s j → i = j`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
