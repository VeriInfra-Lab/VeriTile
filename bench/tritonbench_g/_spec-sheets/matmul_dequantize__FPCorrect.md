# Spec sheet — `bench/tritonbench_g/matmul_dequantize/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/matmul_dequantize/matmul_dequantize.py`

## Public theorem: `matmul_dequantize_standalone_fp16_correctness`

<details><summary>docstring</summary>

```
/-! All three original accepted stages have independent initial-state contracts.
Both matmul stages cover their full K loops; the first covers both NO_GROUPS
branches. Native dot and scalar contraction binding remains explicit. -/
```
</details>

**Statement:**
```lean
specification matmul_dequantize_standalone_fp16_correctness [Inhabited MatmulDequantInt4.Binary16.Value] (rest : Algebra MatmulDequantInt4.Binary16.Value) (p : FPTyped.Standalone.Parameters) (s : State MatmulDequantInt4.Binary16.Value)
    (hInputs : MatmulDequantInt4.FPTyped.PackedInputs p s)
    (hw : ∀ i j, MatmulDequantInt4.FPTyped.active p s i → MatmulDequantInt4.FPTyped.active p s j →
      MatmulDequantInt4.FPTyped.outputOffset p s i = MatmulDequantInt4.FPTyped.outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (MatmulDequantInt4.FPTyped.software rest) (FPTyped.Standalone.source p) s = some final ∧
      MatmulDequantInt4.FPTyped.Post p s final (MatmulDequantInt4.FPTyped.roundedValue p s) ∧
      ∀ i, MatmulDequantInt4.FPTyped.active p s i → ∃ word zero : BitVec 32,
        s.mem p.weight (MatmulDequantInt4.FPTyped.weightOffset p s i) = .mk .int word.toInt ∧
        s.mem p.zero (MatmulDequantInt4.FPTyped.zeroOffset p s i) = .mk .int zero.toInt ∧
        final.mem p.output (MatmulDequantInt4.FPTyped.outputOffset p s i) =
          .mk .fp16 (MatmulDequantInt4.FPTyped.packedResult p s i word zero)
```

**Assumptions / layout contracts:**
- `hInputs : MatmulDequantInt4.FPTyped.PackedInputs p s`
- `hw : ∀ i j, MatmulDequantInt4.FPTyped.active p s i → MatmulDequantInt4.FPTyped.active p s j →
      MatmulDequantInt4.FPTyped.outputOffset p s i = MatmulDequantInt4.FPTyped.outputOffset p s j → i = j`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `matmul_dequantize_second_fp16_correctness`

**Statement:**
```lean
specification matmul_dequantize_second_fp16_correctness
    (rest : Algebra Int4Matmul.FPSoftware.Word) (D : FPTypedSecond.Dot Int4Matmul.FPSoftware.Word)
    (p : FPTypedSecond.Parameters) (s : State Int4Matmul.FPSoftware.Word)
    (hTrip : p.inner = p.blockInner * p.steps) (hSplit : s.pids 1 = 0)
    (hInputs : FPTypedSecond.PackedInputs p s)
    (hw : ∀ i j, FPTypedSecond.active p s i → FPTypedSecond.active p s j →
      FPTypedSecond.outputOffset p s i = FPTypedSecond.outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (FPTypedSecond.backend (Int4Matmul.FPSoftware.scalar rest) D)
      (FPTypedSecond.source p) s = some final ∧
      FPTypedSecond.Post p s final (FPSecondSoftware.machineOutput D p s) ∧
      ∀ step, step < p.steps → ∀ i : FPTypedSecond.BIndex p, ∃ word zero : BitVec 32,
        s.mem p.weight (FPTypedSecond.bOffset p s step i) = .mk .int word.toInt ∧
        s.mem p.zero (FPTypedSecond.zeroOffset p s step i) = .mk .int zero.toInt ∧
        FPTypedSecond.difference p s step i = FPTypedSecond.packedDifference p s i word zero
```

**Assumptions / layout contracts:**
- `hTrip : p.inner = p.blockInner * p.steps`
- `hSplit : s.pids 1 = 0`
- `hInputs : FPTypedSecond.PackedInputs p s`
- `hw : ∀ i j, FPTypedSecond.active p s i → FPTypedSecond.active p s j →
      FPTypedSecond.outputOffset p s i = FPTypedSecond.outputOffset p s j → i = j`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `matmul_dequantize_first_fp16_correctness`

**Statement:**
```lean
specification matmul_dequantize_first_fp16_correctness
    (rest : Algebra Word) (D : FPTypedFirst.Dot Word) (p : FPTypedFirst.Parameters) (s : State Word)
    (hInputs : FPTypedFirst.PackedInputs p s)
    (hw : ∀ i j, FPTypedFirst.active p s i → FPTypedFirst.active p s j →
      FPTypedFirst.outputOffset p s i = FPTypedFirst.outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (FPTypedFirst.backend (FPFirstSoftware.scalar rest) D)
      (FPTypedFirst.source p) s = some final ∧
      FPTypedFirst.Post p s final (FPFirstSoftware.machineOutput D p s) ∧
      ∀ step, step < FPTypedFirst.steps p → ∀ i : FPTypedFirst.BIndex p, ∃ word zero : BitVec 32,
        s.mem p.weight (FPTypedFirst.bOffset p s step i) = .mk .int word.toInt ∧
        s.mem p.zero (FPTypedFirst.zeroOffset p s step (i.2.1, PUnit.unit)) = .mk .int zero.toInt ∧
        FPTypedFirst.weightNibble p s step i = ((word >>> (i.1.val % 8 * 4)) &&& (15 : BitVec 32)).toInt ∧
        FPTypedFirst.zeroNibble p s step (i.2.1, PUnit.unit) =
          ((zero >>> (FPTypedFirst.inputColumn p s i.2.1 % 8 * 4)) &&& (15 : BitVec 32)).toInt
```

**Assumptions / layout contracts:**
- `hInputs : FPTypedFirst.PackedInputs p s`
- `hw : ∀ i j, FPTypedFirst.active p s i → FPTypedFirst.active p s j →
      FPTypedFirst.outputOffset p s i = FPTypedFirst.outputOffset p s j → i = j`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
