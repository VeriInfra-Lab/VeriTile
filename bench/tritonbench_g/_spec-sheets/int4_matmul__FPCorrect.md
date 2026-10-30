# Spec sheet — `bench/tritonbench_g/int4_matmul/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/int4_matmul/int4_matmul.py`

## Public theorem: `int4_matmul_fp16_correctness`

<details><summary>docstring</summary>

```
/-! Full accepted SPLIT_K=1 K-loop, with actual int32 packed storage, direct
binary16 dequantization and binary32 accumulation. D is the explicit typed
half×half→float dot primitive; its GPU implementation binding is separate.
The independent softwareDot instance has a fixed fp32 reduction schedule. -/
```
</details>

**Statement:**
```lean
specification int4_matmul_fp16_correctness
    (rest : Algebra FPSoftware.Word) (D : FPTyped.Dot FPSoftware.Word)
    (p : FPTyped.Parameters) (s : State FPSoftware.Word)
    (hTrip : p.inner = p.blockInner * p.steps) (hSplit : s.pids 1 = 0)
    (hInputs : FPTyped.PackedInputs p s)
    (hw : ∀ i j, FPTyped.active p s i → FPTyped.active p s j →
      FPTyped.outputOffset p s i = FPTyped.outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (FPTyped.backend (FPSoftware.scalar rest) D)
      (FPTyped.source p) s = some final ∧
      FPTyped.Post p s final (FPSoftware.machineOutput D p s) ∧
      ∀ step, step < p.steps → ∀ i : FPTyped.BIndex p, ∃ word zero : BitVec 32,
        s.mem p.weight (FPTyped.bOffset p s step i) = .mk .int word.toInt ∧
        s.mem p.zero (FPTyped.zeroOffset p s step i) = .mk .int zero.toInt ∧
        FPTyped.difference p s step i = FPTyped.packedDifference p s i word zero
```

**Assumptions / layout contracts:**
- `hTrip : p.inner = p.blockInner * p.steps`
- `hSplit : s.pids 1 = 0`
- `hInputs : FPTyped.PackedInputs p s`
- `hw : ∀ i j, FPTyped.active p s i → FPTyped.active p s j →
      FPTyped.outputOffset p s i = FPTyped.outputOffset p s j → i = j`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
