# Spec sheet — `bench/tritonbench_g/int8_dequant_matmul/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/int8_dequant_matmul/int8_dequant_matmul.py`

## Public theorem: `int8_dequant_matmul_fp_correctness`

<details><summary>docstring</summary>

```
/-! The original SPLIT_K=1, EVEN_K=true scope, every K iteration, both bias
branches. Inputs use signed byte words; accumulation wraps to int32 after each
block. All floating scalar operations use concrete software fp32/fp16 bits.
GPU/compiler binding of this interpreter remains a separate native obligation. -/
```
</details>

**Statement:**
```lean
specification int8_dequant_matmul_fp_correctness
    (rest : Algebra Word) (p : FPTyped.Parameters) (s : State Word) (divisor : Word)
    (hK : p.inner = p.blockInner * p.steps) (hpid1 : s.pids 1 = 0)
    (hInputs : FPTyped.ByteInputs p s)
    (hdiv : s.regs .fp32 [] "divfactor" = some (fun _ => divisor))
    (hw : ∀ i j, FPTyped.active p s i → FPTyped.active p s j →
      FPTyped.outputOffset p s i = FPTyped.outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (FPTyped.backend (FPSoftware.scalar rest))
      (FPTyped.source p) s = some final ∧
      FPTyped.Post p s final (FPSoftware.machineOutput p s divisor) ∧
      (∀ k, k < p.steps → ∀ i : FPTyped.AIndex p, ∃ w : BitVec 8,
        s.mem p.input (FPTyped.aOffset p s k i) = .mk .int w.toInt ∧
        FPTyped.aValue p s k i = w.toInt) ∧
      (∀ k, k < p.steps → ∀ i : FPTyped.BIndex p, ∃ w : BitVec 8,
        s.mem p.weight (FPTyped.bOffset p s k i) = .mk .int w.toInt ∧
        FPTyped.bValue p s k i = w.toInt)
```

**Assumptions / layout contracts:**
- `hK : p.inner = p.blockInner * p.steps`
- `hpid1 : s.pids 1 = 0`
- `hInputs : FPTyped.ByteInputs p s`
- `hdiv : s.regs .fp32 [] "divfactor" = some (fun _ => divisor)`
- `hw : ∀ i j, FPTyped.active p s i → FPTyped.active p s j →
      FPTyped.outputOffset p s i = FPTyped.outputOffset p s j → i = j`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
