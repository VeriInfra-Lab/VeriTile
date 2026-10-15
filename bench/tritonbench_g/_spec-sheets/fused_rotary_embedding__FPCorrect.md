# Spec sheet — `bench/tritonbench_g/fused_rotary_embedding/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/fused_rotary_embedding/fused_rotary_embedding.py`

## Public theorem: `fused_rotary_q_fp32_correctness`

<details><summary>docstring</summary>

```
/-- Complete selected in-place Q pair, with concrete binary32 rounding at each
multiply and add/subtract, initial-state reads, successful execution and frame. -/
```
</details>

**Statement:**
```lean
specification fused_rotary_q_fp32_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (p : FPQ.Parameters)
    (s : State (FP.Value .fp32)) (hstride : 0 < p.dimensionStride) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg) (FPQ.source p) s = some final ∧
      FPQ.Post p s final (FPQ.roundedValue cfg p s)
```

**Assumptions / layout contracts:**
- `hstride : 0 < p.dimensionStride`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `fused_rotary_vcache_typed_correctness`

<details><summary>docstring</summary>

```
/-- Complete selected V chain: leader gate, context length, block table,
initial payload, readback and frame. No floating arithmetic law is required. -/
```
</details>

**Statement:**
```lean
specification fused_rotary_vcache_typed_correctness {α : Type} [Inhabited α]
    (rest : Algebra α) (p : FPV.Parameters) (s : State α)
    (hinj : FPV.active p s → Function.Injective (FPV.outputOffset p s)) :
    ∃ final, FP.Structural.exec (KcacheCopyTriton.TypedCopy.copyBackend rest)
      (FPV.source p) s = some final ∧ FPV.Post p s final
```

**Assumptions / layout contracts:**
- `hinj : FPV.active p s → Function.Injective (FPV.outputOffset p s)`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
