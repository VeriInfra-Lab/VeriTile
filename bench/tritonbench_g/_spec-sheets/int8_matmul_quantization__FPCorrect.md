# Spec sheet — `bench/tritonbench_g/int8_matmul_quantization/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/int8_matmul_quantization/int8_matmul_quantization.py`

## Public theorem: `int8_matmul_quantization_producer_machine_correctness`

<details><summary>docstring</summary>

```
/-- Original two-pass quantization producer, all covered K columns, symbolic
block size and trip count, actual fp32 arithmetic and checked int8 conversion,
true fp16 scale encoding, and frame. The GEMM consumer is specified separately below. -/
```
</details>

**Statement:**
```lean
specification int8_matmul_quantization_producer_machine_correctness
    (rest : Algebra FPQuantize.Word) (cfg : FP.Config) (u : Nat → FPQuantize.Word → Int)
    (p : FPQuantize.Parameters) (initial : State FPQuantize.Word)
    (hw : 0<p.blockColumns) (hcover : p.columns ≤ p.steps*p.blockColumns)
    (hsep : p.input ≠ p.output) (hregions : p.output ≠ p.scales)
    (hd : FPQuantize.InputDefined cfg p initial)
    (hinj : ∀ a b : FPQuantize.WriteIndex p, a.1 < p.steps → b.1 < p.steps →
      FPQuantize.active p a.1 a.2 → FPQuantize.active p b.1 b.2 →
      FPQuantize.outputOffset p initial a.1 a.2 = FPQuantize.outputOffset p initial b.1 b.2 → a=b) :
    ∃ final, FP.Structural.exec (FPQuantize.backend rest cfg u) (FPQuantize.source p) initial=some final ∧
      FPQuantize.Post cfg p initial final
```

**Assumptions / layout contracts:**
- `u : Nat → FPQuantize.Word → Int`
- `hw : 0<p.blockColumns`
- `hcover : p.columns ≤ p.steps*p.blockColumns`
- `hsep : p.input ≠ p.output`
- `hregions : p.output ≠ p.scales`
- `hd : FPQuantize.InputDefined cfg p initial`
- `hinj : ∀ a b : FPQuantize.WriteIndex p, a.1 < p.steps → b.1 < p.steps →
      FPQuantize.active p a.1 a.2 → FPQuantize.active p b.1 b.2 →
      FPQuantize.outputOffset p initial a.1 a.2 = FPQuantize.outputOffset p initial b.1 b.2 → a=b`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `int8_matmul_quantization_consumer_machine_correctness`

<details><summary>docstring</summary>

```
/-- Original grouped SPLIT_K=1 GEMM consumer with all K positions covered,
including partial and extra masked blocks. The independent integer product
sum is projected to signed32 before actual fp32 rounding, separate fp32
scale products, and binary16 store. Overflow is retained, not excluded. -/
```
</details>

**Statement:**
```lean
specification int8_matmul_quantization_consumer_machine_correctness
    (rest : Algebra FPMatmul.Word) (cfg : FP.Config)
    (p : FPMatmul.Parameters) (initial : State FPMatmul.Word)
    (hk : 0<p.blockInner) (hcover : p.inner ≤ p.steps*p.blockInner)
    (hpid : initial.pids 1=0)
    (hinj : ∀ i j, FPMatmul.active p initial i → FPMatmul.active p initial j →
      FPMatmul.outputOffset p initial i=FPMatmul.outputOffset p initial j → i=j) :
    ∃ final, FP.Structural.exec (FPMatmul.backend rest cfg) (FPMatmul.source p) initial=some final ∧
      FPMatmul.Post p initial final (FPMatmul.exactOutput cfg p initial) ∧
      (∀ j : Fin p.inner, ∃ k, ∃ lane : Fin p.blockInner, k<p.steps ∧
        FPMatmul.inputInner p initial lane < p.inner-k*p.blockInner ∧
        k*p.blockInner+FPMatmul.inputInner p initial lane=j.val)
```

**Assumptions / layout contracts:**
- `hk : 0<p.blockInner`
- `hcover : p.inner ≤ p.steps*p.blockInner`
- `hpid : initial.pids 1=0`
- `hinj : ∀ i j, FPMatmul.active p initial i → FPMatmul.active p initial j →
      FPMatmul.outputOffset p initial i=FPMatmul.outputOffset p initial j → i=j`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
