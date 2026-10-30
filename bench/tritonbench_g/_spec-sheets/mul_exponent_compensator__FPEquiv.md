# Spec sheet — `bench/tritonbench_g/mul_exponent_compensator/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/mul_exponent_compensator/mul_exponent_compensator.py`

## Public theorem: `mul_compensator_commuted_equiv`

**Statement:**
```lean
specification mul_compensator_commuted_equiv (src dst : RegionName) (B : Nat)
    (_hB : 0 < B) (R : FP.ScalarArithmetic.Rules) :
    original src dst B ≡[R] commuted src dst B
```

**Assumptions / layout contracts:**
- `_hB : 0 < B`

**Closed-form spec defs (transitive):** `original`, `commuted`, `mul_kernel`, `multiplicationSite`, `mul_commuted_kernel`, `exponentCompensator`

<details><summary><code>original</code></summary>

```lean
noncomputable def original (src dst : RegionName) (B : Nat) : Program where
  kernel := mul_kernel src dst B
  domain := [multiplicationSite src dst B]
```
</details>

<details><summary><code>commuted</code></summary>

```lean
noncomputable def commuted (src dst : RegionName) (B : Nat) : Program :=
  { original src dst B with kernel := mul_commuted_kernel src dst B }
```
</details>

<details><summary><code>mul_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `mul_exponent_compensator.py`'s
`mul_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python local constexpr literal `2.0 ** (127 - 15)` is represented by the
  Lean constant `exponentCompensator` and injected as a real scalar antiquote.
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
noncomputable def mul_kernel
    (src dst : RegionName)
    (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  exponent_compensator = $((exponentCompensator : ℝ))
  idxs = tl.program_id(0) * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(src + idxs)
  y = x * exponent_compensator
  tl.store(dst + idxs, y)
}
```
</details>

<details><summary><code>multiplicationSite</code></summary>

```lean
noncomputable def multiplicationSite (src dst : RegionName) (B : Nat) : Site :=
  .mulScalar (Source.prefixBefore (mul_kernel src dst B).surfaceBody "y") B []
    (.ref .real [B] "x") (.ref .real [] "exponent_compensator")
```
</details>

<details><summary><code>mul_commuted_kernel</code></summary>

```
/-- Scalar-left scaling with the same load, constant and store. -/
```
```lean
noncomputable def mul_commuted_kernel (src dst : RegionName) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  exponent_compensator = $((exponentCompensator : ℝ))
  idxs = tl.program_id(0) * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(src + idxs)
  y = exponent_compensator * x
  tl.store(dst + idxs, y)
}
```
</details>

<details><summary><code>exponentCompensator</code></summary>

```
/-- The constexpr multiplier from `mul_exponent_compensator.py`. -/
```
```lean
noncomputable def exponentCompensator : ℝ :=
  (2 : ℝ) ^ (127 - 15)
```
</details>
