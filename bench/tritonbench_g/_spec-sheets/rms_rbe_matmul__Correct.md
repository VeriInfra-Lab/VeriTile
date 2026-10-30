# Spec sheet — `bench/tritonbench_g/rms_rbe_matmul/Correct.lean`

**Python source:** `bench/tritonbench_g/rms_rbe_matmul/rms_rbe_matmul.py`

## Public theorem: `half_formal_real_correct`

<details><summary>docstring</summary>

```
/-- Both complete non-FP8 source variants have actual trace safety, successful
finite-flat execution, the independent padded-input RMS-GEMM formula and the
full raw-cell frame. K need not be divisible by BK. Bounds include all unmasked
padding reads; the output layout need be injective only on active lanes.
The six input storage labels are a formal family: pinned native dot admission,
IEEE rounding, the FP8 value branch and the separate rotary JIT remain outside
this exact-real statement. -/
```
</details>

**Statement:**
```lean
specification half_formal_real_correct (input : TypedSource.Storage) (variant : Bool)
    (A : FlatAlloc) (s : BlockState) (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta EPS : ℝ)
    (hd : A.Disjoint) (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (hu : s.undef = fun _ _ => 0) (hcol : 0 < son) (hrow : N * son ≤ som)
    (hb : AccessSafety.Reads A.extent s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms)
    (ho : ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      outOffset s N BM BN sob som son i < A.extent OUT) :
    VeriTile.Spec.Real (Contracts.ExactIO
      (TypedSource.kernel input Bool.false variant X W RMS OUT M N K sxb sxm sxk swk swn srms
        sob som son BM BN BK start rbe theta EPS)
      A s X W RMS OUT M N K sxb sxm sxk swk swn srms sob som son BM BN BK EPS)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hc : ∀ r, r ∉ A.regions → A.extent r = 0`
- `hu : s.undef = fun _ _ => 0`
- `hcol : 0 < son`
- `hrow : N * son ≤ som`
- `hb : AccessSafety.Reads A.extent s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms`
- `ho : ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      outOffset s N BM BN sob som son i < A.extent OUT`

**Closed-form spec defs (transitive):** `kernel`

<details><summary><code>kernel</code></summary>

```
/-- The full floating-input family. The wrapper does not constrain X dtype;
frontend dot acceptance and native lowering must be checked separately. -/
```
```lean
def kernel (input : Storage) (fp8 variant : Bool) : Source :=
  match input, fp8, variant with
  | .fp16, Bool.false, Bool.false => rmsRbeSource%[tl.float16, half, original]
  | .fp16, Bool.false, Bool.true => rmsRbeSource%[tl.float16, half, commuted]
  | .fp16, Bool.true, Bool.false => rmsRbeSource%[tl.float16, fp8, original]
  | .fp16, Bool.true, Bool.true => rmsRbeSource%[tl.float16, fp8, commuted]
  | .bf16, Bool.false, Bool.false => rmsRbeSource%[tl.bfloat16, half, original]
  | .bf16, Bool.false, Bool.true => rmsRbeSource%[tl.bfloat16, half, commuted]
  | .bf16, Bool.true, Bool.false => rmsRbeSource%[tl.bfloat16, fp8, original]
  | .bf16, Bool.true, Bool.true => rmsRbeSource%[tl.bfloat16, fp8, commuted]
  | .fp32, Bool.false, Bool.false => rmsRbeSource%[tl.float32, half, original]
  | .fp32, Bool.false, Bool.true => rmsRbeSource%[tl.float32, half, commuted]
  | .fp32, Bool.true, Bool.false => rmsRbeSource%[tl.float32, fp8, original]
  | .fp32, Bool.true, Bool.true => rmsRbeSource%[tl.float32, fp8, commuted]
  | .fp64, Bool.false, Bool.false => rmsRbeSource%[tl.float64, half, original]
  | .fp64, Bool.false, Bool.true => rmsRbeSource%[tl.float64, half, commuted]
  | .fp64, Bool.true, Bool.false => rmsRbeSource%[tl.float64, fp8, original]
  | .fp64, Bool.true, Bool.true => rmsRbeSource%[tl.float64, fp8, commuted]
  | .f8e4, Bool.false, Bool.false => rmsRbeSource%[tl.float8e4nv, half, original]
  | .f8e4, Bool.false, Bool.true => rmsRbeSource%[tl.float8e4nv, half, commuted]
  | .f8e4, Bool.true, Bool.false => rmsRbeSource%[tl.float8e4nv, fp8, original]
  | .f8e4, Bool.true, Bool.true => rmsRbeSource%[tl.float8e4nv, fp8, commuted]
  | .f8e5, Bool.false, Bool.false => rmsRbeSource%[tl.float8e5, half, original]
  | .f8e5, Bool.false, Bool.true => rmsRbeSource%[tl.float8e5, half, commuted]
  | .f8e5, Bool.true, Bool.false => rmsRbeSource%[tl.float8e5, fp8, original]
  | .f8e5, Bool.true, Bool.true => rmsRbeSource%[tl.float8e5, fp8, commuted]
```
</details>

## Public theorem: `fp8_formal_real_correct`

<details><summary>docstring</summary>

```
/-- Both complete FP8-weight source variants have actual trace safety,
successful finite-flat execution and the independent padded-array RMS-GEMM
formula with a complete raw-cell frame. All weight premises demand actual
integer bytes with finite E5M2 decodes; the output is the real interpretation
of an E5M2 floating carrier, not a packed-byte encoding theorem. The six X
labels are a formal family; native mixed-dot admission, physical IEEE rounding
and the separate rotary JIT remain outside this exact-real statement. -/
```
</details>

**Statement:**
```lean
specification fp8_formal_real_correct (input : TypedSource.Storage) (variant : Bool)
    (A : FlatAlloc) (s : BlockState) (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta EPS : ℝ)
    (xs : Fin (cdiv K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (cdiv K BK) → Fin BK → ByteExecution.Byte)
    (ws : Fin (cdiv K BK) → Fin BK → Fin BN → ByteExecution.Byte)
    (hd : A.Disjoint) (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (hu : s.undef = fun _ _ => 0) (hcol : 0 < son) (hrow : N * son ≤ som)
    (hXW : X ≠ W) (hXR : X ≠ RMS)
    (hinput : ByteExecution.Inputs s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms xs rs ws)
    (hb : AccessSafety.Reads A.extent s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms)
    (ho : ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      outOffset s N BM BN sob som son i < A.extent OUT) :
    VeriTile.Spec.Real (Contracts.ByteIO
      (TypedSource.kernel input Bool.true variant X W RMS OUT M N K sxb sxm sxk swk swn srms
        sob som son BM BN BK start rbe theta EPS)
      A s OUT M N K sob som son BM BN BK EPS xs rs ws)
```

**Assumptions / layout contracts:**
- `xs : Fin (cdiv K BK) → Fin BM → Fin BK → ℝ`
- `rs : Fin (cdiv K BK) → Fin BK → ByteExecution.Byte`
- `ws : Fin (cdiv K BK) → Fin BK → Fin BN → ByteExecution.Byte`
- `hd : A.Disjoint`
- `hc : ∀ r, r ∉ A.regions → A.extent r = 0`
- `hu : s.undef = fun _ _ => 0`
- `hcol : 0 < son`
- `hrow : N * son ≤ som`
- `hXW : X ≠ W`
- `hXR : X ≠ RMS`
- `hinput : ByteExecution.Inputs s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms xs rs ws`
- `hb : AccessSafety.Reads A.extent s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms`
- `ho : ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      outOffset s N BM BN sob som son i < A.extent OUT`

**Closed-form spec defs (transitive):** `kernel`

<details><summary><code>kernel</code></summary>

```
/-- The full floating-input family. The wrapper does not constrain X dtype;
frontend dot acceptance and native lowering must be checked separately. -/
```
```lean
def kernel (input : Storage) (fp8 variant : Bool) : Source :=
  match input, fp8, variant with
  | .fp16, Bool.false, Bool.false => rmsRbeSource%[tl.float16, half, original]
  | .fp16, Bool.false, Bool.true => rmsRbeSource%[tl.float16, half, commuted]
  | .fp16, Bool.true, Bool.false => rmsRbeSource%[tl.float16, fp8, original]
  | .fp16, Bool.true, Bool.true => rmsRbeSource%[tl.float16, fp8, commuted]
  | .bf16, Bool.false, Bool.false => rmsRbeSource%[tl.bfloat16, half, original]
  | .bf16, Bool.false, Bool.true => rmsRbeSource%[tl.bfloat16, half, commuted]
  | .bf16, Bool.true, Bool.false => rmsRbeSource%[tl.bfloat16, fp8, original]
  | .bf16, Bool.true, Bool.true => rmsRbeSource%[tl.bfloat16, fp8, commuted]
  | .fp32, Bool.false, Bool.false => rmsRbeSource%[tl.float32, half, original]
  | .fp32, Bool.false, Bool.true => rmsRbeSource%[tl.float32, half, commuted]
  | .fp32, Bool.true, Bool.false => rmsRbeSource%[tl.float32, fp8, original]
  | .fp32, Bool.true, Bool.true => rmsRbeSource%[tl.float32, fp8, commuted]
  | .fp64, Bool.false, Bool.false => rmsRbeSource%[tl.float64, half, original]
  | .fp64, Bool.false, Bool.true => rmsRbeSource%[tl.float64, half, commuted]
  | .fp64, Bool.true, Bool.false => rmsRbeSource%[tl.float64, fp8, original]
  | .fp64, Bool.true, Bool.true => rmsRbeSource%[tl.float64, fp8, commuted]
  | .f8e4, Bool.false, Bool.false => rmsRbeSource%[tl.float8e4nv, half, original]
  | .f8e4, Bool.false, Bool.true => rmsRbeSource%[tl.float8e4nv, half, commuted]
  | .f8e4, Bool.true, Bool.false => rmsRbeSource%[tl.float8e4nv, fp8, original]
  | .f8e4, Bool.true, Bool.true => rmsRbeSource%[tl.float8e4nv, fp8, commuted]
  | .f8e5, Bool.false, Bool.false => rmsRbeSource%[tl.float8e5, half, original]
  | .f8e5, Bool.false, Bool.true => rmsRbeSource%[tl.float8e5, half, commuted]
  | .f8e5, Bool.true, Bool.false => rmsRbeSource%[tl.float8e5, fp8, original]
  | .f8e5, Bool.true, Bool.true => rmsRbeSource%[tl.float8e5, fp8, commuted]
```
</details>

## Also present (pinned special-case summaries)
- `rms_matmul_rbe_closed_form_correct`
