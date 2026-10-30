# Spec sheet — `bench/tritonbench_g/rms_matmul_rbe/Correct.lean`

**Python source:** `bench/tritonbench_g/rms_matmul_rbe/rms_matmul_rbe.py`

## Public theorem: `half_formal_real_correct`

<details><summary>docstring</summary>

```
/-- Both complete non-FP8 source variants have actual trace safety, successful
finite-flat execution, the independent padded-input RMS-GEMM formula and the
full raw-cell frame. K need not be divisible by BK. Bounds include all unmasked
padding reads; the output layout need be injective only on active lanes.
The six input storage labels are a formal family: pinned native dot admission,
IEEE rounding, the FP8 value branch remain outside
this exact-real statement. -/
```
</details>

**Statement:**
```lean
specification half_formal_real_correct (input : TypedSource.Storage) (outputFP8 variant : Bool)
    (A : FlatAlloc) (s : BlockState) (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta EPS : ℝ)
    (hd : A.Disjoint) (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (hu : s.undef = fun _ _ => 0) (hcol : 0 < son) (hrow : N * son ≤ som)
    (hb : AccessSafety.Reads A.extent s X W RMS M N K BM BN BK sxb sxm sxk swk swn srms)
    (ho : ∀ i : TileIndex [BM, BN], active s M N BM BN i →
      outOffset s N BM BN sob som son i < A.extent OUT) :
    VeriTile.Spec.Real (Contracts.ExactIO
      (TypedSource.kernel input Bool.false outputFP8 variant X W RMS OUT M N K sxb sxm sxk swk swn srms
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
/-- Output storage can be selected independently of USE_FP8. Both fp16
and E5M2 output profiles are needed by the source QKV caller. -/
```
```lean
def kernel (input : Storage) (fp8 outputFP8 variant : Bool) : Source :=
  match input, fp8, outputFP8, variant with
  | .fp16, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float16, half, tl.float16, original]
  | .fp16, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float16, half, tl.float16, commuted]
  | .fp16, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float16, half, tl.float8e5, original]
  | .fp16, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float16, half, tl.float8e5, commuted]
  | .fp16, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float16, fp8, tl.float16, original]
  | .fp16, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float16, fp8, tl.float16, commuted]
  | .fp16, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float16, fp8, tl.float8e5, original]
  | .fp16, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float16, fp8, tl.float8e5, commuted]
  | .bf16, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.bfloat16, half, tl.float16, original]
  | .bf16, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.bfloat16, half, tl.float16, commuted]
  | .bf16, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.bfloat16, half, tl.float8e5, original]
  | .bf16, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.bfloat16, half, tl.float8e5, commuted]
  | .bf16, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.bfloat16, fp8, tl.float16, original]
  | .bf16, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.bfloat16, fp8, tl.float16, commuted]
  | .bf16, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.bfloat16, fp8, tl.float8e5, original]
  | .bf16, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.bfloat16, fp8, tl.float8e5, commuted]
  | .fp32, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float32, half, tl.float16, original]
  | .fp32, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float32, half, tl.float16, commuted]
  | .fp32, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float32, half, tl.float8e5, original]
  | .fp32, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float32, half, tl.float8e5, commuted]
  | .fp32, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float32, fp8, tl.float16, original]
  | .fp32, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float32, fp8, tl.float16, commuted]
  | .fp32, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float32, fp8, tl.float8e5, original]
  | .fp32, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float32, fp8, tl.float8e5, commuted]
  | .fp64, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float64, half, tl.float16, original]
  | .fp64, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float64, half, tl.float16, commuted]
  | .fp64, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float64, half, tl.float8e5, original]
  | .fp64, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float64, half, tl.float8e5, commuted]
  | .fp64, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float64, fp8, tl.float16, original]
  | .fp64, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float64, fp8, tl.float16, commuted]
  | .fp64, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float64, fp8, tl.float8e5, original]
  | .fp64, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float64, fp8, tl.float8e5, commuted]
  | .f8e4, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float8e4nv, half, tl.float16, original]
  | .f8e4, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float8e4nv, half, tl.float16, commuted]
  | .f8e4, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float8e4nv, half, tl.float8e5, original]
  | .f8e4, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float8e4nv, half, tl.float8e5, commuted]
  | .f8e4, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float8e4nv, fp8, tl.float16, original]
  | .f8e4, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float8e4nv, fp8, tl.float16, commuted]
  | .f8e4, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float8e4nv, fp8, tl.float8e5, original]
  | .f8e4, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float8e4nv, fp8, tl.float8e5, commuted]
  | .f8e5, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float8e5, half, tl.float16, original]
  | .f8e5, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float8e5, half, tl.float16, commuted]
  | .f8e5, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float8e5, half, tl.float8e5, original]
  | .f8e5, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float8e5, half, tl.float8e5, commuted]
  | .f8e5, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float8e5, fp8, tl.float16, original]
  | .f8e5, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float8e5, fp8, tl.float16, commuted]
  | .f8e5, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float8e5, fp8, tl.float8e5, original]
  | .f8e5, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float8e5, fp8, tl.float8e5, commuted]
```
</details>

## Public theorem: `qkv_half_formal_real_correct`

<details><summary>docstring</summary>

```
/-- Complete sequential QKV real IO for both variants and all formal
floating X storage labels. Each output uses its own initial weight matrix and
stride tuple; all unmasked padding reads require finite bounds. Native/IEEE
admission and the byte-value real theorem are separate obligations. -/
```
</details>

**Statement:**
```lean
specification qkv_half_formal_real_correct (input : TypedSource.Storage) (variant : Bool) (A : FlatAlloc)
    (X QW KW VW RMS Q KOut V : RegionName) (s : BlockState)
    (M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK : Nat)
    (start : Int) (theta EPS : ℝ)
    (hu : s.undef = fun _ _ => 0)
    (hiQ : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN sqb sqm sqn i = outOffset s N BM BN sqb sqm sqn j → i = j)
    (hiK : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN skb skm skn i = outOffset s N BM BN skb skm skn j → i = j)
    (hiV : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN svb svm svn i = outOffset s N BM BN svb svm svn j → i = j)
    (hXQ : X ≠ Q) (hRQ : RMS ≠ Q) (hKWQ : KW ≠ Q) (hVWQ : VW ≠ Q)
    (hXK : X ≠ KOut) (hRK : RMS ≠ KOut) (hVWK : VW ≠ KOut)
    (hQK : Q ≠ KOut) (hQV : Q ≠ V) (hKV : KOut ≠ V)
    (hd : A.Disjoint) (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (bq : AccessSafety.Reads A.extent s X QW RMS M N K BM BN BK sxb sxm sxk sqwk sqwn srms)
    (bk : AccessSafety.Reads A.extent s X KW RMS M N K BM BN BK sxb sxm sxk skwk skwn srms)
    (bv : AccessSafety.Reads A.extent s X VW RMS M N K BM BN BK sxb sxm sxk svwk svwn srms)
    (oq : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN sqb sqm sqn i < A.extent Q)
    (ok : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN skb skm skn i < A.extent KOut)
    (ov : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN svb svm svn i < A.extent V) :
    VeriTile.Spec.Real (Contracts.QKVExactIO
      (TypedSource.qkv input Bool.false variant X QW KW VW RMS Q KOut V M N K
        sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
        sqb sqm sqn skb skm skn svb svm svn BM BN BK start theta EPS)
      A s X QW KW VW RMS Q KOut V M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
        sqb sqm sqn skb skm skn svb svm svn BM BN BK EPS)
```

**Assumptions / layout contracts:**
- `hu : s.undef = fun _ _ => 0`
- `hiQ : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN sqb sqm sqn i = outOffset s N BM BN sqb sqm sqn j → i = j`
- `hiK : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN skb skm skn i = outOffset s N BM BN skb skm skn j → i = j`
- `hiV : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN svb svm svn i = outOffset s N BM BN svb svm svn j → i = j`
- `hXQ : X ≠ Q`
- `hRQ : RMS ≠ Q`
- `hKWQ : KW ≠ Q`
- `hVWQ : VW ≠ Q`
- `hXK : X ≠ KOut`
- `hRK : RMS ≠ KOut`
- `hVWK : VW ≠ KOut`
- `hQK : Q ≠ KOut`
- `hQV : Q ≠ V`
- `hKV : KOut ≠ V`
- `hd : A.Disjoint`
- `hc : ∀ r, r ∉ A.regions → A.extent r = 0`
- `oq : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN sqb sqm sqn i < A.extent Q`
- `ok : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN skb skm skn i < A.extent KOut`
- `ov : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN svb svm svn i < A.extent V`

**Closed-form spec defs (transitive):** `qkv`, `kernel`

<details><summary><code>qkv</code></summary>

```
/-- The exact three-call expansion of rms_matmul_rbe_qkv. All passes restart
from program ids and preserve their independent weight/output strides.
The source's rotary flags True, True, False are retained at the call sites;
the callee does not use them or apply a rotary epilogue. -/
```
```lean
def qkv (input : Storage) (fp8 variant : Bool)
    (X QW KW VW RMS Q KOut V : RegionName)
    (M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK : Nat)
    (start : Int) (theta EPS : ℝ) : ComputeKernel :=
  .mk [X, QW, KW, VW, RMS] [Q, KOut, V]
    ((kernel input fp8 Bool.false variant X QW RMS Q M N K
      sxb sxm sxk sqwk sqwn srms sqb sqm sqn BM BN BK start Bool.true theta EPS).surfaceBody ++
     (kernel input fp8 fp8 variant X KW RMS KOut M N K
      sxb sxm sxk skwk skwn srms skb skm skn BM BN BK start Bool.true theta EPS).surfaceBody ++
     (kernel input fp8 fp8 variant X VW RMS V M N K
      sxb sxm sxk svwk svwn srms svb svm svn BM BN BK start Bool.false theta EPS).surfaceBody)
```
</details>

<details><summary><code>kernel</code></summary>

```
/-- Output storage can be selected independently of USE_FP8. Both fp16
and E5M2 output profiles are needed by the source QKV caller. -/
```
```lean
def kernel (input : Storage) (fp8 outputFP8 variant : Bool) : Source :=
  match input, fp8, outputFP8, variant with
  | .fp16, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float16, half, tl.float16, original]
  | .fp16, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float16, half, tl.float16, commuted]
  | .fp16, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float16, half, tl.float8e5, original]
  | .fp16, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float16, half, tl.float8e5, commuted]
  | .fp16, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float16, fp8, tl.float16, original]
  | .fp16, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float16, fp8, tl.float16, commuted]
  | .fp16, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float16, fp8, tl.float8e5, original]
  | .fp16, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float16, fp8, tl.float8e5, commuted]
  | .bf16, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.bfloat16, half, tl.float16, original]
  | .bf16, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.bfloat16, half, tl.float16, commuted]
  | .bf16, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.bfloat16, half, tl.float8e5, original]
  | .bf16, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.bfloat16, half, tl.float8e5, commuted]
  | .bf16, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.bfloat16, fp8, tl.float16, original]
  | .bf16, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.bfloat16, fp8, tl.float16, commuted]
  | .bf16, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.bfloat16, fp8, tl.float8e5, original]
  | .bf16, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.bfloat16, fp8, tl.float8e5, commuted]
  | .fp32, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float32, half, tl.float16, original]
  | .fp32, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float32, half, tl.float16, commuted]
  | .fp32, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float32, half, tl.float8e5, original]
  | .fp32, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float32, half, tl.float8e5, commuted]
  | .fp32, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float32, fp8, tl.float16, original]
  | .fp32, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float32, fp8, tl.float16, commuted]
  | .fp32, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float32, fp8, tl.float8e5, original]
  | .fp32, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float32, fp8, tl.float8e5, commuted]
  | .fp64, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float64, half, tl.float16, original]
  | .fp64, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float64, half, tl.float16, commuted]
  | .fp64, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float64, half, tl.float8e5, original]
  | .fp64, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float64, half, tl.float8e5, commuted]
  | .fp64, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float64, fp8, tl.float16, original]
  | .fp64, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float64, fp8, tl.float16, commuted]
  | .fp64, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float64, fp8, tl.float8e5, original]
  | .fp64, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float64, fp8, tl.float8e5, commuted]
  | .f8e4, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float8e4nv, half, tl.float16, original]
  | .f8e4, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float8e4nv, half, tl.float16, commuted]
  | .f8e4, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float8e4nv, half, tl.float8e5, original]
  | .f8e4, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float8e4nv, half, tl.float8e5, commuted]
  | .f8e4, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float8e4nv, fp8, tl.float16, original]
  | .f8e4, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float8e4nv, fp8, tl.float16, commuted]
  | .f8e4, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float8e4nv, fp8, tl.float8e5, original]
  | .f8e4, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float8e4nv, fp8, tl.float8e5, commuted]
  | .f8e5, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float8e5, half, tl.float16, original]
  | .f8e5, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float8e5, half, tl.float16, commuted]
  | .f8e5, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float8e5, half, tl.float8e5, original]
  | .f8e5, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float8e5, half, tl.float8e5, commuted]
  | .f8e5, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float8e5, fp8, tl.float16, original]
  | .f8e5, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float8e5, fp8, tl.float16, commuted]
  | .f8e5, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float8e5, fp8, tl.float8e5, original]
  | .f8e5, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float8e5, fp8, tl.float8e5, commuted]
```
</details>

## Public theorem: `fp8_formal_real_correct`

<details><summary>docstring</summary>

```
/-- Both complete FP8-weight source variants have actual trace safety,
successful finite-flat execution and the independent padded-array RMS-GEMM
formula with a complete raw-cell frame. All weight premises demand actual
integer bytes with finite E5M2 decodes; the output is the real interpretation
of the selected fp16 or E5M2 carrier, not a packed-byte encoding theorem. The
six X labels are a formal family; native mixed-dot admission and physical IEEE
rounding remain outside this exact-real statement. -/
```
</details>

**Statement:**
```lean
specification fp8_formal_real_correct (input : TypedSource.Storage) (outputFP8 variant : Bool)
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
      (TypedSource.kernel input Bool.true outputFP8 variant X W RMS OUT M N K sxb sxm sxk swk swn srms
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
/-- Output storage can be selected independently of USE_FP8. Both fp16
and E5M2 output profiles are needed by the source QKV caller. -/
```
```lean
def kernel (input : Storage) (fp8 outputFP8 variant : Bool) : Source :=
  match input, fp8, outputFP8, variant with
  | .fp16, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float16, half, tl.float16, original]
  | .fp16, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float16, half, tl.float16, commuted]
  | .fp16, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float16, half, tl.float8e5, original]
  | .fp16, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float16, half, tl.float8e5, commuted]
  | .fp16, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float16, fp8, tl.float16, original]
  | .fp16, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float16, fp8, tl.float16, commuted]
  | .fp16, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float16, fp8, tl.float8e5, original]
  | .fp16, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float16, fp8, tl.float8e5, commuted]
  | .bf16, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.bfloat16, half, tl.float16, original]
  | .bf16, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.bfloat16, half, tl.float16, commuted]
  | .bf16, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.bfloat16, half, tl.float8e5, original]
  | .bf16, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.bfloat16, half, tl.float8e5, commuted]
  | .bf16, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.bfloat16, fp8, tl.float16, original]
  | .bf16, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.bfloat16, fp8, tl.float16, commuted]
  | .bf16, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.bfloat16, fp8, tl.float8e5, original]
  | .bf16, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.bfloat16, fp8, tl.float8e5, commuted]
  | .fp32, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float32, half, tl.float16, original]
  | .fp32, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float32, half, tl.float16, commuted]
  | .fp32, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float32, half, tl.float8e5, original]
  | .fp32, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float32, half, tl.float8e5, commuted]
  | .fp32, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float32, fp8, tl.float16, original]
  | .fp32, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float32, fp8, tl.float16, commuted]
  | .fp32, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float32, fp8, tl.float8e5, original]
  | .fp32, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float32, fp8, tl.float8e5, commuted]
  | .fp64, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float64, half, tl.float16, original]
  | .fp64, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float64, half, tl.float16, commuted]
  | .fp64, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float64, half, tl.float8e5, original]
  | .fp64, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float64, half, tl.float8e5, commuted]
  | .fp64, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float64, fp8, tl.float16, original]
  | .fp64, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float64, fp8, tl.float16, commuted]
  | .fp64, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float64, fp8, tl.float8e5, original]
  | .fp64, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float64, fp8, tl.float8e5, commuted]
  | .f8e4, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float8e4nv, half, tl.float16, original]
  | .f8e4, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float8e4nv, half, tl.float16, commuted]
  | .f8e4, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float8e4nv, half, tl.float8e5, original]
  | .f8e4, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float8e4nv, half, tl.float8e5, commuted]
  | .f8e4, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float8e4nv, fp8, tl.float16, original]
  | .f8e4, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float8e4nv, fp8, tl.float16, commuted]
  | .f8e4, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float8e4nv, fp8, tl.float8e5, original]
  | .f8e4, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float8e4nv, fp8, tl.float8e5, commuted]
  | .f8e5, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float8e5, half, tl.float16, original]
  | .f8e5, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float8e5, half, tl.float16, commuted]
  | .f8e5, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float8e5, half, tl.float8e5, original]
  | .f8e5, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float8e5, half, tl.float8e5, commuted]
  | .f8e5, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float8e5, fp8, tl.float16, original]
  | .f8e5, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float8e5, fp8, tl.float16, commuted]
  | .f8e5, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float8e5, fp8, tl.float8e5, original]
  | .f8e5, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float8e5, fp8, tl.float8e5, commuted]
```
</details>

## Public theorem: `qkv_fp8_formal_real_correct`

<details><summary>docstring</summary>

```
/-- Full finite-flat QKV byte IO from independent finite decoded arrays.
Q retains fp16 storage; K/V use E5M2. Every padded read and every unwritten
raw cell is covered. This theorem uses exact real arithmetic, not IEEE rounding. -/
```
</details>

**Statement:**
```lean
specification qkv_fp8_formal_real_correct (input : TypedSource.Storage) (variant : Bool) (A : FlatAlloc)
    (X QW KW VW RMS Q KOut V : RegionName) (s : BlockState)
    (M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK : Nat)
    (start : Int) (theta EPS : ℝ)
    (xs : Fin (cdiv K BK) → Fin BM → Fin BK → ℝ)
    (rs : Fin (cdiv K BK) → Fin BK → ByteExecution.Byte)
    (qws kws vws : Fin (cdiv K BK) → Fin BK → Fin BN → ByteExecution.Byte)
    (hu : s.undef = fun _ _ => 0)
    (hiQ : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN sqb sqm sqn i = outOffset s N BM BN sqb sqm sqn j → i = j)
    (hiK : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN skb skm skn i = outOffset s N BM BN skb skm skn j → i = j)
    (hiV : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN svb svm svn i = outOffset s N BM BN svb svm svn j → i = j)
    (hXQW : X ≠ QW) (hXKW : X ≠ KW) (hXVW : X ≠ VW) (hXR : X ≠ RMS)
    (inputQ : ByteExecution.Inputs s X QW RMS M N K BM BN BK sxb sxm sxk sqwk sqwn srms xs rs qws)
    (inputK : ByteExecution.Inputs s X KW RMS M N K BM BN BK sxb sxm sxk skwk skwn srms xs rs kws)
    (inputV : ByteExecution.Inputs s X VW RMS M N K BM BN BK sxb sxm sxk svwk svwn srms xs rs vws)
    (hXQ : X ≠ Q) (hRQ : RMS ≠ Q) (hKWQ : KW ≠ Q) (hVWQ : VW ≠ Q)
    (hXK : X ≠ KOut) (hRK : RMS ≠ KOut) (hVWK : VW ≠ KOut)
    (hQK : Q ≠ KOut) (hQV : Q ≠ V) (hKV : KOut ≠ V)
    (hd : A.Disjoint) (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (bq : AccessSafety.Reads A.extent s X QW RMS M N K BM BN BK sxb sxm sxk sqwk sqwn srms)
    (bk : AccessSafety.Reads A.extent s X KW RMS M N K BM BN BK sxb sxm sxk skwk skwn srms)
    (bv : AccessSafety.Reads A.extent s X VW RMS M N K BM BN BK sxb sxm sxk svwk svwn srms)
    (oq : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN sqb sqm sqn i < A.extent Q)
    (ok : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN skb skm skn i < A.extent KOut)
    (ov : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN svb svm svn i < A.extent V) :
    VeriTile.Spec.Real (Contracts.QKVByteIO
      (TypedSource.qkv input Bool.true variant X QW KW VW RMS Q KOut V M N K
        sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
        sqb sqm sqn skb skm skn svb svm svn BM BN BK start theta EPS)
      A s Q KOut V M N K sqb sqm sqn skb skm skn svb svm svn BM BN BK EPS xs rs qws kws vws)
```

**Assumptions / layout contracts:**
- `xs : Fin (cdiv K BK) → Fin BM → Fin BK → ℝ`
- `rs : Fin (cdiv K BK) → Fin BK → ByteExecution.Byte`
- `qws kws vws : Fin (cdiv K BK) → Fin BK → Fin BN → ByteExecution.Byte`
- `hu : s.undef = fun _ _ => 0`
- `hiQ : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN sqb sqm sqn i = outOffset s N BM BN sqb sqm sqn j → i = j`
- `hiK : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN skb skm skn i = outOffset s N BM BN skb skm skn j → i = j`
- `hiV : ∀ i j : TileIndex [BM, BN], active s M N BM BN i → active s M N BM BN j →
      outOffset s N BM BN svb svm svn i = outOffset s N BM BN svb svm svn j → i = j`
- `hXQW : X ≠ QW`
- `hXKW : X ≠ KW`
- `hXVW : X ≠ VW`
- `hXR : X ≠ RMS`
- `hXQ : X ≠ Q`
- `hRQ : RMS ≠ Q`
- `hKWQ : KW ≠ Q`
- `hVWQ : VW ≠ Q`
- `hXK : X ≠ KOut`
- `hRK : RMS ≠ KOut`
- `hVWK : VW ≠ KOut`
- `hQK : Q ≠ KOut`
- `hQV : Q ≠ V`
- `hKV : KOut ≠ V`
- `hd : A.Disjoint`
- `hc : ∀ r, r ∉ A.regions → A.extent r = 0`
- `oq : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN sqb sqm sqn i < A.extent Q`
- `ok : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN skb skm skn i < A.extent KOut`
- `ov : ∀ i : TileIndex [BM, BN], active s M N BM BN i → outOffset s N BM BN svb svm svn i < A.extent V`

**Closed-form spec defs (transitive):** `qkv`, `kernel`

<details><summary><code>qkv</code></summary>

```
/-- The exact three-call expansion of rms_matmul_rbe_qkv. All passes restart
from program ids and preserve their independent weight/output strides.
The source's rotary flags True, True, False are retained at the call sites;
the callee does not use them or apply a rotary epilogue. -/
```
```lean
def qkv (input : Storage) (fp8 variant : Bool)
    (X QW KW VW RMS Q KOut V : RegionName)
    (M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK : Nat)
    (start : Int) (theta EPS : ℝ) : ComputeKernel :=
  .mk [X, QW, KW, VW, RMS] [Q, KOut, V]
    ((kernel input fp8 Bool.false variant X QW RMS Q M N K
      sxb sxm sxk sqwk sqwn srms sqb sqm sqn BM BN BK start Bool.true theta EPS).surfaceBody ++
     (kernel input fp8 fp8 variant X KW RMS KOut M N K
      sxb sxm sxk skwk skwn srms skb skm skn BM BN BK start Bool.true theta EPS).surfaceBody ++
     (kernel input fp8 fp8 variant X VW RMS V M N K
      sxb sxm sxk svwk svwn srms svb svm svn BM BN BK start Bool.false theta EPS).surfaceBody)
```
</details>

<details><summary><code>kernel</code></summary>

```
/-- Output storage can be selected independently of USE_FP8. Both fp16
and E5M2 output profiles are needed by the source QKV caller. -/
```
```lean
def kernel (input : Storage) (fp8 outputFP8 variant : Bool) : Source :=
  match input, fp8, outputFP8, variant with
  | .fp16, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float16, half, tl.float16, original]
  | .fp16, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float16, half, tl.float16, commuted]
  | .fp16, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float16, half, tl.float8e5, original]
  | .fp16, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float16, half, tl.float8e5, commuted]
  | .fp16, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float16, fp8, tl.float16, original]
  | .fp16, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float16, fp8, tl.float16, commuted]
  | .fp16, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float16, fp8, tl.float8e5, original]
  | .fp16, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float16, fp8, tl.float8e5, commuted]
  | .bf16, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.bfloat16, half, tl.float16, original]
  | .bf16, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.bfloat16, half, tl.float16, commuted]
  | .bf16, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.bfloat16, half, tl.float8e5, original]
  | .bf16, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.bfloat16, half, tl.float8e5, commuted]
  | .bf16, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.bfloat16, fp8, tl.float16, original]
  | .bf16, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.bfloat16, fp8, tl.float16, commuted]
  | .bf16, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.bfloat16, fp8, tl.float8e5, original]
  | .bf16, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.bfloat16, fp8, tl.float8e5, commuted]
  | .fp32, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float32, half, tl.float16, original]
  | .fp32, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float32, half, tl.float16, commuted]
  | .fp32, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float32, half, tl.float8e5, original]
  | .fp32, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float32, half, tl.float8e5, commuted]
  | .fp32, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float32, fp8, tl.float16, original]
  | .fp32, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float32, fp8, tl.float16, commuted]
  | .fp32, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float32, fp8, tl.float8e5, original]
  | .fp32, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float32, fp8, tl.float8e5, commuted]
  | .fp64, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float64, half, tl.float16, original]
  | .fp64, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float64, half, tl.float16, commuted]
  | .fp64, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float64, half, tl.float8e5, original]
  | .fp64, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float64, half, tl.float8e5, commuted]
  | .fp64, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float64, fp8, tl.float16, original]
  | .fp64, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float64, fp8, tl.float16, commuted]
  | .fp64, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float64, fp8, tl.float8e5, original]
  | .fp64, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float64, fp8, tl.float8e5, commuted]
  | .f8e4, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float8e4nv, half, tl.float16, original]
  | .f8e4, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float8e4nv, half, tl.float16, commuted]
  | .f8e4, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float8e4nv, half, tl.float8e5, original]
  | .f8e4, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float8e4nv, half, tl.float8e5, commuted]
  | .f8e4, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float8e4nv, fp8, tl.float16, original]
  | .f8e4, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float8e4nv, fp8, tl.float16, commuted]
  | .f8e4, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float8e4nv, fp8, tl.float8e5, original]
  | .f8e4, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float8e4nv, fp8, tl.float8e5, commuted]
  | .f8e5, Bool.false, Bool.false, Bool.false => rmsQkvSource%[tl.float8e5, half, tl.float16, original]
  | .f8e5, Bool.false, Bool.false, Bool.true => rmsQkvSource%[tl.float8e5, half, tl.float16, commuted]
  | .f8e5, Bool.false, Bool.true, Bool.false => rmsQkvSource%[tl.float8e5, half, tl.float8e5, original]
  | .f8e5, Bool.false, Bool.true, Bool.true => rmsQkvSource%[tl.float8e5, half, tl.float8e5, commuted]
  | .f8e5, Bool.true, Bool.false, Bool.false => rmsQkvSource%[tl.float8e5, fp8, tl.float16, original]
  | .f8e5, Bool.true, Bool.false, Bool.true => rmsQkvSource%[tl.float8e5, fp8, tl.float16, commuted]
  | .f8e5, Bool.true, Bool.true, Bool.false => rmsQkvSource%[tl.float8e5, fp8, tl.float8e5, original]
  | .f8e5, Bool.true, Bool.true, Bool.true => rmsQkvSource%[tl.float8e5, fp8, tl.float8e5, commuted]
```
</details>

## Also present (pinned special-case summaries)
- `rms_matmul_rbe_closed_form_correct`
- `rms_matmul_rbe_qkv_closed_form_correct`
