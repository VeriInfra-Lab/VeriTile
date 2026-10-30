# Spec sheet — `bench/tritonbench_g/rms_matmul_rbe/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/rms_matmul_rbe/rms_matmul_rbe.py`

## Public theorem: `source_equivalence`

<details><summary>docstring</summary>

```
/-- Full typed state and failure equivalence for the exact output-index
commutation, for arbitrary arithmetic, integer-power, dot and decode backends.
All E5M2 payloads remain in scope; no success or finiteness premise is assumed.
This formal floating-input family is not a native frontend admission result. -/
```
</details>

**Statement:**
```lean
specification source_equivalence (input : Storage) (fp8 outputFP8 : Bool) (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta eps : ℝ) :
    program input fp8 outputFP8 Bool.false X W RMS OUT M N K sxb sxm sxk swk swn srms
      sob som son BM BN BK start rbe theta eps ≡[assumptions]
    program input fp8 outputFP8 Bool.true X W RMS OUT M N K sxb sxm sxk swk swn srms
      sob som son BM BN BK start rbe theta eps
```

**Closed-form spec defs (transitive):** `program`, `assumptions`, `kernel`

<details><summary><code>program</code></summary>

```lean
def program (input : Storage) (fp8 outputFP8 variant : Bool) (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta eps : ℝ) : Program where
  kernel := kernel input fp8 outputFP8 variant X W RMS OUT M N K sxb sxm sxk swk swn srms
    sob som son BM BN BK start rbe theta eps
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
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

## Public theorem: `qkv_equivalence`

<details><summary>docstring</summary>

```
/-- Full sequential QKV state and failure equivalence. Q retains fp16 storage
in the byte-weight branch; K and V retain E5M2. No whole-program equivalence,
execution-success or finite-payload premise is supplied. -/
```
</details>

**Statement:**
```lean
specification qkv_equivalence (input : Storage) (fp8 : Bool)
    (X QW KW VW RMS Q KOut V : RegionName)
    (M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK : Nat)
    (start : Int) (theta EPS : ℝ) :
    qkvProgram input fp8 Bool.false X QW KW VW RMS Q KOut V M N K
      sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK start theta EPS ≡[assumptions]
    qkvProgram input fp8 Bool.true X QW KW VW RMS Q KOut V M N K
      sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK start theta EPS
```

**Closed-form spec defs (transitive):** `qkvProgram`, `assumptions`, `kernel`, `qkv`

<details><summary><code>qkvProgram</code></summary>

```lean
def qkvProgram (input : Storage) (fp8 variant : Bool)
    (X QW KW VW RMS Q KOut V : RegionName)
    (M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK : Nat)
    (start : Int) (theta EPS : ℝ) : Program where
  kernel := qkv input fp8 variant X QW KW VW RMS Q KOut V M N K
    sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
    sqb sqm sqn skb skm skn svb svm svn BM BN BK start theta EPS
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
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
