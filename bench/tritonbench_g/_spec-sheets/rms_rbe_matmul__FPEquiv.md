# Spec sheet — `bench/tritonbench_g/rms_rbe_matmul/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/rms_rbe_matmul/rms_rbe_matmul.py`

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
specification source_equivalence (input : Storage) (fp8 : Bool) (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta eps : ℝ) :
    program input fp8 Bool.false X W RMS OUT M N K sxb sxm sxk swk swn srms
      sob som son BM BN BK start rbe theta eps ≡[assumptions]
    program input fp8 Bool.true X W RMS OUT M N K sxb sxm sxk swk swn srms
      sob som son BM BN BK start rbe theta eps
```

**Closed-form spec defs (transitive):** `program`, `assumptions`, `kernel`

<details><summary><code>program</code></summary>

```lean
def program (input : Storage) (fp8 variant : Bool) (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta eps : ℝ) : Program where
  kernel := kernel input fp8 variant X W RMS OUT M N K sxb sxm sxk swk swn srms
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
