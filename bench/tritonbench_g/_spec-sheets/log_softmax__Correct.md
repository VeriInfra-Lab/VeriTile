# Spec sheet — `bench/tritonbench_g/log_softmax/Correct.lean`

**Python source:** `bench/tritonbench_g/log_softmax/log_softmax.py`

## Public theorem: `forward_typed_correct`

<details><summary>docstring</summary>

```
/-- Independent full-row log-softmax for both complete typed programs, with successful
flat execution, active address bounds and the complete unwritten-memory frame. -/
```
</details>

**Statement:**
```lean
specification forward_typed_correct (input output : Storage) (v : Bool)
    (OUT X : RegionName) (M N K R C : Nat)
    (hC : 0 < C) (hK : 0 < K) (hNC : N ≤ C) :
    Masked2DKernelIO₁.Implements (typedForwardIO input output v OUT X M N K R C)
      (fun _ _ xs j ↦ forwardExpected N R C hNC xs j)
```

**Assumptions / layout contracts:**
- `hC : 0 < C`
- `hK : 0 < K`
- `hNC : N ≤ C`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `backward_typed_correct`

<details><summary>docstring</summary>

```
/-- Independent full-row VJP for both complete typed programs under the flat
interface's explicit zero-undefined launch invariant. -/
```
</details>

**Statement:**
```lean
specification backward_typed_correct (input upstream output : Storage) (v : Bool)
    (Y G DX : RegionName) (M N K R C : Nat) (hK : 0 < K) (hNC : N ≤ C) :
    Masked2DKernelIO₂.Implements (typedBackwardIO input upstream output v Y G DX M N K R C)
      (fun _ _ xs ys j ↦ backwardExpected N R C hNC xs ys j)
```

**Assumptions / layout contracts:**
- `hK : 0 < K`
- `hNC : N ≤ C`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `forward_region_correct`

<details><summary>docstring</summary>

```
/-- Complete source execution with independent full-row log-softmax and raw memory
frame, allowing arbitrary undefined values and input/output aliases. -/
```
</details>

**Statement:**
```lean
specification forward_region_correct (input output : Storage) (v : Bool)
    (OUT X : RegionName) (M N K R C : Nat)
    (hC : 0 < C) (hK : 0 < K) (hNC : N ≤ C) (s : BlockState) :
    ∃ t, exec (typedForward input output v OUT X M N K R C).eraseDType.toAlgKernel s = some t ∧
      (∀ (r : Fin R) (i : Fin C) (hr : s.pids 0*R+r.val<M) (hi : i.val<N),
        t.readMem OUT (outOffset (s.pids 0) (s.pids 1) N K R (r,i,PUnit.unit)) =
          Math.forward (fun j : Fin N ↦
            s.readMem X ((s.pids 0*R+r.val)*N*K+j.val*K+s.pids 1)) ⟨i.val,hi⟩) ∧
      (∀ r o, (∀ i : TileIndex [R,C], active (s.pids 0) M N R i →
        ¬(OUT = r ∧ outOffset (s.pids 0) (s.pids 1) N K R i = o)) →
        t.mem r o = s.mem r o)
```

**Assumptions / layout contracts:**
- `hC : 0 < C`
- `hK : 0 < K`
- `hNC : N ≤ C`
- `hr : s.pids 0*R+r.val<M`
- `hi : i.val<N`

**Closed-form spec defs (transitive):** `typedForward`

<details><summary><code>typedForward</code></summary>

```lean
def typedForward (input output : Storage) (v : Bool) : ForwardSource :=
  match input, output, v with
  | .fp32, .fp32, Bool.false => logSoftmaxForward%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => logSoftmaxForward%[tl.float32, tl.float32, commuted]
  | .fp32, .fp16, Bool.false => logSoftmaxForward%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => logSoftmaxForward%[tl.float32, tl.float16, commuted]
  | .fp32, .bf16, Bool.false => logSoftmaxForward%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => logSoftmaxForward%[tl.float32, tl.bfloat16, commuted]
  | .fp32, .fp64, Bool.false => logSoftmaxForward%[tl.float32, tl.float64, original]
  | .fp32, .fp64, Bool.true => logSoftmaxForward%[tl.float32, tl.float64, commuted]
  | .fp32, .f8e4, Bool.false => logSoftmaxForward%[tl.float32, tl.float8e4nv, original]
  | .fp32, .f8e4, Bool.true => logSoftmaxForward%[tl.float32, tl.float8e4nv, commuted]
  | .fp32, .f8e5, Bool.false => logSoftmaxForward%[tl.float32, tl.float8e5, original]
  | .fp32, .f8e5, Bool.true => logSoftmaxForward%[tl.float32, tl.float8e5, commuted]
  | .fp16, .fp32, Bool.false => logSoftmaxForward%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => logSoftmaxForward%[tl.float16, tl.float32, commuted]
  | .fp16, .fp16, Bool.false => logSoftmaxForward%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => logSoftmaxForward%[tl.float16, tl.float16, commuted]
  | .fp16, .bf16, Bool.false => logSoftmaxForward%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => logSoftmaxForward%[tl.float16, tl.bfloat16, commuted]
  | .fp16, .fp64, Bool.false => logSoftmaxForward%[tl.float16, tl.float64, original]
  | .fp16, .fp64, Bool.true => logSoftmaxForward%[tl.float16, tl.float64, commuted]
  | .fp16, .f8e4, Bool.false => logSoftmaxForward%[tl.float16, tl.float8e4nv, original]
  | .fp16, .f8e4, Bool.true => logSoftmaxForward%[tl.float16, tl.float8e4nv, commuted]
  | .fp16, .f8e5, Bool.false => logSoftmaxForward%[tl.float16, tl.float8e5, original]
  | .fp16, .f8e5, Bool.true => logSoftmaxForward%[tl.float16, tl.float8e5, commuted]
  | .bf16, .fp32, Bool.false => logSoftmaxForward%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => logSoftmaxForward%[tl.bfloat16, tl.float32, commuted]
  | .bf16, .fp16, Bool.false => logSoftmaxForward%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => logSoftmaxForward%[tl.bfloat16, tl.float16, commuted]
  | .bf16, .bf16, Bool.false => logSoftmaxForward%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => logSoftmaxForward%[tl.bfloat16, tl.bfloat16, commuted]
  | .bf16, .fp64, Bool.false => logSoftmaxForward%[tl.bfloat16, tl.float64, original]
  | .bf16, .fp64, Bool.true => logSoftmaxForward%[tl.bfloat16, tl.float64, commuted]
  | .bf16, .f8e4, Bool.false => logSoftmaxForward%[tl.bfloat16, tl.float8e4nv, original]
  | .bf16, .f8e4, Bool.true => logSoftmaxForward%[tl.bfloat16, tl.float8e4nv, commuted]
  | .bf16, .f8e5, Bool.false => logSoftmaxForward%[tl.bfloat16, tl.float8e5, original]
  | .bf16, .f8e5, Bool.true => logSoftmaxForward%[tl.bfloat16, tl.float8e5, commuted]
  | .fp64, .fp32, Bool.false => logSoftmaxForward%[tl.float64, tl.float32, original]
  | .fp64, .fp32, Bool.true => logSoftmaxForward%[tl.float64, tl.float32, commuted]
  | .fp64, .fp16, Bool.false => logSoftmaxForward%[tl.float64, tl.float16, original]
  | .fp64, .fp16, Bool.true => logSoftmaxForward%[tl.float64, tl.float16, commuted]
  | .fp64, .bf16, Bool.false => logSoftmaxForward%[tl.float64, tl.bfloat16, original]
  | .fp64, .bf16, Bool.true => logSoftmaxForward%[tl.float64, tl.bfloat16, commuted]
  | .fp64, .fp64, Bool.false => logSoftmaxForward%[tl.float64, tl.float64, original]
  | .fp64, .fp64, Bool.true => logSoftmaxForward%[tl.float64, tl.float64, commuted]
  | .fp64, .f8e4, Bool.false => logSoftmaxForward%[tl.float64, tl.float8e4nv, original]
  | .fp64, .f8e4, Bool.true => logSoftmaxForward%[tl.float64, tl.float8e4nv, commuted]
  | .fp64, .f8e5, Bool.false => logSoftmaxForward%[tl.float64, tl.float8e5, original]
  | .fp64, .f8e5, Bool.true => logSoftmaxForward%[tl.float64, tl.float8e5, commuted]
  | .f8e4, .fp32, Bool.false => logSoftmaxForward%[tl.float8e4nv, tl.float32, original]
  | .f8e4, .fp32, Bool.true => logSoftmaxForward%[tl.float8e4nv, tl.float32, commuted]
  | .f8e4, .fp16, Bool.false => logSoftmaxForward%[tl.float8e4nv, tl.float16, original]
  | .f8e4, .fp16, Bool.true => logSoftmaxForward%[tl.float8e4nv, tl.float16, commuted]
  | .f8e4, .bf16, Bool.false => logSoftmaxForward%[tl.float8e4nv, tl.bfloat16, original]
  | .f8e4, .bf16, Bool.true => logSoftmaxForward%[tl.float8e4nv, tl.bfloat16, commuted]
  | .f8e4, .fp64, Bool.false => logSoftmaxForward%[tl.float8e4nv, tl.float64, original]
  | .f8e4, .fp64, Bool.true => logSoftmaxForward%[tl.float8e4nv, tl.float64, commuted]
  | .f8e4, .f8e4, Bool.false => logSoftmaxForward%[tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, .f8e4, Bool.true => logSoftmaxForward%[tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e4, .f8e5, Bool.false => logSoftmaxForward%[tl.float8e4nv, tl.float8e5, original]
  | .f8e4, .f8e5, Bool.true => logSoftmaxForward%[tl.float8e4nv, tl.float8e5, commuted]
  | .f8e5, .fp32, Bool.false => logSoftmaxForward%[tl.float8e5, tl.float32, original]
  | .f8e5, .fp32, Bool.true => logSoftmaxForward%[tl.float8e5, tl.float32, commuted]
  | .f8e5, .fp16, Bool.false => logSoftmaxForward%[tl.float8e5, tl.float16, original]
  | .f8e5, .fp16, Bool.true => logSoftmaxForward%[tl.float8e5, tl.float16, commuted]
  | .f8e5, .bf16, Bool.false => logSoftmaxForward%[tl.float8e5, tl.bfloat16, original]
  | .f8e5, .bf16, Bool.true => logSoftmaxForward%[tl.float8e5, tl.bfloat16, commuted]
  | .f8e5, .fp64, Bool.false => logSoftmaxForward%[tl.float8e5, tl.float64, original]
  | .f8e5, .fp64, Bool.true => logSoftmaxForward%[tl.float8e5, tl.float64, commuted]
  | .f8e5, .f8e4, Bool.false => logSoftmaxForward%[tl.float8e5, tl.float8e4nv, original]
  | .f8e5, .f8e4, Bool.true => logSoftmaxForward%[tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, .f8e5, Bool.false => logSoftmaxForward%[tl.float8e5, tl.float8e5, original]
  | .f8e5, .f8e5, Bool.true => logSoftmaxForward%[tl.float8e5, tl.float8e5, commuted]
```
</details>

## Public theorem: `backward_region_correct`

<details><summary>docstring</summary>

```
/-- Complete backward execution with its actual padding-dependent mathematical
formula and full raw frame, allowing arbitrary undefined values and aliases. -/
```
</details>

**Statement:**
```lean
specification backward_region_correct (input upstream output : Storage) (v : Bool)
    (Y G DX : RegionName) (M N K R C : Nat) (hK : 0 < K) (s : BlockState) :
    ∃ t, exec (typedBackward input upstream output v Y G DX M N K R C).eraseDType.toAlgKernel s = some t ∧
      (∀ (r : Fin R) (i : Fin C), s.pids 0*R+r.val<M → i.val<N →
        t.readMem DX (outOffset (s.pids 0) (s.pids 1) N K R (r,i,PUnit.unit)) =
          Math.backwardPadded C
            (Formula.paddedSample s Y (s.pids 0*R+r.val) N K (s.pids 1))
            (Formula.paddedSample s G (s.pids 0*R+r.val) N K (s.pids 1)) i.val) ∧
      (∀ r o, (∀ i : TileIndex [R,C], active (s.pids 0) M N R i →
        ¬(DX = r ∧ outOffset (s.pids 0) (s.pids 1) N K R i = o)) →
        t.mem r o = s.mem r o)
```

**Assumptions / layout contracts:**
- `hK : 0 < K`

**Closed-form spec defs (transitive):** `typedBackward`, `typedBackward_fp32_fp32`, `typedBackward_fp32_fp16`, `typedBackward_fp32_bf16`, `typedBackward_fp32_fp64`, `typedBackward_fp32_f8e4`, `typedBackward_fp32_f8e5`, `typedBackward_fp16_fp32`, `typedBackward_fp16_fp16`, `typedBackward_fp16_bf16`, `typedBackward_fp16_fp64`, `typedBackward_fp16_f8e4`, `typedBackward_fp16_f8e5`, `typedBackward_bf16_fp32`, `typedBackward_bf16_fp16`, `typedBackward_bf16_bf16`, `typedBackward_bf16_fp64`, `typedBackward_bf16_f8e4`, `typedBackward_bf16_f8e5`, `typedBackward_fp64_fp32`, `typedBackward_fp64_fp16`, `typedBackward_fp64_bf16`, `typedBackward_fp64_fp64`, `typedBackward_fp64_f8e4`, `typedBackward_fp64_f8e5`, `typedBackward_f8e4_fp32`, `typedBackward_f8e4_fp16`, `typedBackward_f8e4_bf16`, `typedBackward_f8e4_fp64`, `typedBackward_f8e4_f8e4`, `typedBackward_f8e4_f8e5`, `typedBackward_f8e5_fp32`, `typedBackward_f8e5_fp16`, `typedBackward_f8e5_bf16`, `typedBackward_f8e5_fp64`, `typedBackward_f8e5_f8e4`, `typedBackward_f8e5_f8e5`

<details><summary><code>typedBackward</code></summary>

```lean
def typedBackward (input upstream output : Storage) (v : Bool) : BackwardSource :=
  match input, upstream with
  | .fp32, .fp32 => typedBackward_fp32_fp32 output v
  | .fp32, .fp16 => typedBackward_fp32_fp16 output v
  | .fp32, .bf16 => typedBackward_fp32_bf16 output v
  | .fp32, .fp64 => typedBackward_fp32_fp64 output v
  | .fp32, .f8e4 => typedBackward_fp32_f8e4 output v
  | .fp32, .f8e5 => typedBackward_fp32_f8e5 output v
  | .fp16, .fp32 => typedBackward_fp16_fp32 output v
  | .fp16, .fp16 => typedBackward_fp16_fp16 output v
  | .fp16, .bf16 => typedBackward_fp16_bf16 output v
  | .fp16, .fp64 => typedBackward_fp16_fp64 output v
  | .fp16, .f8e4 => typedBackward_fp16_f8e4 output v
  | .fp16, .f8e5 => typedBackward_fp16_f8e5 output v
  | .bf16, .fp32 => typedBackward_bf16_fp32 output v
  | .bf16, .fp16 => typedBackward_bf16_fp16 output v
  | .bf16, .bf16 => typedBackward_bf16_bf16 output v
  | .bf16, .fp64 => typedBackward_bf16_fp64 output v
  | .bf16, .f8e4 => typedBackward_bf16_f8e4 output v
  | .bf16, .f8e5 => typedBackward_bf16_f8e5 output v
  | .fp64, .fp32 => typedBackward_fp64_fp32 output v
  | .fp64, .fp16 => typedBackward_fp64_fp16 output v
  | .fp64, .bf16 => typedBackward_fp64_bf16 output v
  | .fp64, .fp64 => typedBackward_fp64_fp64 output v
  | .fp64, .f8e4 => typedBackward_fp64_f8e4 output v
  | .fp64, .f8e5 => typedBackward_fp64_f8e5 output v
  | .f8e4, .fp32 => typedBackward_f8e4_fp32 output v
  | .f8e4, .fp16 => typedBackward_f8e4_fp16 output v
  | .f8e4, .bf16 => typedBackward_f8e4_bf16 output v
  | .f8e4, .fp64 => typedBackward_f8e4_fp64 output v
  | .f8e4, .f8e4 => typedBackward_f8e4_f8e4 output v
  | .f8e4, .f8e5 => typedBackward_f8e4_f8e5 output v
  | .f8e5, .fp32 => typedBackward_f8e5_fp32 output v
  | .f8e5, .fp16 => typedBackward_f8e5_fp16 output v
  | .f8e5, .bf16 => typedBackward_f8e5_bf16 output v
  | .f8e5, .fp64 => typedBackward_f8e5_fp64 output v
  | .f8e5, .f8e4 => typedBackward_f8e5_f8e4 output v
  | .f8e5, .f8e5 => typedBackward_f8e5_f8e5 output v
```
</details>

<details><summary><code>typedBackward_fp32_fp32</code></summary>

```lean
def typedBackward_fp32_fp32 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float32, tl.float32, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float32, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float32, tl.float32, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float32, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float32, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float32, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float32, tl.float32, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float32, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float32, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float32, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float32, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float32, tl.float32, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_fp32_fp16</code></summary>

```lean
def typedBackward_fp32_fp16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float32, tl.float16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float32, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float32, tl.float16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float32, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float32, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float32, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float32, tl.float16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float32, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float32, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float32, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float32, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float32, tl.float16, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_fp32_bf16</code></summary>

```lean
def typedBackward_fp32_bf16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_fp32_fp64</code></summary>

```lean
def typedBackward_fp32_fp64 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float32, tl.float64, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float32, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float32, tl.float64, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float32, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float32, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float32, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float32, tl.float64, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float32, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float32, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float32, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float32, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float32, tl.float64, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_fp32_f8e4</code></summary>

```lean
def typedBackward_fp32_f8e4 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_fp32_f8e5</code></summary>

```lean
def typedBackward_fp32_f8e5 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_fp16_fp32</code></summary>

```lean
def typedBackward_fp16_fp32 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float16, tl.float32, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float16, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float16, tl.float32, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float16, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float16, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float16, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float16, tl.float32, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float16, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float16, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float16, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float16, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float16, tl.float32, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_fp16_fp16</code></summary>

```lean
def typedBackward_fp16_fp16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float16, tl.float16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float16, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float16, tl.float16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float16, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float16, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float16, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float16, tl.float16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float16, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float16, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float16, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float16, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float16, tl.float16, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_fp16_bf16</code></summary>

```lean
def typedBackward_fp16_bf16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_fp16_fp64</code></summary>

```lean
def typedBackward_fp16_fp64 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float16, tl.float64, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float16, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float16, tl.float64, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float16, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float16, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float16, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float16, tl.float64, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float16, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float16, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float16, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float16, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float16, tl.float64, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_fp16_f8e4</code></summary>

```lean
def typedBackward_fp16_f8e4 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_fp16_f8e5</code></summary>

```lean
def typedBackward_fp16_f8e5 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_bf16_fp32</code></summary>

```lean
def typedBackward_bf16_fp32 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_bf16_fp16</code></summary>

```lean
def typedBackward_bf16_fp16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_bf16_bf16</code></summary>

```lean
def typedBackward_bf16_bf16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_bf16_fp64</code></summary>

```lean
def typedBackward_bf16_fp64 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_bf16_f8e4</code></summary>

```lean
def typedBackward_bf16_f8e4 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_bf16_f8e5</code></summary>

```lean
def typedBackward_bf16_f8e5 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_fp64_fp32</code></summary>

```lean
def typedBackward_fp64_fp32 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float64, tl.float32, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float64, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float64, tl.float32, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float64, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float64, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float64, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float64, tl.float32, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float64, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float64, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float64, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float64, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float64, tl.float32, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_fp64_fp16</code></summary>

```lean
def typedBackward_fp64_fp16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float64, tl.float16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float64, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float64, tl.float16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float64, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float64, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float64, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float64, tl.float16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float64, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float64, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float64, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float64, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float64, tl.float16, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_fp64_bf16</code></summary>

```lean
def typedBackward_fp64_bf16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_fp64_fp64</code></summary>

```lean
def typedBackward_fp64_fp64 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float64, tl.float64, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float64, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float64, tl.float64, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float64, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float64, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float64, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float64, tl.float64, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float64, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float64, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float64, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float64, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float64, tl.float64, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_fp64_f8e4</code></summary>

```lean
def typedBackward_fp64_f8e4 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_fp64_f8e5</code></summary>

```lean
def typedBackward_fp64_f8e5 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_f8e4_fp32</code></summary>

```lean
def typedBackward_f8e4_fp32 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_f8e4_fp16</code></summary>

```lean
def typedBackward_f8e4_fp16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_f8e4_bf16</code></summary>

```lean
def typedBackward_f8e4_bf16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_f8e4_fp64</code></summary>

```lean
def typedBackward_f8e4_fp64 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_f8e4_f8e4</code></summary>

```lean
def typedBackward_f8e4_f8e4 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_f8e4_f8e5</code></summary>

```lean
def typedBackward_f8e4_f8e5 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_f8e5_fp32</code></summary>

```lean
def typedBackward_f8e5_fp32 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_f8e5_fp16</code></summary>

```lean
def typedBackward_f8e5_fp16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_f8e5_bf16</code></summary>

```lean
def typedBackward_f8e5_bf16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_f8e5_fp64</code></summary>

```lean
def typedBackward_f8e5_fp64 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_f8e5_f8e4</code></summary>

```lean
def typedBackward_f8e5_f8e4 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.float8e5, commuted]
```
</details>

<details><summary><code>typedBackward_f8e5_f8e5</code></summary>

```lean
def typedBackward_f8e5_f8e5 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.float8e5, commuted]
```
</details>
