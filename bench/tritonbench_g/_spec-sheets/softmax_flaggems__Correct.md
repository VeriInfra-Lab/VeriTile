# Spec sheet — `bench/tritonbench_g/softmax_flaggems/Correct.lean`

**Python source:** `bench/tritonbench_g/softmax_flaggems/softmax_flaggems.py`

## Public theorem: `forward_non_inner_correctness`

<details><summary>docstring</summary>

```
/-! Complete source contracts for both versions of all four kernels. Real inputs
are finite mathematical reals. Forward results are full-column softmax; backward
results retain the source's arbitrary masked-load padding contribution. See the
separate formula lemmas for the conventional VJP under vanishing padding.
SourceProfileSupported excludes original fp64 fallback specializations whose
loop-carried types change. ForwardSourceDomain ensures every fallback lane has
an initially active finite input. Mathematical projection lemmas remain more general. -/
```
</details>

**Statement:**
```lean
specification forward_non_inner_correctness (input output : Storage)
    (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (Y X : RegionName) (M N K B C : Nat) (one : Bool)
    (_hprofile : SourceProfileSupported input one)
    (_hforward : ForwardSourceDomain one N B)
    (h : RealForwardNonInner.ExecutionDomain one X Y N B)
    (hr : s.pids 0 < M) (hX : M * N * K ≤ A.extent X)
    (hY : M * N * K ≤ A.extent Y)
    (hYr : Y ∈ A.regions) :
    SourceCorrect A s (softmax_kernel_non_inner input output Y X M N K B C one)
      (RealForwardNonInner.IOResult s · Y X N K C) (RealForwardNonInner.FlatIOResult A s · Y X N K C) ∧
    SourceCorrect A s (softmax_kernel_non_inner_candidate input output Y X M N K B C one)
      (RealForwardNonInner.IOResult s · Y X N K C) (RealForwardNonInner.FlatIOResult A s · Y X N K C)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hcov : ∀ r, r ∉ A.regions → A.extent r = 0`
- `h : RealForwardNonInner.ExecutionDomain one X Y N B`
- `hr : s.pids 0 < M`
- `hX : M * N * K ≤ A.extent X`
- `hY : M * N * K ≤ A.extent Y`
- `hYr : Y ∈ A.regions`

**Closed-form spec defs (transitive):** `softmax_kernel_non_inner`, `softmax_kernel_non_inner_candidate`

<details><summary><code>softmax_kernel_non_inner</code></summary>

```lean
def softmax_kernel_non_inner (inputType outputType : Storage) :=
  match inputType, outputType with
  | .fp16, .fp16 => flaggems0%[tl.float16, tl.float16, original]
  | .fp16, .bf16 => flaggems0%[tl.float16, tl.bfloat16, original]
  | .fp16, .fp32 => flaggems0%[tl.float16, tl.float32, original]
  | .fp16, .fp64 => flaggems0%[tl.float16, tl.float64, original]
  | .bf16, .fp16 => flaggems0%[tl.bfloat16, tl.float16, original]
  | .bf16, .bf16 => flaggems0%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .fp32 => flaggems0%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp64 => flaggems0%[tl.bfloat16, tl.float64, original]
  | .fp32, .fp16 => flaggems0%[tl.float32, tl.float16, original]
  | .fp32, .bf16 => flaggems0%[tl.float32, tl.bfloat16, original]
  | .fp32, .fp32 => flaggems0%[tl.float32, tl.float32, original]
  | .fp32, .fp64 => flaggems0%[tl.float32, tl.float64, original]
  | .fp64, .fp16 => flaggems0%[tl.float64, tl.float16, original]
  | .fp64, .bf16 => flaggems0%[tl.float64, tl.bfloat16, original]
  | .fp64, .fp32 => flaggems0%[tl.float64, tl.float32, original]
  | .fp64, .fp64 => flaggems0%[tl.float64, tl.float64, original]
```
</details>

<details><summary><code>softmax_kernel_non_inner_candidate</code></summary>

```lean
def softmax_kernel_non_inner_candidate (inputType outputType : Storage) :=
  match inputType, outputType with
  | .fp16, .fp16 => flaggems0%[tl.float16, tl.float16, address_commuted]
  | .fp16, .bf16 => flaggems0%[tl.float16, tl.bfloat16, address_commuted]
  | .fp16, .fp32 => flaggems0%[tl.float16, tl.float32, address_commuted]
  | .fp16, .fp64 => flaggems0%[tl.float16, tl.float64, address_commuted]
  | .bf16, .fp16 => flaggems0%[tl.bfloat16, tl.float16, address_commuted]
  | .bf16, .bf16 => flaggems0%[tl.bfloat16, tl.bfloat16, address_commuted]
  | .bf16, .fp32 => flaggems0%[tl.bfloat16, tl.float32, address_commuted]
  | .bf16, .fp64 => flaggems0%[tl.bfloat16, tl.float64, address_commuted]
  | .fp32, .fp16 => flaggems0%[tl.float32, tl.float16, address_commuted]
  | .fp32, .bf16 => flaggems0%[tl.float32, tl.bfloat16, address_commuted]
  | .fp32, .fp32 => flaggems0%[tl.float32, tl.float32, address_commuted]
  | .fp32, .fp64 => flaggems0%[tl.float32, tl.float64, address_commuted]
  | .fp64, .fp16 => flaggems0%[tl.float64, tl.float16, address_commuted]
  | .fp64, .bf16 => flaggems0%[tl.float64, tl.bfloat16, address_commuted]
  | .fp64, .fp32 => flaggems0%[tl.float64, tl.float32, address_commuted]
  | .fp64, .fp64 => flaggems0%[tl.float64, tl.float64, address_commuted]
```
</details>

## Public theorem: `forward_inner_correctness`

**Statement:**
```lean
specification forward_inner_correctness (input output : Storage)
    (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (Y X : RegionName) (M N B : Nat) (one : Bool)
    (_hprofile : SourceProfileSupported input one)
    (_hforward : ForwardSourceDomain one N B)
    (h : RealForwardInner.ExecutionDomain one X Y N B)
    (hr : s.pids 0 < M) (hX : M * N ≤ A.extent X)
    (hY : M * N ≤ A.extent Y)
    (hYr : Y ∈ A.regions) :
    SourceCorrect A s (softmax_kernel_inner input output Y X M N B one)
      (RealForwardInner.IOResult s · Y X N) (RealForwardInner.FlatIOResult A s · Y X N) ∧
    SourceCorrect A s (softmax_kernel_inner_candidate input output Y X M N B one)
      (RealForwardInner.IOResult s · Y X N) (RealForwardInner.FlatIOResult A s · Y X N)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hcov : ∀ r, r ∉ A.regions → A.extent r = 0`
- `h : RealForwardInner.ExecutionDomain one X Y N B`
- `hr : s.pids 0 < M`
- `hX : M * N ≤ A.extent X`
- `hY : M * N ≤ A.extent Y`
- `hYr : Y ∈ A.regions`

**Closed-form spec defs (transitive):** `softmax_kernel_inner`, `softmax_kernel_inner_candidate`

<details><summary><code>softmax_kernel_inner</code></summary>

```lean
def softmax_kernel_inner (inputType outputType : Storage) :=
  match inputType, outputType with
  | .fp16, .fp16 => flaggems1%[tl.float16, tl.float16, original]
  | .fp16, .bf16 => flaggems1%[tl.float16, tl.bfloat16, original]
  | .fp16, .fp32 => flaggems1%[tl.float16, tl.float32, original]
  | .fp16, .fp64 => flaggems1%[tl.float16, tl.float64, original]
  | .bf16, .fp16 => flaggems1%[tl.bfloat16, tl.float16, original]
  | .bf16, .bf16 => flaggems1%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .fp32 => flaggems1%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp64 => flaggems1%[tl.bfloat16, tl.float64, original]
  | .fp32, .fp16 => flaggems1%[tl.float32, tl.float16, original]
  | .fp32, .bf16 => flaggems1%[tl.float32, tl.bfloat16, original]
  | .fp32, .fp32 => flaggems1%[tl.float32, tl.float32, original]
  | .fp32, .fp64 => flaggems1%[tl.float32, tl.float64, original]
  | .fp64, .fp16 => flaggems1%[tl.float64, tl.float16, original]
  | .fp64, .bf16 => flaggems1%[tl.float64, tl.bfloat16, original]
  | .fp64, .fp32 => flaggems1%[tl.float64, tl.float32, original]
  | .fp64, .fp64 => flaggems1%[tl.float64, tl.float64, original]
```
</details>

<details><summary><code>softmax_kernel_inner_candidate</code></summary>

```lean
def softmax_kernel_inner_candidate (inputType outputType : Storage) :=
  match inputType, outputType with
  | .fp16, .fp16 => flaggems1%[tl.float16, tl.float16, address_commuted]
  | .fp16, .bf16 => flaggems1%[tl.float16, tl.bfloat16, address_commuted]
  | .fp16, .fp32 => flaggems1%[tl.float16, tl.float32, address_commuted]
  | .fp16, .fp64 => flaggems1%[tl.float16, tl.float64, address_commuted]
  | .bf16, .fp16 => flaggems1%[tl.bfloat16, tl.float16, address_commuted]
  | .bf16, .bf16 => flaggems1%[tl.bfloat16, tl.bfloat16, address_commuted]
  | .bf16, .fp32 => flaggems1%[tl.bfloat16, tl.float32, address_commuted]
  | .bf16, .fp64 => flaggems1%[tl.bfloat16, tl.float64, address_commuted]
  | .fp32, .fp16 => flaggems1%[tl.float32, tl.float16, address_commuted]
  | .fp32, .bf16 => flaggems1%[tl.float32, tl.bfloat16, address_commuted]
  | .fp32, .fp32 => flaggems1%[tl.float32, tl.float32, address_commuted]
  | .fp32, .fp64 => flaggems1%[tl.float32, tl.float64, address_commuted]
  | .fp64, .fp16 => flaggems1%[tl.float64, tl.float16, address_commuted]
  | .fp64, .bf16 => flaggems1%[tl.float64, tl.bfloat16, address_commuted]
  | .fp64, .fp32 => flaggems1%[tl.float64, tl.float32, address_commuted]
  | .fp64, .fp64 => flaggems1%[tl.float64, tl.float64, address_commuted]
```
</details>

## Public theorem: `backward_non_inner_correctness`

**Statement:**
```lean
specification backward_non_inner_correctness (storage : Storage)
    (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (P G Y : RegionName) (M N K B C : Nat) (one : Bool)
    (_hprofile : SourceProfileSupported storage one)
    (h : RealBackwardNonInner.ExecutionDomain one P G Y N B)
    (hr : s.pids 0 < M) (hP : M * N * K ≤ A.extent P)
    (hG : M * N * K ≤ A.extent G)
    (hY : M * N * K ≤ A.extent Y)
    (hYr : Y ∈ A.regions) :
    SourceCorrect A s (softmax_backward_kernel_non_inner storage P G Y M N K B C one)
      (RealBackwardNonInner.IOResult s · P G Y N K B C one) (RealBackwardNonInner.FlatIOResult A s · P G Y N K B C one) ∧
    SourceCorrect A s (softmax_backward_kernel_non_inner_candidate storage P G Y M N K B C one)
      (RealBackwardNonInner.IOResult s · P G Y N K B C one) (RealBackwardNonInner.FlatIOResult A s · P G Y N K B C one)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hcov : ∀ r, r ∉ A.regions → A.extent r = 0`
- `h : RealBackwardNonInner.ExecutionDomain one P G Y N B`
- `hr : s.pids 0 < M`
- `hP : M * N * K ≤ A.extent P`
- `hG : M * N * K ≤ A.extent G`
- `hY : M * N * K ≤ A.extent Y`
- `hYr : Y ∈ A.regions`

**Closed-form spec defs (transitive):** `softmax_backward_kernel_non_inner`, `softmax_backward_kernel_non_inner_candidate`

<details><summary><code>softmax_backward_kernel_non_inner</code></summary>

```lean
def softmax_backward_kernel_non_inner (storage : Storage) :=
  match storage with
  | .fp16 => flaggems2%[tl.float16, tl.float16, original]
  | .bf16 => flaggems2%[tl.bfloat16, tl.bfloat16, original]
  | .fp32 => flaggems2%[tl.float32, tl.float32, original]
  | .fp64 => flaggems2%[tl.float64, tl.float64, original]
```
</details>

<details><summary><code>softmax_backward_kernel_non_inner_candidate</code></summary>

```lean
def softmax_backward_kernel_non_inner_candidate (storage : Storage) :=
  match storage with
  | .fp16 => flaggems2%[tl.float16, tl.float16, address_commuted]
  | .bf16 => flaggems2%[tl.bfloat16, tl.bfloat16, address_commuted]
  | .fp32 => flaggems2%[tl.float32, tl.float32, address_commuted]
  | .fp64 => flaggems2%[tl.float64, tl.float64, address_commuted]
```
</details>

## Public theorem: `backward_inner_correctness`

**Statement:**
```lean
specification backward_inner_correctness (storage : Storage)
    (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (P G Y : RegionName) (M N B C : Nat) (one : Bool)
    (_hprofile : SourceProfileSupported storage one)
    (h : RealBackwardInner.ExecutionDomain one P G Y N B)
    (hP : M * N ≤ A.extent P)
    (hG : M * N ≤ A.extent G)
    (hY : M * N ≤ A.extent Y)
    (hYr : Y ∈ A.regions) :
    SourceCorrect A s (softmax_backward_kernel_inner storage P G Y M N C B one)
      (RealBackwardInner.IOResult s · P G Y M N B C one) (RealBackwardInner.FlatIOResult A s · P G Y M N B C one) ∧
    SourceCorrect A s (softmax_backward_kernel_inner_candidate storage P G Y M N C B one)
      (RealBackwardInner.IOResult s · P G Y M N B C one) (RealBackwardInner.FlatIOResult A s · P G Y M N B C one)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hcov : ∀ r, r ∉ A.regions → A.extent r = 0`
- `h : RealBackwardInner.ExecutionDomain one P G Y N B`
- `hP : M * N ≤ A.extent P`
- `hG : M * N ≤ A.extent G`
- `hY : M * N ≤ A.extent Y`
- `hYr : Y ∈ A.regions`

**Closed-form spec defs (transitive):** `softmax_backward_kernel_inner`, `softmax_backward_kernel_inner_candidate`

<details><summary><code>softmax_backward_kernel_inner</code></summary>

```lean
def softmax_backward_kernel_inner (storage : Storage) :=
  match storage with
  | .fp16 => flaggems3%[tl.float16, tl.float16, original]
  | .bf16 => flaggems3%[tl.bfloat16, tl.bfloat16, original]
  | .fp32 => flaggems3%[tl.float32, tl.float32, original]
  | .fp64 => flaggems3%[tl.float64, tl.float64, original]
```
</details>

<details><summary><code>softmax_backward_kernel_inner_candidate</code></summary>

```lean
def softmax_backward_kernel_inner_candidate (storage : Storage) :=
  match storage with
  | .fp16 => flaggems3%[tl.float16, tl.float16, address_commuted]
  | .bf16 => flaggems3%[tl.bfloat16, tl.bfloat16, address_commuted]
  | .fp32 => flaggems3%[tl.float32, tl.float32, address_commuted]
  | .fp64 => flaggems3%[tl.float64, tl.float64, address_commuted]
```
</details>

## Also present (pinned special-case summaries)
- `softmax_kernel_inner_one_tile_compute_correct`
- `softmax_kernel_non_inner_one_tile_compute_correct`
- `softmax_backward_kernel_inner_one_tile_compute_correct`
- `softmax_backward_kernel_non_inner_one_tile_compute_correct`
