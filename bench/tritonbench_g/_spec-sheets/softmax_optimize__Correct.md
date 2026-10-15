# Spec sheet — `bench/tritonbench_g/softmax_optimize/Correct.lean`

**Python source:** `bench/tritonbench_g/softmax_optimize/softmax_optimize.py`

## Public theorem: `full_source_original_correct`

<details><summary>docstring</summary>

```
/-- Complete signed four-loop original: successful full-row softmax IO, exact
row frame and access safety, with the flat bridge's zero-undef premise explicit.
The numerical domain excludes never-active accumulator lanes; the exact source
and its FP equivalence separately retain N=0 signed control flow. -/
```
</details>

**Statement:**
```lean
specification full_source_original_correct
    (input : Storage) (output : Accumulation)
    (A : FlatAlloc) (hd : A.Disjoint) (hcov : ∀ r, r∉A.regions → A.extent r=0)
    (s : BlockState) (Y X : RegionName) (M N B : Nat)
    (hdom : NumericalDomain N B) (hYX : Y≠X) (hr : s.pids 0<M)
    (hX : M*N≤A.extent X) (hY : M*N≤A.extent Y) (hYr : Y∈A.regions) :
    SourceCorrect A s (kernel input output Bool.false Y X M N B) Y X N
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hcov : ∀ r, r∉A.regions → A.extent r=0`
- `hdom : NumericalDomain N B`
- `hYX : Y≠X`
- `hr : s.pids 0<M`
- `hX : M*N≤A.extent X`
- `hY : M*N≤A.extent Y`
- `hYr : Y∈A.regions`

**Closed-form spec defs (transitive):** `kernel`

<details><summary><code>kernel</code></summary>

```lean
def kernel (input : Storage) (output : Accumulation) (v : Bool) : Source :=
  match input, output, v with
  | .fp32, .fp32, Bool.false => softmaxOptimizeFull%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => softmaxOptimizeFull%[tl.float32, tl.float32, commuted]
  | .fp32, .fp64, Bool.false => softmaxOptimizeFull%[tl.float32, tl.float64, original]
  | .fp32, .fp64, Bool.true => softmaxOptimizeFull%[tl.float32, tl.float64, commuted]
  | .fp16, .fp32, Bool.false => softmaxOptimizeFull%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => softmaxOptimizeFull%[tl.float16, tl.float32, commuted]
  | .fp16, .fp64, Bool.false => softmaxOptimizeFull%[tl.float16, tl.float64, original]
  | .fp16, .fp64, Bool.true => softmaxOptimizeFull%[tl.float16, tl.float64, commuted]
  | .bf16, .fp32, Bool.false => softmaxOptimizeFull%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => softmaxOptimizeFull%[tl.bfloat16, tl.float32, commuted]
  | .bf16, .fp64, Bool.false => softmaxOptimizeFull%[tl.bfloat16, tl.float64, original]
  | .bf16, .fp64, Bool.true => softmaxOptimizeFull%[tl.bfloat16, tl.float64, commuted]
  | .fp64, .fp32, Bool.false => softmaxOptimizeFull%[tl.float64, tl.float32, original]
  | .fp64, .fp32, Bool.true => softmaxOptimizeFull%[tl.float64, tl.float32, commuted]
  | .fp64, .fp64, Bool.false => softmaxOptimizeFull%[tl.float64, tl.float64, original]
  | .fp64, .fp64, Bool.true => softmaxOptimizeFull%[tl.float64, tl.float64, commuted]
  | .f8e4, .fp32, Bool.false => softmaxOptimizeFull%[tl.float8e4nv, tl.float32, original]
  | .f8e4, .fp32, Bool.true => softmaxOptimizeFull%[tl.float8e4nv, tl.float32, commuted]
  | .f8e4, .fp64, Bool.false => softmaxOptimizeFull%[tl.float8e4nv, tl.float64, original]
  | .f8e4, .fp64, Bool.true => softmaxOptimizeFull%[tl.float8e4nv, tl.float64, commuted]
  | .f8e5, .fp32, Bool.false => softmaxOptimizeFull%[tl.float8e5, tl.float32, original]
  | .f8e5, .fp32, Bool.true => softmaxOptimizeFull%[tl.float8e5, tl.float32, commuted]
  | .f8e5, .fp64, Bool.false => softmaxOptimizeFull%[tl.float8e5, tl.float64, original]
  | .f8e5, .fp64, Bool.true => softmaxOptimizeFull%[tl.float8e5, tl.float64, commuted]
```
</details>

## Public theorem: `full_source_candidate_correct`

<details><summary>docstring</summary>

```
/-- Complete signed four-loop candidate: successful full-row softmax IO, exact
row frame and access safety, with the flat bridge's zero-undef premise explicit.
The numerical domain excludes never-active accumulator lanes; the exact source
and its FP equivalence separately retain N=0 signed control flow. -/
```
</details>

**Statement:**
```lean
specification full_source_candidate_correct
    (input : Storage) (output : Accumulation)
    (A : FlatAlloc) (hd : A.Disjoint) (hcov : ∀ r, r∉A.regions → A.extent r=0)
    (s : BlockState) (Y X : RegionName) (M N B : Nat)
    (hdom : NumericalDomain N B) (hYX : Y≠X) (hr : s.pids 0<M)
    (hX : M*N≤A.extent X) (hY : M*N≤A.extent Y) (hYr : Y∈A.regions) :
    SourceCorrect A s (kernel input output Bool.true Y X M N B) Y X N
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hcov : ∀ r, r∉A.regions → A.extent r=0`
- `hdom : NumericalDomain N B`
- `hYX : Y≠X`
- `hr : s.pids 0<M`
- `hX : M*N≤A.extent X`
- `hY : M*N≤A.extent Y`
- `hYr : Y∈A.regions`

**Closed-form spec defs (transitive):** `kernel`

<details><summary><code>kernel</code></summary>

```lean
def kernel (input : Storage) (output : Accumulation) (v : Bool) : Source :=
  match input, output, v with
  | .fp32, .fp32, Bool.false => softmaxOptimizeFull%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => softmaxOptimizeFull%[tl.float32, tl.float32, commuted]
  | .fp32, .fp64, Bool.false => softmaxOptimizeFull%[tl.float32, tl.float64, original]
  | .fp32, .fp64, Bool.true => softmaxOptimizeFull%[tl.float32, tl.float64, commuted]
  | .fp16, .fp32, Bool.false => softmaxOptimizeFull%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => softmaxOptimizeFull%[tl.float16, tl.float32, commuted]
  | .fp16, .fp64, Bool.false => softmaxOptimizeFull%[tl.float16, tl.float64, original]
  | .fp16, .fp64, Bool.true => softmaxOptimizeFull%[tl.float16, tl.float64, commuted]
  | .bf16, .fp32, Bool.false => softmaxOptimizeFull%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => softmaxOptimizeFull%[tl.bfloat16, tl.float32, commuted]
  | .bf16, .fp64, Bool.false => softmaxOptimizeFull%[tl.bfloat16, tl.float64, original]
  | .bf16, .fp64, Bool.true => softmaxOptimizeFull%[tl.bfloat16, tl.float64, commuted]
  | .fp64, .fp32, Bool.false => softmaxOptimizeFull%[tl.float64, tl.float32, original]
  | .fp64, .fp32, Bool.true => softmaxOptimizeFull%[tl.float64, tl.float32, commuted]
  | .fp64, .fp64, Bool.false => softmaxOptimizeFull%[tl.float64, tl.float64, original]
  | .fp64, .fp64, Bool.true => softmaxOptimizeFull%[tl.float64, tl.float64, commuted]
  | .f8e4, .fp32, Bool.false => softmaxOptimizeFull%[tl.float8e4nv, tl.float32, original]
  | .f8e4, .fp32, Bool.true => softmaxOptimizeFull%[tl.float8e4nv, tl.float32, commuted]
  | .f8e4, .fp64, Bool.false => softmaxOptimizeFull%[tl.float8e4nv, tl.float64, original]
  | .f8e4, .fp64, Bool.true => softmaxOptimizeFull%[tl.float8e4nv, tl.float64, commuted]
  | .f8e5, .fp32, Bool.false => softmaxOptimizeFull%[tl.float8e5, tl.float32, original]
  | .f8e5, .fp32, Bool.true => softmaxOptimizeFull%[tl.float8e5, tl.float32, commuted]
  | .f8e5, .fp64, Bool.false => softmaxOptimizeFull%[tl.float8e5, tl.float64, original]
  | .f8e5, .fp64, Bool.true => softmaxOptimizeFull%[tl.float8e5, tl.float64, commuted]
```
</details>
