# Spec sheet — `bench/tritonbench_g/softmax_optimize/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/softmax_optimize/softmax_optimize.py`

## Public theorem: `full_source_equiv`

**Statement:**
```lean
specification full_source_equiv (input : Storage) (output : Accumulation)
    (Y X : RegionName) (M N B : Nat) :
    program (kernel input output Bool.false Y X M N B) ≡[assumptions]
    program (kernel input output Bool.true Y X M N B)
```

**Closed-form spec defs (transitive):** `program`, `kernel`, `assumptions`

<details><summary><code>program</code></summary>

```lean
def program (k : ComputeKernel) : Program := ⟨k, []⟩
```
</details>

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

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>
