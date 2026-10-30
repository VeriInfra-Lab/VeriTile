# Spec sheet — `bench/tritonbench_g/nested_loops_processing/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/nested_loops_processing/nested_loops_processing.py`

## Public theorem: `source_equivalence`

<details><summary>docstring</summary>

```
/-- Full nested-loop state equivalence, including aliases and evaluator failures.
Source-facing address safety and nonnegative strides belong to the real IO contract. -/
```
</details>

**Statement:**
```lean
specification source_equivalence (storage : Storage)
    (src dst : RegionName) (sm sn : Int) :
    program storage Bool.false src dst sm sn ≡[assumptions]
      program storage Bool.true src dst sm sn
```

**Closed-form spec defs (transitive):** `program`, `assumptions`, `kernel`

<details><summary><code>program</code></summary>

```
/-- The default fp32 backend profile retains explicit source-local fp64 loads.
No payload arithmetic or conversion is introduced for homogeneous storage. -/
```
```lean
def program (storage : Storage) (candidate : Bool)
    (src dst : RegionName) (sm sn : Int) : Program where
  kernel := kernel storage candidate src dst sm sn
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>kernel</code></summary>

```lean
def kernel (storage : Storage) (candidate : Bool) : Source :=
  match storage, candidate with
  | .int8, Bool.false => nestedCopy%[tl.int8, original]
  | .int8, Bool.true => nestedCopy%[tl.int8, commuted]
  | .int16, Bool.false => nestedCopy%[tl.int16, original]
  | .int16, Bool.true => nestedCopy%[tl.int16, commuted]
  | .int32, Bool.false => nestedCopy%[tl.int32, original]
  | .int32, Bool.true => nestedCopy%[tl.int32, commuted]
  | .int64, Bool.false => nestedCopy%[tl.int64, original]
  | .int64, Bool.true => nestedCopy%[tl.int64, commuted]
  | .uint8, Bool.false => nestedCopy%[tl.uint8, original]
  | .uint8, Bool.true => nestedCopy%[tl.uint8, commuted]
  | .uint16, Bool.false => nestedCopy%[tl.uint16, original]
  | .uint16, Bool.true => nestedCopy%[tl.uint16, commuted]
  | .uint32, Bool.false => nestedCopy%[tl.uint32, original]
  | .uint32, Bool.true => nestedCopy%[tl.uint32, commuted]
  | .uint64, Bool.false => nestedCopy%[tl.uint64, original]
  | .uint64, Bool.true => nestedCopy%[tl.uint64, commuted]
  | .boolean, Bool.false => nestedCopy%[tl.int1, original]
  | .boolean, Bool.true => nestedCopy%[tl.int1, commuted]
  | .fp32, Bool.false => nestedCopy%[tl.float32, original]
  | .fp32, Bool.true => nestedCopy%[tl.float32, commuted]
  | .fp16, Bool.false => nestedCopy%[tl.float16, original]
  | .fp16, Bool.true => nestedCopy%[tl.float16, commuted]
  | .bf16, Bool.false => nestedCopy%[tl.bfloat16, original]
  | .bf16, Bool.true => nestedCopy%[tl.bfloat16, commuted]
  | .fp64, Bool.false => nestedCopy%[tl.float64, original]
  | .fp64, Bool.true => nestedCopy%[tl.float64, commuted]
  | .f8e4, Bool.false => nestedCopy%[tl.float8e4nv, original]
  | .f8e4, Bool.true => nestedCopy%[tl.float8e4nv, commuted]
  | .f8e5, Bool.false => nestedCopy%[tl.float8e5, original]
  | .f8e5, Bool.true => nestedCopy%[tl.float8e5, commuted]

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.TypedSource

#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>
