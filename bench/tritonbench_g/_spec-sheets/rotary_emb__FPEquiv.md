# Spec sheet — `bench/tritonbench_g/rotary_emb/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/rotary_emb/rotary_emb.py`

## Public theorem: `source_equivalence`

<details><summary>docstring</summary>

```
/-- All four stores, every register, arbitrary aliases and undefined values, and
execution failure are preserved. The only change is exact natural multiplication;
no numerical admission or finite-input premise is required. This is typed-AST
equivalence, not native acceptance of every storage profile; homogeneous fp8
fails native lowering in the recorded Triton 3.7.1 sm90 source probe. -/
```
</details>

**Statement:**
```lean
specification source_equivalence (q k cos sin : Storage) (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) :
    program q k cos sin Bool.false Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd
      L HQ HK BH BS D ≡[assumptions]
    program q k cos sin Bool.true Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd
      L HQ HK BH BS D
```

**Closed-form spec defs (transitive):** `program`, `assumptions`, `kernel`, `original`, `template`

<details><summary><code>program</code></summary>

```lean
def program (q k cos sin : Storage) (v : Bool) (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) : Program :=
  ⟨kernel q k cos sin v Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd
    L HQ HK BH BS D, []⟩
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>kernel</code></summary>

```
/-- The sole candidate change commutes exact integer factors of the head base. -/
```
```lean
def kernel (q k cos sin : Storage) (v : Bool) (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) : ComputeKernel :=
  let a := original q k cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D
  if v then ComputeKernel.mk a.inputs a.outputs
    (a.surfaceBody.take 2 ++
      [.assign .nat [BH] "cur_head_range" (.alg (.add .nat .scalarL
        (.mul .nat .nil (.constNat BH) (.ref .nat [] "cur_head_index")) (.arange BH)))] ++
      a.surfaceBody.drop 3)
  else a
```
</details>

<details><summary><code>original</code></summary>

```
/-- All four source stores with independent Q/K/Cos/Sin storage. The first
20 statements include Q stores; K is loaded only after those writes. -/
```
```lean
def original (q k cos sin : Storage) (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) : ComputeKernel :=
  let a := template q cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D
  let b := template k cos sin Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D
  ComputeKernel.mk [Q,K,Cos,Sin] [Q,K] (a.surfaceBody.take 20 ++ b.surfaceBody.drop 20)
```
</details>

<details><summary><code>template</code></summary>

```
/-- Source template with independent data/cosine/sine storage. Q and K phases
are composed below to give independent Q/K types without duplicating the AST. -/
```
```lean
def template (data cos sin : Storage) : Source :=
  match data, cos with
  | .fp32, .fp32 => template_fp32_fp32 sin
  | .fp32, .fp64 => template_fp32_fp64 sin
  | .fp32, .fp16 => template_fp32_fp16 sin
  | .fp32, .bf16 => template_fp32_bf16 sin
  | .fp32, .f8e4 => template_fp32_f8e4 sin
  | .fp32, .f8e5 => template_fp32_f8e5 sin
  | .fp64, .fp32 => template_fp64_fp32 sin
  | .fp64, .fp64 => template_fp64_fp64 sin
  | .fp64, .fp16 => template_fp64_fp16 sin
  | .fp64, .bf16 => template_fp64_bf16 sin
  | .fp64, .f8e4 => template_fp64_f8e4 sin
  | .fp64, .f8e5 => template_fp64_f8e5 sin
  | .fp16, .fp32 => template_fp16_fp32 sin
  | .fp16, .fp64 => template_fp16_fp64 sin
  | .fp16, .fp16 => template_fp16_fp16 sin
  | .fp16, .bf16 => template_fp16_bf16 sin
  | .fp16, .f8e4 => template_fp16_f8e4 sin
  | .fp16, .f8e5 => template_fp16_f8e5 sin
  | .bf16, .fp32 => template_bf16_fp32 sin
  | .bf16, .fp64 => template_bf16_fp64 sin
  | .bf16, .fp16 => template_bf16_fp16 sin
  | .bf16, .bf16 => template_bf16_bf16 sin
  | .bf16, .f8e4 => template_bf16_f8e4 sin
  | .bf16, .f8e5 => template_bf16_f8e5 sin
  | .f8e4, .fp32 => template_f8e4_fp32 sin
  | .f8e4, .fp64 => template_f8e4_fp64 sin
  | .f8e4, .fp16 => template_f8e4_fp16 sin
  | .f8e4, .bf16 => template_f8e4_bf16 sin
  | .f8e4, .f8e4 => template_f8e4_f8e4 sin
  | .f8e4, .f8e5 => template_f8e4_f8e5 sin
  | .f8e5, .fp32 => template_f8e5_fp32 sin
  | .f8e5, .fp64 => template_f8e5_fp64 sin
  | .f8e5, .fp16 => template_f8e5_fp16 sin
  | .f8e5, .bf16 => template_f8e5_bf16 sin
  | .f8e5, .f8e4 => template_f8e5_f8e4 sin
  | .f8e5, .f8e5 => template_f8e5_f8e5 sin
```
</details>
