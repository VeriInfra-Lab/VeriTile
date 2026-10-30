# Spec sheet — `bench/tritonbench_g/ksoftmax_triton/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/ksoftmax_triton/ksoftmax_triton.py`

## Public theorem: `ksoftmax_backward_index_equiv`

<details><summary>docstring</summary>

```
/-- Whole typed state and failure equality for all independent storage triples
and flags. No floating arithmetic law or domain assumption is required. -/
```
</details>

**Statement:**
```lean
specification ksoftmax_backward_index_equiv
    (gradient saved destination : Backward.Typed.Storage)
    (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) :
    program gradient saved destination Bool.false GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16 ≡[assumptions]
    program gradient saved destination Bool.true GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16
```

**Closed-form spec defs (transitive):** `program`, `assumptions`

<details><summary><code>program</code></summary>

```lean
def program (gradient saved destination : Backward.Typed.Storage) (v : Bool)
    (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) : Program :=
  Backward.Floating.fullProgram gradient saved destination v GradIn GradOut Out
    bm bn gm gn om on K D log causal fp16
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

## Public theorem: `ksoftmax_forward_index_equiv`

<details><summary>docstring</summary>

```
/-- Complete typed original/candidate equivalence for the four checked
source profiles. Forward.Floating.assumptions is the empty list. -/
```
</details>

**Statement:**
```lean
specification ksoftmax_forward_index_equiv
    (profile : Forward.Typed.CheckedProfile)
    (Y X M : RegionName) (ym yn xm xn sm K D : Nat)
    (log : Bool) (mode : Option String) (causal fp16 : Bool) :
    Forward.Floating.program profile Bool.false Y X M ym yn xm xn sm K D
      log mode causal fp16 ≡[Forward.Floating.assumptions]
    Forward.Floating.program profile Bool.true Y X M ym yn xm xn sm K D
      log mode causal fp16
```

**Closed-form spec defs (transitive):** `program`, `assumptions`

<details><summary><code>program</code></summary>

```lean
def program (gradient saved destination : Backward.Typed.Storage) (v : Bool)
    (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) : Program :=
  Backward.Floating.fullProgram gradient saved destination v GradIn GradOut Out
    bm bn gm gn om on K D log causal fp16
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>
