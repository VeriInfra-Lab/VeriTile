# Spec sheet — `bench/tritonbench_g/var_len_copy/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/var_len_copy/var_len_copy.py`

## Public theorem: `source_equivalence`

<details><summary>docstring</summary>

```
/-- Complete state and failure equality covers every signed loop iteration,
all aliases and every floating storage pair, including the source's overcopy. -/
```
</details>

**Statement:**
```lean
specification source_equivalence (input output : Storage)
    (oldWidth lengthWidth newWidth : Metadata)
    (os ol ns : Region .int) (src dst : RegionName) (B : Nat) :
    program input output oldWidth lengthWidth newWidth Bool.false os ol ns src dst B ≡[assumptions]
      program input output oldWidth lengthWidth newWidth Bool.true os ol ns src dst B
```

**Closed-form spec defs (transitive):** `program`, `assumptions`, `kernel`, `typedKernel`

<details><summary><code>program</code></summary>

```
/-- The engine uses its default fp32 profile; source fp64 and storage conversion
nodes retain their explicit local precision. Metadata widths share Int semantics. -/
```
```lean
def program (input output : Storage) (oldWidth lengthWidth newWidth : Metadata) (v : Bool)
    (os ol ns : Region .int) (src dst : RegionName) (B : Nat) : Program where
  kernel := typedKernel input output oldWidth lengthWidth newWidth v os ol ns src dst B
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```
/-- No floating algebraic law is used by signed address commutation. -/
```
```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>kernel</code></summary>

```
/-- Homogeneous raw copies have no payload conversion or arithmetic. -/
```
```lean
def kernel (storage : Storage) (v : Bool) : TypedSource.Source :=
  match storage, v with
  | .int8, Bool.false => varLenCopy%[tl.int8, tl.int8, original]
  | .int8, Bool.true => varLenCopy%[tl.int8, tl.int8, commuted]
  | .int16, Bool.false => varLenCopy%[tl.int16, tl.int16, original]
  | .int16, Bool.true => varLenCopy%[tl.int16, tl.int16, commuted]
  | .int32, Bool.false => varLenCopy%[tl.int32, tl.int32, original]
  | .int32, Bool.true => varLenCopy%[tl.int32, tl.int32, commuted]
  | .int64, Bool.false => varLenCopy%[tl.int64, tl.int64, original]
  | .int64, Bool.true => varLenCopy%[tl.int64, tl.int64, commuted]
  | .uint8, Bool.false => varLenCopy%[tl.uint8, tl.uint8, original]
  | .uint8, Bool.true => varLenCopy%[tl.uint8, tl.uint8, commuted]
  | .uint16, Bool.false => varLenCopy%[tl.uint16, tl.uint16, original]
  | .uint16, Bool.true => varLenCopy%[tl.uint16, tl.uint16, commuted]
  | .uint32, Bool.false => varLenCopy%[tl.uint32, tl.uint32, original]
  | .uint32, Bool.true => varLenCopy%[tl.uint32, tl.uint32, commuted]
  | .uint64, Bool.false => varLenCopy%[tl.uint64, tl.uint64, original]
  | .uint64, Bool.true => varLenCopy%[tl.uint64, tl.uint64, commuted]
  | .bool, Bool.false => varLenCopy%[tl.int1, tl.int1, original]
  | .bool, Bool.true => varLenCopy%[tl.int1, tl.int1, commuted]

end VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteSource

#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>

<details><summary><code>typedKernel</code></summary>

```
/-- Metadata buffers can independently use either signed width. Width abstraction
is explicit; the source has no cast of its loaded signed starts or length. -/
```
```lean
def typedKernel (input output : Storage) (_oldStart _length _newStart : Metadata)
    (v : Bool) : Source := kernel input output v

end VeriTile.Bench.TritonBenchG.VarLenCopy.TypedSource

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteSource
open VeriTile Triton

/-- Width labels distinguish source spellings; Int/Nat retain mathematical carriers. -/
inductive Storage where
  | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | bool
  deriving DecidableEq, Repr

inductive Kind where
  | signed | unsigned | boolean
  deriving DecidableEq, Repr
```
</details>

## Public theorem: `discrete_source_equivalence`

<details><summary>docstring</summary>

```
/-- Complete typed state and failure equality for all nine discrete profiles. -/
```
</details>

**Statement:**
```lean
specification discrete_source_equivalence (storage : DiscreteSource.Storage)
    (os ol ns : Region .int) (src dst : RegionName) (B : Nat) :
    discrete_program storage Bool.false os ol ns src dst B ≡[assumptions]
      discrete_program storage Bool.true os ol ns src dst B
```

**Closed-form spec defs (transitive):** `discrete_program`, `assumptions`, `kernel`

<details><summary><code>discrete_program</code></summary>

```
/-- Homogeneous discrete profiles perform no payload conversion. -/
```
```lean
def discrete_program (storage : DiscreteSource.Storage) (v : Bool)
    (os ol ns : Region .int) (src dst : RegionName) (B : Nat) : Program where
  kernel := DiscreteSource.kernel storage v os ol ns src dst B
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```
/-- No floating algebraic law is used by signed address commutation. -/
```
```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>kernel</code></summary>

```
/-- Homogeneous raw copies have no payload conversion or arithmetic. -/
```
```lean
def kernel (storage : Storage) (v : Bool) : TypedSource.Source :=
  match storage, v with
  | .int8, Bool.false => varLenCopy%[tl.int8, tl.int8, original]
  | .int8, Bool.true => varLenCopy%[tl.int8, tl.int8, commuted]
  | .int16, Bool.false => varLenCopy%[tl.int16, tl.int16, original]
  | .int16, Bool.true => varLenCopy%[tl.int16, tl.int16, commuted]
  | .int32, Bool.false => varLenCopy%[tl.int32, tl.int32, original]
  | .int32, Bool.true => varLenCopy%[tl.int32, tl.int32, commuted]
  | .int64, Bool.false => varLenCopy%[tl.int64, tl.int64, original]
  | .int64, Bool.true => varLenCopy%[tl.int64, tl.int64, commuted]
  | .uint8, Bool.false => varLenCopy%[tl.uint8, tl.uint8, original]
  | .uint8, Bool.true => varLenCopy%[tl.uint8, tl.uint8, commuted]
  | .uint16, Bool.false => varLenCopy%[tl.uint16, tl.uint16, original]
  | .uint16, Bool.true => varLenCopy%[tl.uint16, tl.uint16, commuted]
  | .uint32, Bool.false => varLenCopy%[tl.uint32, tl.uint32, original]
  | .uint32, Bool.true => varLenCopy%[tl.uint32, tl.uint32, commuted]
  | .uint64, Bool.false => varLenCopy%[tl.uint64, tl.uint64, original]
  | .uint64, Bool.true => varLenCopy%[tl.uint64, tl.uint64, commuted]
  | .bool, Bool.false => varLenCopy%[tl.int1, tl.int1, original]
  | .bool, Bool.true => varLenCopy%[tl.int1, tl.int1, commuted]

end VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteSource

#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>
