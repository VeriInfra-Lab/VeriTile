# Spec sheet — `bench/tritonbench_g/var_len_copy/Correct.lean`

**Python source:** `bench/tritonbench_g/var_len_copy/var_len_copy.py`

## Public theorem: `typed_real_correct`

<details><summary>docstring</summary>

```
/-- Full signed-loop IO and frame for either source version. The oracle uses
initial source cells and the actual overcopy count, with no output premises. -/
```
</details>

**Statement:**
```lean
specification typed_real_correct (input output : Storage)
    (oldWidth lengthWidth newWidth : Metadata) (v : Bool) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int))
    (hLb : s.pids 0 < A.extent ol.cast) (hOb : s.pids 0 < A.extent os.cast)
    (hNb : s.pids 0 < A.extent ns.cast)
    (hsrc : oldStart + Math.copiedLength length B ≤ A.extent src)
    (hdst : newStart + Math.copiedLength length B ≤ A.extent dst)
    (hdr : dst ∈ A.regions) (hu : s.undef = fun _ _ ↦ 0) :
    Spec.Real (floatingFlatCopyContract input output oldWidth lengthWidth newWidth v
      A s os ol ns src dst B length oldStart newStart)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hcov : ∀ r, r ∉ A.regions → A.extent r = 0`
- `hB : 0 < B`
- `hsd : src ≠ dst`
- `hL : s.readMemValue .int ol.cast (s.pids 0) = length`
- `hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int)`
- `hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int)`
- `hLb : s.pids 0 < A.extent ol.cast`
- `hOb : s.pids 0 < A.extent os.cast`
- `hNb : s.pids 0 < A.extent ns.cast`
- `hsrc : oldStart + Math.copiedLength length B ≤ A.extent src`
- `hdst : newStart + Math.copiedLength length B ≤ A.extent dst`
- `hdr : dst ∈ A.regions`
- `hu : s.undef = fun _ _ ↦ 0`

**Closed-form spec defs (transitive):** `floatingFlatCopyContract`, `typedKernel`, `kernel`

<details><summary><code>floatingFlatCopyContract</code></summary>

```
/-- Complete floating flat-copy conclusion: successful source execution,
actual-span frame, and initial-source readback. -/
```
```lean
def floatingFlatCopyContract (input output : Storage)
    (oldWidth lengthWidth newWidth : Metadata) (v : Bool) (A : FlatAlloc) (s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName) (B : Nat) (length : Int)
    (oldStart newStart : Nat) : Prop :=
    ∃ final, exec (A.flattenKernel
        (typedKernel input output oldWidth lengthWidth newWidth v os ol ns src dst B).eraseDType.toAlgKernel) (A.flattenState s) = some final ∧
      BlockState.WriteWithin (flatFootprint A (footprint dst newStart length B))
        (A.flattenState s) final ∧
      (∀ n, n < Math.copiedLength length B →
        final.readMem A.flat (A.addr dst (newStart + n)) = s.readMem src (oldStart + n))
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

## Public theorem: `original_real_correct`

**Statement:**
```lean
specification original_real_correct (input output : Storage)
    (oldWidth lengthWidth newWidth : Metadata) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int))
    (hLb : s.pids 0 < A.extent ol.cast) (hOb : s.pids 0 < A.extent os.cast)
    (hNb : s.pids 0 < A.extent ns.cast)
    (hsrc : oldStart + Math.copiedLength length B ≤ A.extent src)
    (hdst : newStart + Math.copiedLength length B ≤ A.extent dst)
    (hdr : dst ∈ A.regions) (hu : s.undef = fun _ _ ↦ 0) :
    Spec.Real (floatingFlatCopyContract input output oldWidth lengthWidth newWidth Bool.false
      A s os ol ns src dst B length oldStart newStart)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hcov : ∀ r, r ∉ A.regions → A.extent r = 0`
- `hB : 0 < B`
- `hsd : src ≠ dst`
- `hL : s.readMemValue .int ol.cast (s.pids 0) = length`
- `hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int)`
- `hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int)`
- `hLb : s.pids 0 < A.extent ol.cast`
- `hOb : s.pids 0 < A.extent os.cast`
- `hNb : s.pids 0 < A.extent ns.cast`
- `hsrc : oldStart + Math.copiedLength length B ≤ A.extent src`
- `hdst : newStart + Math.copiedLength length B ≤ A.extent dst`
- `hdr : dst ∈ A.regions`
- `hu : s.undef = fun _ _ ↦ 0`

**Closed-form spec defs (transitive):** `floatingFlatCopyContract`, `typedKernel`, `kernel`

<details><summary><code>floatingFlatCopyContract</code></summary>

```
/-- Complete floating flat-copy conclusion: successful source execution,
actual-span frame, and initial-source readback. -/
```
```lean
def floatingFlatCopyContract (input output : Storage)
    (oldWidth lengthWidth newWidth : Metadata) (v : Bool) (A : FlatAlloc) (s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName) (B : Nat) (length : Int)
    (oldStart newStart : Nat) : Prop :=
    ∃ final, exec (A.flattenKernel
        (typedKernel input output oldWidth lengthWidth newWidth v os ol ns src dst B).eraseDType.toAlgKernel) (A.flattenState s) = some final ∧
      BlockState.WriteWithin (flatFootprint A (footprint dst newStart length B))
        (A.flattenState s) final ∧
      (∀ n, n < Math.copiedLength length B →
        final.readMem A.flat (A.addr dst (newStart + n)) = s.readMem src (oldStart + n))
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

## Public theorem: `candidate_real_correct`

**Statement:**
```lean
specification candidate_real_correct (input output : Storage)
    (oldWidth lengthWidth newWidth : Metadata) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int))
    (hLb : s.pids 0 < A.extent ol.cast) (hOb : s.pids 0 < A.extent os.cast)
    (hNb : s.pids 0 < A.extent ns.cast)
    (hsrc : oldStart + Math.copiedLength length B ≤ A.extent src)
    (hdst : newStart + Math.copiedLength length B ≤ A.extent dst)
    (hdr : dst ∈ A.regions) (hu : s.undef = fun _ _ ↦ 0) :
    Spec.Real (floatingFlatCopyContract input output oldWidth lengthWidth newWidth Bool.true
      A s os ol ns src dst B length oldStart newStart)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hcov : ∀ r, r ∉ A.regions → A.extent r = 0`
- `hB : 0 < B`
- `hsd : src ≠ dst`
- `hL : s.readMemValue .int ol.cast (s.pids 0) = length`
- `hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int)`
- `hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int)`
- `hLb : s.pids 0 < A.extent ol.cast`
- `hOb : s.pids 0 < A.extent os.cast`
- `hNb : s.pids 0 < A.extent ns.cast`
- `hsrc : oldStart + Math.copiedLength length B ≤ A.extent src`
- `hdst : newStart + Math.copiedLength length B ≤ A.extent dst`
- `hdr : dst ∈ A.regions`
- `hu : s.undef = fun _ _ ↦ 0`

**Closed-form spec defs (transitive):** `floatingFlatCopyContract`, `typedKernel`, `kernel`

<details><summary><code>floatingFlatCopyContract</code></summary>

```
/-- Complete floating flat-copy conclusion: successful source execution,
actual-span frame, and initial-source readback. -/
```
```lean
def floatingFlatCopyContract (input output : Storage)
    (oldWidth lengthWidth newWidth : Metadata) (v : Bool) (A : FlatAlloc) (s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName) (B : Nat) (length : Int)
    (oldStart newStart : Nat) : Prop :=
    ∃ final, exec (A.flattenKernel
        (typedKernel input output oldWidth lengthWidth newWidth v os ol ns src dst B).eraseDType.toAlgKernel) (A.flattenState s) = some final ∧
      BlockState.WriteWithin (flatFootprint A (footprint dst newStart length B))
        (A.flattenState s) final ∧
      (∀ n, n < Math.copiedLength length B →
        final.readMem A.flat (A.addr dst (newStart + n)) = s.readMem src (oldStart + n))
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

## Public theorem: `discrete_typed_real_correct`

<details><summary>docstring</summary>

```
/-- Homogeneous integer and boolean payloads retain their exact typed carrier. -/
```
</details>

**Statement:**
```lean
specification discrete_typed_real_correct (storage : DiscreteSource.Storage)
    (v : Bool) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int))
    (hLb : s.pids 0 < A.extent ol.cast) (hOb : s.pids 0 < A.extent os.cast)
    (hNb : s.pids 0 < A.extent ns.cast)
    (hsrc : oldStart + Math.copiedLength length B ≤ A.extent src)
    (hdst : newStart + Math.copiedLength length B ≤ A.extent dst)
    (hdr : dst ∈ A.regions) (hu : s.undef = fun _ _ ↦ 0) :
    Spec.Real (discreteFlatCopyContract storage v A s os ol ns src dst B
      length oldStart newStart)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hcov : ∀ r, r ∉ A.regions → A.extent r = 0`
- `hB : 0 < B`
- `hsd : src ≠ dst`
- `hL : s.readMemValue .int ol.cast (s.pids 0) = length`
- `hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int)`
- `hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int)`
- `hLb : s.pids 0 < A.extent ol.cast`
- `hOb : s.pids 0 < A.extent os.cast`
- `hNb : s.pids 0 < A.extent ns.cast`
- `hsrc : oldStart + Math.copiedLength length B ≤ A.extent src`
- `hdst : newStart + Math.copiedLength length B ≤ A.extent dst`
- `hdr : dst ∈ A.regions`
- `hu : s.undef = fun _ _ ↦ 0`

**Closed-form spec defs (transitive):** `discreteFlatCopyContract`, `kernel`

<details><summary><code>discreteFlatCopyContract</code></summary>

```
/-- Complete homogeneous discrete flat-copy conclusion, with exact typed
readback and the same full raw-cell frame as the floating contract. -/
```
```lean
def discreteFlatCopyContract (storage : DiscreteSource.Storage) (v : Bool)
    (A : FlatAlloc) (s : BlockState) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat) : Prop :=
    ∃ final, exec (A.flattenKernel
        (DiscreteSource.kernel storage v os ol ns src dst B).eraseDType.toAlgKernel) (A.flattenState s) = some final ∧
      BlockState.WriteWithin (flatFootprint A (footprint dst newStart length B))
        (A.flattenState s) final ∧
      (∀ n, n < Math.copiedLength length B →
        final.readMemValue storage.kind.dtype A.flat (A.addr dst (newStart + n)) = s.readMemValue storage.kind.dtype src (oldStart + n))
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

## Public theorem: `discrete_original_real_correct`

**Statement:**
```lean
specification discrete_original_real_correct (storage : DiscreteSource.Storage)
    (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int))
    (hLb : s.pids 0 < A.extent ol.cast) (hOb : s.pids 0 < A.extent os.cast)
    (hNb : s.pids 0 < A.extent ns.cast)
    (hsrc : oldStart + Math.copiedLength length B ≤ A.extent src)
    (hdst : newStart + Math.copiedLength length B ≤ A.extent dst)
    (hdr : dst ∈ A.regions) (hu : s.undef = fun _ _ ↦ 0) :
    Spec.Real (discreteFlatCopyContract storage Bool.false A s os ol ns src dst B
      length oldStart newStart)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hcov : ∀ r, r ∉ A.regions → A.extent r = 0`
- `hB : 0 < B`
- `hsd : src ≠ dst`
- `hL : s.readMemValue .int ol.cast (s.pids 0) = length`
- `hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int)`
- `hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int)`
- `hLb : s.pids 0 < A.extent ol.cast`
- `hOb : s.pids 0 < A.extent os.cast`
- `hNb : s.pids 0 < A.extent ns.cast`
- `hsrc : oldStart + Math.copiedLength length B ≤ A.extent src`
- `hdst : newStart + Math.copiedLength length B ≤ A.extent dst`
- `hdr : dst ∈ A.regions`
- `hu : s.undef = fun _ _ ↦ 0`

**Closed-form spec defs (transitive):** `discreteFlatCopyContract`, `kernel`

<details><summary><code>discreteFlatCopyContract</code></summary>

```
/-- Complete homogeneous discrete flat-copy conclusion, with exact typed
readback and the same full raw-cell frame as the floating contract. -/
```
```lean
def discreteFlatCopyContract (storage : DiscreteSource.Storage) (v : Bool)
    (A : FlatAlloc) (s : BlockState) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat) : Prop :=
    ∃ final, exec (A.flattenKernel
        (DiscreteSource.kernel storage v os ol ns src dst B).eraseDType.toAlgKernel) (A.flattenState s) = some final ∧
      BlockState.WriteWithin (flatFootprint A (footprint dst newStart length B))
        (A.flattenState s) final ∧
      (∀ n, n < Math.copiedLength length B →
        final.readMemValue storage.kind.dtype A.flat (A.addr dst (newStart + n)) = s.readMemValue storage.kind.dtype src (oldStart + n))
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

## Public theorem: `discrete_candidate_real_correct`

**Statement:**
```lean
specification discrete_candidate_real_correct (storage : DiscreteSource.Storage)
    (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int))
    (hLb : s.pids 0 < A.extent ol.cast) (hOb : s.pids 0 < A.extent os.cast)
    (hNb : s.pids 0 < A.extent ns.cast)
    (hsrc : oldStart + Math.copiedLength length B ≤ A.extent src)
    (hdst : newStart + Math.copiedLength length B ≤ A.extent dst)
    (hdr : dst ∈ A.regions) (hu : s.undef = fun _ _ ↦ 0) :
    Spec.Real (discreteFlatCopyContract storage Bool.true A s os ol ns src dst B
      length oldStart newStart)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hcov : ∀ r, r ∉ A.regions → A.extent r = 0`
- `hB : 0 < B`
- `hsd : src ≠ dst`
- `hL : s.readMemValue .int ol.cast (s.pids 0) = length`
- `hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int)`
- `hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int)`
- `hLb : s.pids 0 < A.extent ol.cast`
- `hOb : s.pids 0 < A.extent os.cast`
- `hNb : s.pids 0 < A.extent ns.cast`
- `hsrc : oldStart + Math.copiedLength length B ≤ A.extent src`
- `hdst : newStart + Math.copiedLength length B ≤ A.extent dst`
- `hdr : dst ∈ A.regions`
- `hu : s.undef = fun _ _ ↦ 0`

**Closed-form spec defs (transitive):** `discreteFlatCopyContract`, `kernel`

<details><summary><code>discreteFlatCopyContract</code></summary>

```
/-- Complete homogeneous discrete flat-copy conclusion, with exact typed
readback and the same full raw-cell frame as the floating contract. -/
```
```lean
def discreteFlatCopyContract (storage : DiscreteSource.Storage) (v : Bool)
    (A : FlatAlloc) (s : BlockState) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat) : Prop :=
    ∃ final, exec (A.flattenKernel
        (DiscreteSource.kernel storage v os ol ns src dst B).eraseDType.toAlgKernel) (A.flattenState s) = some final ∧
      BlockState.WriteWithin (flatFootprint A (footprint dst newStart length B))
        (A.flattenState s) final ∧
      (∀ n, n < Math.copiedLength length B →
        final.readMemValue storage.kind.dtype A.flat (A.addr dst (newStart + n)) = s.readMemValue storage.kind.dtype src (oldStart + n))
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
