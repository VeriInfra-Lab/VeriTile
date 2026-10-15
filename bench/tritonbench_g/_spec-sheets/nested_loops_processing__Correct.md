# Spec sheet — `bench/tritonbench_g/nested_loops_processing/Correct.lean`

**Python source:** `bench/tritonbench_g/nested_loops_processing/nested_loops_processing.py`

## Public theorem: `nested3_output_summary_general`

<details><summary>docstring</summary>

```
/-! ## Dimension-general output summary

The full kernel body is stride-parametric: the lowering and every per-store
copy-correctness fact above hold over symbolic `(stride_m, stride_n)`. The
summary below bundles, over arbitrary strides, (a) that the full nested-loop
surface lowers and (b) the complete genuine copy-correctness sweep — all 24
`2×2` block stores of the three nested `range(0, 2)` loops, each carrying its own
`2×2`-offset injectivity hypothesis and asserting a verbatim copy from the
shifted INPUT region to the shifted OUTPUT region (`expected` reads input
memory, never `exec(...).readMem`). The Python wrapper's contiguous square
layouts (`stride_m ∈ {8, 4, 16, 2}`, `stride_n = 1`) are instances. -/
```
</details>

**Statement:**
```lean
specification nested3_output_summary_general
    (in_ptr out_ptr : RegionName) (stride_m stride_n : Nat)
    (s : BlockState)
    (hInj0 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffset stride_m stride_n idx))
    (hInj2 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 2 idx))
    (hInj4 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 4 idx))
    (hInj6 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 6 idx))
    (hInj8 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 8 idx))
    (hInj10 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 10 idx))
    (hInj12 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 12 idx))
    (hInj14 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 14 idx))
    (hInj16 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 16 idx))
    (hInj18 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 18 idx))
    (hInj20 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 20 idx))
    (hInj22 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 22 idx))
    (hInj24 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 24 idx))
    (hInj26 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 26 idx))
    (hInj28 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 28 idx))
    (hInj30 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 30 idx))
    (hInj32 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 32 idx))
    (hInj34 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 34 idx))
    (hInj36 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 36 idx))
    (hInj38 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 38 idx))
    (hInj40 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 40 idx))
    (hInj42 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 42 idx))
    (hInj44 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 44 idx))
    (hInj46 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 46 idx)) :
    -- (a) The full nested-loop surface lowers to the algorithm layer.
    (∃ alg, (nested3 in_ptr out_ptr stride_m stride_n).toAlgorithm? =
      Except.ok alg) ∧
    -- (b1) Initial a1 store: verbatim copy at shift 0.
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_first_a1_store in_ptr out_ptr stride_m stride_n)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffset stride_m stride_n idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffset stride_m stride_n idx)) ∧
    -- (b2) First a2 store (i,j,k)=(0,0,0).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_first_a2_store in_ptr out_ptr stride_m stride_n)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 2 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 2 idx)) ∧
    -- (b3) First a3 store (i,j,k)=(0,0,0).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_first_a3_store in_ptr out_ptr stride_m stride_n)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 4 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 4 idx)) ∧
    -- (b4) Second k: a1 reused (in 0 → out 6).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 0 6)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 6 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 0 idx)) ∧
    -- (b5) Second k: a2 (in 2 → out 8).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 2 8)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 8 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 2 idx)) ∧
    -- (b6) Second k: a3 (in 6 → out 10).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 6 10)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 10 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 6 idx)) ∧
    -- (b7) Second j, first k: a1 reused (in 0 → out 12).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 0 12)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 12 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 0 idx)) ∧
    -- (b8) Second j, first k: a2 (in 8 → out 14).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 8 14)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 14 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 8 idx)) ∧
    -- (b9) Second j, first k: a3 (in 10 → out 16).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 10 16)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 16 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 10 idx)) ∧
    -- (b10) Second j, second k: a1 reused (in 0 → out 18).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 0 18)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 18 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 0 idx)) ∧
    -- (b11) Second j, second k: a2 reused (in 8 → out 20).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 8 20)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 20 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 8 idx)) ∧
    -- (b12) Second j, second k: a3 (in 12 → out 22).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 12 22)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 22 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 12 idx)) ∧
    -- (b13) Second i, first j, first k: a1 (in 14 → out 24).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 14 24)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 24 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 14 idx)) ∧
    -- (b14) Second i, first j, first k: a2 (in 16 → out 26).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 16 26)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 26 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 16 idx)) ∧
    -- (b15) Second i, first j, first k: a3 (in 18 → out 28).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 18 28)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 28 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 18 idx)) ∧
    -- (b16) Second i, first j, second k: a1 reused (in 14 → out 30).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 14 30)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 30 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 14 idx)) ∧
    -- (b17) Second i, first j, second k: a2 reused (in 16 → out 32).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 16 32)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 32 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 16 idx)) ∧
    -- (b18) Second i, first j, second k: a3 (in 20 → out 34).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 20 34)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 34 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 20 idx)) ∧
    -- (b19) Second i, second j, first k: a1 reused (in 14 → out 36).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 14 36)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 36 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 14 idx)) ∧
    -- (b20) Second i, second j, first k: a2 (in 22 → out 38).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 22 38)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 38 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 22 idx)) ∧
    -- (b21) Second i, second j, first k: a3 (in 24 → out 40).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 24 40)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 40 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 24 idx)) ∧
    -- (b22) Second i, second j, second k: a1 reused (in 14 → out 42).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 14 42)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 42 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 14 idx)) ∧
    -- (b23) Second i, second j, second k: a2 reused (in 22 → out 44).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 22 44)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 44 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 22 idx)) ∧
    -- (b24) Second i, second j, second k: final a3 (in 26 → out 46).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 26 46)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 46 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 26 idx))
```

**Assumptions / layout contracts:**
- `hInj0 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffset stride_m stride_n idx)`
- `hInj2 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 2 idx)`
- `hInj4 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 4 idx)`
- `hInj6 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 6 idx)`
- `hInj8 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 8 idx)`
- `hInj10 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 10 idx)`
- `hInj12 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 12 idx)`
- `hInj14 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 14 idx)`
- `hInj16 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 16 idx)`
- `hInj18 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 18 idx)`
- `hInj20 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 20 idx)`
- `hInj22 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 22 idx)`
- `hInj24 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 24 idx)`
- `hInj26 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 26 idx)`
- `hInj28 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 28 idx)`
- `hInj30 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 30 idx)`
- `hInj32 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 32 idx)`
- `hInj34 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 34 idx)`
- `hInj36 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 36 idx)`
- `hInj38 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 38 idx)`
- `hInj40 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 40 idx)`
- `hInj42 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 42 idx)`
- `hInj44 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 44 idx)`
- `hInj46 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 46 idx)`

**Closed-form spec defs (transitive):** `kernel`

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

## Public theorem: `nested3_first_a1_store_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for `nested_loops_processing.py`'s first
`a1` writeback: for every disjoint flat placement of `in_ptr` / `out_ptr` whose
2×2 strided window is in bounds, and every launch state whose `in_ptr` window
holds `xs`, the kernel terminates, every lane of the `out_ptr` window holds
`xs idx`, and every other memory cell is unchanged.

The store is unmasked, so the mask is `True` and the readback is total on the
window. General in both strides; honest side-condition = window injectivity, the
same hypothesis the per-write-map summary takes (`stride_m = stride_n` aliases
the two diagonal lanes). The tile extents are literal `2 × 2` because the Python
kernel's `tl.arange(0, 2)` are literals. -/
```
</details>

**Statement:**
```lean
specification nested3_first_a1_store_io_correctness
    (in_ptr out_ptr : RegionName) (stride_m stride_n : Nat)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        idx.1.val * stride_m + idx.2.1.val * stride_n)) :
    a1StoreIO in_ptr out_ptr stride_m stride_n ⊨ fun _pid xs idx => xs idx
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        idx.1.val * stride_m + idx.2.1.val * stride_n)`

**Closed-form spec defs (transitive):** `a1StoreIO`, `kernel`

<details><summary><code>a1StoreIO</code></summary>

```
/-- IO signature of the `a1` 2×2 store. -/
```
```lean
def a1StoreIO (in_ptr out_ptr : RegionName) (stride_m stride_n : Nat) :
    MaskedTileKernelIO₁ where
  kernel := nested3_first_a1_store in_ptr out_ptr stride_m stride_n
  inp := in_ptr
  out := out_ptr
  shape := [2, 2]
  read := fun _pid idx => idx.1.val * stride_m + idx.2.1.val * stride_n
  write := fun _pid idx => idx.1.val * stride_m + idx.2.1.val * stride_n
  mask := fun _pid _idx => True
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

## Public theorem: `nested3_first_a1_store_io_correctnessR`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` headline** for `nested_loops_processing.py`'s first `a1`
writeback: for **every** rounding model `R`, the same Hoare triple as
`nested3_first_a1_store_io_correctness`, but run under `execR R` and read back as
`.real`-typed cells holding `R.round .real (xs idx)`.

The store is a pure copy carrying no `.to(...)`, so the slice is cast-free and
the exact run transports verbatim. The content of the rounding face here is
exactly that: *this kernel introduces no rounding event of its own*, at any
`R`. -/
```
</details>

**Statement:**
```lean
specification nested3_first_a1_store_io_correctnessR (R : RoundingModel)
    (in_ptr out_ptr : RegionName) (stride_m stride_n : Nat)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        idx.1.val * stride_m + idx.2.1.val * stride_n)) :
    a1StoreIO in_ptr out_ptr stride_m stride_n
      ⊨[R, FloatDType.real] fun _pid xs idx => xs idx
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        idx.1.val * stride_m + idx.2.1.val * stride_n)`

**Closed-form spec defs (transitive):** `a1StoreIO`, `kernel`

<details><summary><code>a1StoreIO</code></summary>

```
/-- IO signature of the `a1` 2×2 store. -/
```
```lean
def a1StoreIO (in_ptr out_ptr : RegionName) (stride_m stride_n : Nat) :
    MaskedTileKernelIO₁ where
  kernel := nested3_first_a1_store in_ptr out_ptr stride_m stride_n
  inp := in_ptr
  out := out_ptr
  shape := [2, 2]
  read := fun _pid idx => idx.1.val * stride_m + idx.2.1.val * stride_n
  write := fun _pid idx => idx.1.val * stride_m + idx.2.1.val * stride_n
  mask := fun _pid _idx => True
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

## Public theorem: `original_real_correct`

<details><summary>docstring</summary>

```
/-- Complete original execution, independent output oracle, raw frame, and access safety. -/
```
</details>

**Statement:**
```lean
specification original_real_correct (storage : Storage) (bounds : RegionBounds)
    (s : BlockState) (src dst : RegionName) (sm sn : Nat)
    (hsep : src ≠ dst) (hinjective : FullReal.TileInjective sm sn)
    (hsrc : sm + 27 * sn < bounds src) (hdst : sm + 47 * sn < bounds dst) :
    Spec.Real (Contracts.RegionCopy storage Bool.false bounds s src dst sm sn)
```

**Assumptions / layout contracts:**
- `hsep : src ≠ dst`
- `hinjective : FullReal.TileInjective sm sn`
- `hsrc : sm + 27 * sn < bounds src`
- `hdst : sm + 47 * sn < bounds dst`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `candidate_real_correct`

<details><summary>docstring</summary>

```
/-- The actual candidate has the same independently specified ordered output and full frame. -/
```
</details>

**Statement:**
```lean
specification candidate_real_correct (storage : Storage) (bounds : RegionBounds)
    (s : BlockState) (src dst : RegionName) (sm sn : Nat)
    (hsep : src ≠ dst) (hinjective : FullReal.TileInjective sm sn)
    (hsrc : sm + 27 * sn < bounds src) (hdst : sm + 47 * sn < bounds dst) :
    Spec.Real (Contracts.RegionCopy storage Bool.true bounds s src dst sm sn)
```

**Assumptions / layout contracts:**
- `hsep : src ≠ dst`
- `hinjective : FullReal.TileInjective sm sn`
- `hsrc : sm + 27 * sn < bounds src`
- `hdst : sm + 47 * sn < bounds dst`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `original_flat_real_correct`

<details><summary>docstring</summary>

```
/-- Complete original flat-memory contract; the library bridge's zero-undef
requirement is explicit and is absent from the region-memory contract. -/
```
</details>

**Statement:**
```lean
specification original_flat_real_correct (storage : Storage) (A : FlatAlloc)
    (hd : A.Disjoint) (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (src dst : RegionName) (sm sn : Nat)
    (hsep : src ≠ dst) (hinjective : FullReal.TileInjective sm sn)
    (hsrc : sm + 27 * sn < A.extent src) (hdst : sm + 47 * sn < A.extent dst)
    (hu : s.undef = fun _ _ ↦ 0) :
    Spec.Real (Contracts.FlatCopy storage Bool.false A s src dst sm sn)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hcov : ∀ r, r ∉ A.regions → A.extent r = 0`
- `hsep : src ≠ dst`
- `hinjective : FullReal.TileInjective sm sn`
- `hsrc : sm + 27 * sn < A.extent src`
- `hdst : sm + 47 * sn < A.extent dst`
- `hu : s.undef = fun _ _ ↦ 0`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `candidate_flat_real_correct`

<details><summary>docstring</summary>

```
/-- Complete candidate flat-memory contract, including untouched raw cell types. -/
```
</details>

**Statement:**
```lean
specification candidate_flat_real_correct (storage : Storage) (A : FlatAlloc)
    (hd : A.Disjoint) (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (src dst : RegionName) (sm sn : Nat)
    (hsep : src ≠ dst) (hinjective : FullReal.TileInjective sm sn)
    (hsrc : sm + 27 * sn < A.extent src) (hdst : sm + 47 * sn < A.extent dst)
    (hu : s.undef = fun _ _ ↦ 0) :
    Spec.Real (Contracts.FlatCopy storage Bool.true A s src dst sm sn)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hcov : ∀ r, r ∉ A.regions → A.extent r = 0`
- `hsep : src ≠ dst`
- `hinjective : FullReal.TileInjective sm sn`
- `hsrc : sm + 27 * sn < A.extent src`
- `hdst : sm + 47 * sn < A.extent dst`
- `hu : s.undef = fun _ _ ↦ 0`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
