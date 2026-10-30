# Spec sheet — `bench/tritonbench_g/relu_strided_buffer/Correct.lean`

**Python source:** `bench/tritonbench_g/relu_strided_buffer/relu_strided_buffer.py`

## Public theorem: `original_one_tile_correctness`

**Statement:**
```lean
specification original_one_tile_correctness
    (x y : RegionName) (sx sy n tasks tiles B : Nat) (hs : 0 < sy)
    (_hB : 0 < B) :
    Spec.Real (reluOneTileIO x y sx sy n tasks tiles B
      ⊨ fun _pid xs i => TiledActivation.relu (xs i))
```

**Assumptions / layout contracts:**
- `hs : 0 < sy`
- `_hB : 0 < B`

**Closed-form spec defs (transitive):** `reluOneTileIO`, `relu_forward_kernel_rank_1_one_tile_surface`

<details><summary><code>reluOneTileIO</code></summary>

```
/-- IO signature of the `one_tile_per_cta = true` branch on the **tile-indexed**
surface: lane `i` of program `pid` covers task `t = pid·tile_size0 + i`, reads
`in0_ptr` at `t·in0_stride0`, writes `out0_ptr` at `t·out0_stride0`, and is
active exactly on the `boundary_check=(0,)` guard `t < s0`. -/
```
```lean
def reluOneTileIO (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    MaskedTileKernelIO₁ where
  kernel := relu_forward_kernel_rank_1_one_tile_surface in0_ptr out0_ptr
    in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0
  inp := in0_ptr
  out := out0_ptr
  shape := [tile_size0]
  read := fun pid i => taskIndex pid tile_size0 i.1 * in0_stride0
  write := fun pid i => taskIndex pid tile_size0 i.1 * out0_stride0
  mask := fun pid i => taskIndex pid tile_size0 i.1 < s0
```
</details>

<details><summary><code>relu_forward_kernel_rank_1_one_tile_surface</code></summary>

```
/-- Faithful transcription of `relu_strided_buffer.py`'s
`relu_forward_kernel_rank_1`, specialized to the `one_tile_per_cta = true`
(monolithic) branch: one `tile_size0`-wide tile per program, block-pointer
load/store with `boundary_check` on axis 0, `relu_forward` inlined as
`tl.where(in0 > 0, in0, 0)`. -/
```
```lean
def relu_forward_kernel_rank_1_one_tile_surface
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  num_tiles0 = $((s0 + tile_size0 - 1) / tile_size0)
  tile_id = pid
  tile_id0 = tile_id
  offset0 = tile_id0 * $(tile_size0)
  in0_bptr = tl.make_block_ptr(base=in0_ptr, shape=($(s0)), strides=($(in0_stride0)),
    offsets=(offset0), block_shape=($(tile_size0)), order=(0))
  in0 = (tl.load(in0_bptr, boundary_check=([0] : List Nat))).to(in0_ptr.type.element_ty)
  out0 = tl.where(in0 > 0, in0, 0)
  out0_bptr = tl.make_block_ptr(base=out0_ptr, shape=($(s0)), strides=($(out0_stride0)),
    offsets=(offset0), block_shape=($(tile_size0)), order=(0))
  tl.store(out0_bptr, (out0).to(out0_bptr.type.element_ty), boundary_check=([0] : List Nat))
}
```
</details>

## Public theorem: `original_grid_stride_correctness`

**Statement:**
```lean
specification original_grid_stride_correctness (R : RoundingModel)
    (x y : RegionName) (sx sy n tasks tiles B : Nat) (hs : 0 < sy) (hxy : x ≠ y)
    (_hB : 0 < B) :
    Spec.Real (reluGridStrideIO x y sx sy n tasks tiles B
      ⊨[R] fun _p₀ _p₁ _grid xs t i => TiledActivation.relu (xs t i))
```

**Assumptions / layout contracts:**
- `hs : 0 < sy`
- `hxy : x ≠ y`
- `_hB : 0 < B`

**Closed-form spec defs (transitive):** `reluGridStrideIO`, `relu_forward_kernel_rank_1_grid_stride_surface`

<details><summary><code>reluGridStrideIO</code></summary>

```
/-- IO signature of the `one_tile_per_cta = false` **grid-stride** branch on the
grid-width-aware streaming surface: step `j` of program `(pid₀, _)` in a launch
grid of width `nCtas` covers the flat tile `pid₀ + j·nCtas`, whose lane `i` reads
`in0_ptr` at `t·in0_stride0` and writes `out0_ptr` at `t·out0_stride0` for the
task index `t`, active exactly on the `boundary_check=(0,)` guard `t < s0`.

`pre` is the grid-stride idiom's own launch legality `pid₀ < nCtas` — a program's
id is below its grid's width. `BlockState` carries no invariant tying `pids` to
`numPids`, and the kernel's loop genuinely needs `0 < nCtas` (at `nCtas = 0`
every step would revisit tile `pid₀`), so it is assumed here rather than
pretended free. `outDType` stays at the default `.real`: both source casts are
`.to(...element_ty)`, which the DSL erases. -/
```
```lean
def reluGridStrideIO (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    StreamGridStrideEmitMasked2DKernelIO₁ where
  kernel := relu_forward_kernel_rank_1_grid_stride_surface in0_ptr out0_ptr
    in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0
  inp1 := in0_ptr
  out := out0_ptr
  T := tiles_per_cta
  B1 := tile_size0
  C := tile_size0
  pre := fun pid₀ _ nCtas => pid₀ < nCtas
  read1 := fun p₀ _ nCtas t j =>
    taskIndex (p₀ + t.val * nCtas) tile_size0 j * in0_stride0
  write := fun p₀ _ nCtas t j =>
    taskIndex (p₀ + t.val * nCtas) tile_size0 j * out0_stride0
  mask1 := fun p₀ _ nCtas t j => taskIndex (p₀ + t.val * nCtas) tile_size0 j < s0
  writeMask := fun p₀ _ nCtas t j =>
    taskIndex (p₀ + t.val * nCtas) tile_size0 j < s0
```
</details>

<details><summary><code>relu_forward_kernel_rank_1_grid_stride_surface</code></summary>

```
/-- Faithful transcription of `relu_forward_kernel_rank_1`, specialized to the
`one_tile_per_cta = false` (grid-stride-loop) branch: program `pid` covers
tiles `pid + j·num_ctas` for `j < tiles_per_cta`, with
`num_ctas = tl.num_programs(0)`. -/
```
```lean
def relu_forward_kernel_rank_1_grid_stride_surface
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  num_tiles0 = $((s0 + tile_size0 - 1) / tile_size0)
  num_ctas = tl.num_programs(0)
  for j in range($(0), $(tiles_per_cta)) {
    tile_id = pid + j * num_ctas
    tile_id0 = tile_id
    offset0 = tile_id0 * $(tile_size0)
    in0_bptr = tl.make_block_ptr(base=in0_ptr, shape=($(s0)), strides=($(in0_stride0)),
      offsets=(offset0), block_shape=($(tile_size0)), order=(0))
    in0 = (tl.load(in0_bptr, boundary_check=([0] : List Nat))).to(in0_ptr.type.element_ty)
    out0 = tl.where(in0 > 0, in0, 0)
    out0_bptr = tl.make_block_ptr(base=out0_ptr, shape=($(s0)), strides=($(out0_stride0)),
      offsets=(offset0), block_shape=($(tile_size0)), order=(0))
    tl.store(out0_bptr, (out0).to(out0_bptr.type.element_ty), boundary_check=([0] : List Nat))
  }
}
```
</details>

## Public theorem: `candidate_one_tile_correctness`

**Statement:**
```lean
specification candidate_one_tile_correctness
    (x y : RegionName) (sx sy n tasks tiles B : Nat) (hs : 0 < sy)
    (_hB : 0 < B) :
    Spec.Real (Candidate.reluOneTileIO x y sx sy n tasks tiles B
      ⊨ fun _pid xs i => TiledActivation.relu (xs i))
```

**Assumptions / layout contracts:**
- `hs : 0 < sy`
- `_hB : 0 < B`

**Closed-form spec defs (transitive):** `reluOneTileIO`, `relu_forward_kernel_rank_1_one_tile_surface`

<details><summary><code>reluOneTileIO</code></summary>

```
/-- IO signature of the `one_tile_per_cta = true` branch on the **tile-indexed**
surface: lane `i` of program `pid` covers task `t = pid·tile_size0 + i`, reads
`in0_ptr` at `t·in0_stride0`, writes `out0_ptr` at `t·out0_stride0`, and is
active exactly on the `boundary_check=(0,)` guard `t < s0`. -/
```
```lean
def reluOneTileIO (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    MaskedTileKernelIO₁ where
  kernel := relu_forward_kernel_rank_1_one_tile_surface in0_ptr out0_ptr
    in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0
  inp := in0_ptr
  out := out0_ptr
  shape := [tile_size0]
  read := fun pid i => taskIndex pid tile_size0 i.1 * in0_stride0
  write := fun pid i => taskIndex pid tile_size0 i.1 * out0_stride0
  mask := fun pid i => taskIndex pid tile_size0 i.1 < s0
```
</details>

<details><summary><code>relu_forward_kernel_rank_1_one_tile_surface</code></summary>

```
/-- Faithful transcription of `relu_strided_buffer.py`'s
`relu_forward_kernel_rank_1`, specialized to the `one_tile_per_cta = true`
(monolithic) branch: one `tile_size0`-wide tile per program, block-pointer
load/store with `boundary_check` on axis 0, `relu_forward` inlined as
`tl.where(in0 > 0, in0, 0)`. -/
```
```lean
def relu_forward_kernel_rank_1_one_tile_surface
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  num_tiles0 = $((s0 + tile_size0 - 1) / tile_size0)
  tile_id = pid
  tile_id0 = tile_id
  offset0 = tile_id0 * $(tile_size0)
  in0_bptr = tl.make_block_ptr(base=in0_ptr, shape=($(s0)), strides=($(in0_stride0)),
    offsets=(offset0), block_shape=($(tile_size0)), order=(0))
  in0 = (tl.load(in0_bptr, boundary_check=([0] : List Nat))).to(in0_ptr.type.element_ty)
  out0 = tl.where(in0 > 0, in0, 0)
  out0_bptr = tl.make_block_ptr(base=out0_ptr, shape=($(s0)), strides=($(out0_stride0)),
    offsets=(offset0), block_shape=($(tile_size0)), order=(0))
  tl.store(out0_bptr, (out0).to(out0_bptr.type.element_ty), boundary_check=([0] : List Nat))
}
```
</details>

## Public theorem: `candidate_grid_stride_correctness`

**Statement:**
```lean
specification candidate_grid_stride_correctness (R : RoundingModel)
    (x y : RegionName) (sx sy n tasks tiles B : Nat) (hs : 0 < sy) (hxy : x ≠ y)
    (_hB : 0 < B) :
    Spec.Real (Candidate.reluGridStrideIO x y sx sy n tasks tiles B
      ⊨[R] fun _p₀ _p₁ _grid xs t i => TiledActivation.relu (xs t i))
```

**Assumptions / layout contracts:**
- `hs : 0 < sy`
- `hxy : x ≠ y`
- `_hB : 0 < B`

**Closed-form spec defs (transitive):** `reluGridStrideIO`, `relu_forward_kernel_rank_1_grid_stride_surface`

<details><summary><code>reluGridStrideIO</code></summary>

```
/-- IO signature of the `one_tile_per_cta = false` **grid-stride** branch on the
grid-width-aware streaming surface: step `j` of program `(pid₀, _)` in a launch
grid of width `nCtas` covers the flat tile `pid₀ + j·nCtas`, whose lane `i` reads
`in0_ptr` at `t·in0_stride0` and writes `out0_ptr` at `t·out0_stride0` for the
task index `t`, active exactly on the `boundary_check=(0,)` guard `t < s0`.

`pre` is the grid-stride idiom's own launch legality `pid₀ < nCtas` — a program's
id is below its grid's width. `BlockState` carries no invariant tying `pids` to
`numPids`, and the kernel's loop genuinely needs `0 < nCtas` (at `nCtas = 0`
every step would revisit tile `pid₀`), so it is assumed here rather than
pretended free. `outDType` stays at the default `.real`: both source casts are
`.to(...element_ty)`, which the DSL erases. -/
```
```lean
def reluGridStrideIO (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    StreamGridStrideEmitMasked2DKernelIO₁ where
  kernel := relu_forward_kernel_rank_1_grid_stride_surface in0_ptr out0_ptr
    in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0
  inp1 := in0_ptr
  out := out0_ptr
  T := tiles_per_cta
  B1 := tile_size0
  C := tile_size0
  pre := fun pid₀ _ nCtas => pid₀ < nCtas
  read1 := fun p₀ _ nCtas t j =>
    taskIndex (p₀ + t.val * nCtas) tile_size0 j * in0_stride0
  write := fun p₀ _ nCtas t j =>
    taskIndex (p₀ + t.val * nCtas) tile_size0 j * out0_stride0
  mask1 := fun p₀ _ nCtas t j => taskIndex (p₀ + t.val * nCtas) tile_size0 j < s0
  writeMask := fun p₀ _ nCtas t j =>
    taskIndex (p₀ + t.val * nCtas) tile_size0 j < s0
```
</details>

<details><summary><code>relu_forward_kernel_rank_1_grid_stride_surface</code></summary>

```
/-- Faithful transcription of `relu_forward_kernel_rank_1`, specialized to the
`one_tile_per_cta = false` (grid-stride-loop) branch: program `pid` covers
tiles `pid + j·num_ctas` for `j < tiles_per_cta`, with
`num_ctas = tl.num_programs(0)`. -/
```
```lean
def relu_forward_kernel_rank_1_grid_stride_surface
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  num_tiles0 = $((s0 + tile_size0 - 1) / tile_size0)
  num_ctas = tl.num_programs(0)
  for j in range($(0), $(tiles_per_cta)) {
    tile_id = pid + j * num_ctas
    tile_id0 = tile_id
    offset0 = tile_id0 * $(tile_size0)
    in0_bptr = tl.make_block_ptr(base=in0_ptr, shape=($(s0)), strides=($(in0_stride0)),
      offsets=(offset0), block_shape=($(tile_size0)), order=(0))
    in0 = (tl.load(in0_bptr, boundary_check=([0] : List Nat))).to(in0_ptr.type.element_ty)
    out0 = tl.where(in0 > 0, in0, 0)
    out0_bptr = tl.make_block_ptr(base=out0_ptr, shape=($(s0)), strides=($(out0_stride0)),
      offsets=(offset0), block_shape=($(tile_size0)), order=(0))
    tl.store(out0_bptr, (out0).to(out0_bptr.type.element_ty), boundary_check=([0] : List Nat))
  }
}
```
</details>
