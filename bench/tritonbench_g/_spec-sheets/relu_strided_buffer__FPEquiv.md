# Spec sheet — `bench/tritonbench_g/relu_strided_buffer/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/relu_strided_buffer/relu_strided_buffer.py`

## Public theorem: `one_tile_equiv`

**Statement:**
```lean
specification one_tile_equiv (_hB : 0 < B) :
    oneTile x y sx sy n tasks tiles B ≡[assumptions]
      oneTileFolded x y sx sy n tasks tiles B
```

**Assumptions / layout contracts:**
- `_hB : 0 < B`

**Closed-form spec defs (transitive):** `oneTile`, `assumptions`, `oneTileFolded`, `relu_forward_kernel_rank_1_one_tile_surface`

<details><summary><code>oneTile</code></summary>

```lean
def oneTile : Program where
  kernel := relu_forward_kernel_rank_1_one_tile_surface x y sx sy n tasks tiles B
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>oneTileFolded</code></summary>

```lean
def oneTileFolded : Program where
  kernel := Candidate.relu_forward_kernel_rank_1_one_tile_surface x y sx sy n tasks tiles B
  domain := []
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

## Public theorem: `grid_stride_equiv`

**Statement:**
```lean
specification grid_stride_equiv (_hB : 0 < B) :
    gridStride x y sx sy n tasks tiles B ≡[assumptions]
      gridStrideFolded x y sx sy n tasks tiles B
```

**Assumptions / layout contracts:**
- `_hB : 0 < B`

**Closed-form spec defs (transitive):** `gridStride`, `assumptions`, `gridStrideFolded`, `relu_forward_kernel_rank_1_grid_stride_surface`

<details><summary><code>gridStride</code></summary>

```lean
def gridStride : Program where
  kernel := relu_forward_kernel_rank_1_grid_stride_surface x y sx sy n tasks tiles B
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>gridStrideFolded</code></summary>

```lean
def gridStrideFolded : Program where
  kernel := Candidate.relu_forward_kernel_rank_1_grid_stride_surface x y sx sy n tasks tiles B
  domain := []
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
