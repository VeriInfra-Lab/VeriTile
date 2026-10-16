import bench.tritonbench_g.pow_scalar_tensor.Proofs.IntegerContracts
import bench.tritonbench_g.pow_scalar_tensor.Proofs.IntegerWitness
import bench.tritonbench_g.pow_scalar_tensor.Proofs.IntegerFailure
import bench.tritonbench_g.pow_scalar_tensor.Proofs.LegacyReal
import bench.tritonbench_g.pow_scalar_tensor.Proofs.Contracts
import bench.tritonbench_g.pow_scalar_tensor.Proofs.Witness
import bench.tritonbench_g.pow_scalar_tensor.Proofs.ExecutionValues
import bench.tritonbench_g.pow_scalar_tensor.Proofs.Precision
import VeriTile.Meta.StatementAudit

/-! Historical mathematical and slice contracts in this file are ordinary helper
theorems. Only the retained `specification` entries below are public source contracts.
The helpers retain their statements and proofs for downstream reuse. -/


namespace VeriTile.Bench.TritonBenchG.PowScalarTensor
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedTileKernelIO₁


/-- **The headline on the IO surface** for `pow_scalar_tensor.py`'s
`pow_func_scalar_tensor_kernel_rank_1`, `one_tile_per_cta = true` branch: for
every disjoint flat placement of the two buffers, every program id whose active
lanes are in bounds, and every launch state whose input window holds `xs`, the
translated pointer kernel terminates, every active lane of the output tile holds
the genuine scalar-base power `val0 ^ xs i` (`Real.rpow`), and every other
memory cell is unchanged.

Dimension-general in `s0`, both strides, `tile_size0`, and the scalar base
`val0`; the only side-condition is `0 < out0_stride0` (store-footprint
injectivity — a torch stride of a non-degenerate rank-1 buffer is ≥ 1). The
strided, two-stride footprint is carried verbatim by `MaskedTileKernelIO₁`'s
address functions. -/
theorem pow_scalar_tensor_one_tile_io_correctness (val0 : ℝ)
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (hStride : 0 < out0_stride0) :
    powOneTileIO val0 in0_ptr out0_ptr in0_stride0 out0_stride0 s0 num_tasks
        tiles_per_cta tile_size0
      ⊨ fun _pid xs i => Real.rpow val0 (xs i) := by
  refine MaskedTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact pow_one_tile_flattenOk val0 in0_ptr out0_ptr in0_stride0 out0_stride0
      s0 num_tasks tiles_per_cta tile_size0
  · intro bounds s h1 h2
    exact pow_one_tile_traceSafe val0 in0_ptr out0_ptr in0_stride0 out0_stride0
      s0 num_tasks tiles_per_cta tile_size0 bounds s
      (fun i hi => h1 (i, PUnit.unit) hi) (fun i hi => h2 (i, PUnit.unit) hi)
  · intro s₀ xs hin
    exact pow_one_tile_region_run val0 in0_ptr out0_ptr in0_stride0 out0_stride0
      s0 num_tasks tiles_per_cta tile_size0 hStride s₀ xs hin

/-- Full actual one-tile source correctness with a dimension-general finite
allocation, partial tiles and independent natural input/output strides. -/
specification floating_one_tile_correct (output : TypedSource.FloatingOutput) (candidate : Bool)
    (alloc : FlatAlloc) (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat)
    (s : BlockState) (xs : Fin B → ℝ)
    (hd : alloc.Disjoint) (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0)
    (hy : Y ∈ alloc.regions) (hstride : 0 < sy) (hu : s.undef = fun _ _ => 0)
    (hinput : ∀ i, taskIndex (s.pids 0) B i < N →
      s.readMem X (taskIndex (s.pids 0) B i * sx) = xs i)
    (hin : ∀ i, taskIndex (s.pids 0) B i < N → taskIndex (s.pids 0) B i * sx < alloc.extent X)
    (hout : ∀ i, taskIndex (s.pids 0) B i < N → taskIndex (s.pids 0) B i * sy < alloc.extent Y) :
    Spec.Real (Contracts.ExactIO (TypedSource.kernel output Bool.true candidate base X Y sx sy N tasks tiles B)
      alloc s Y base (fun i => taskIndex (s.pids 0) B i * sy)
      (fun i => taskIndex (s.pids 0) B i < N) xs) :=
  Contracts.one_tile output candidate alloc base X Y sx sy N tasks tiles B s xs hd hcov hy hstride hu hinput hin hout


/-- Full actual grid-stride source correctness, including all dynamic
iterations and a partial final tile, against independent exponent arrays. -/
specification floating_grid_correct (output : TypedSource.FloatingOutput) (candidate : Bool)
    (alloc : FlatAlloc) (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat)
    (s : BlockState) (xs : Fin tiles → Fin B → ℝ)
    (hd : alloc.Disjoint) (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0)
    (hy : Y ∈ alloc.regions) (hstride : 0 < sy) (hu : s.undef = fun _ _ => 0)
    (hdisj : X ≠ Y) (hgrid : 0 < s.numPids 0)
    (hinput : ∀ j i, taskIndex (s.pids 0 + j.val * s.numPids 0) B i < N →
      s.readMem X (taskIndex (s.pids 0 + j.val * s.numPids 0) B i * sx) = xs j i)
    (hin : ∀ (j : Fin tiles) i, taskIndex (s.pids 0 + j.val * s.numPids 0) B i < N →
      taskIndex (s.pids 0 + j.val * s.numPids 0) B i * sx < alloc.extent X)
    (hout : ∀ (j : Fin tiles) i, taskIndex (s.pids 0 + j.val * s.numPids 0) B i < N →
      taskIndex (s.pids 0 + j.val * s.numPids 0) B i * sy < alloc.extent Y) :
    Spec.Real (Contracts.ExactIO (TypedSource.kernel output Bool.false candidate base X Y sx sy N tasks tiles B)
      alloc s Y base
      (fun p : Fin tiles × Fin B => taskIndex (s.pids 0 + p.1.val * s.numPids 0) B p.2 * sy)
      (fun p => taskIndex (s.pids 0 + p.1.val * s.numPids 0) B p.2 < N)
      (fun p => xs p.1 p.2)) :=
  Contracts.grid output candidate alloc base X Y sx sy N tasks tiles B s xs hd hcov hy hstride hu hdisj hgrid hinput hin hout

/-- Complete int32 single-tile source with arbitrary dimensions, actual trace safety,
finite-flat execution, independent signed-integer zpow and a full raw-memory frame. -/
specification integer_one_tile_correct (output : TypedSource.FloatingOutput) (candidate : Bool)
    (alloc : FlatAlloc) (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat)
    (s : BlockState) (xs : Fin B → Int)
    (hd : alloc.Disjoint) (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0)
    (hy : Y ∈ alloc.regions) (hstride : 0 < sy) (hu : s.undef = fun _ _ => 0)
    (hinput : ∀ i, taskIndex (s.pids 0) B i < N →
      s.readMemTyped .int X (taskIndex (s.pids 0) B i * sx) = some (xs i))
    (hvalid : ∀ i, taskIndex (s.pids 0) B i < N → Int32Bits.inRange (xs i) = Bool.true)
    (hin : ∀ i, taskIndex (s.pids 0) B i < N → taskIndex (s.pids 0) B i * sx < alloc.extent X)
    (hout : ∀ i, taskIndex (s.pids 0) B i < N → taskIndex (s.pids 0) B i * sy < alloc.extent Y) :
    Spec.Real (IntegerContracts.ExactIO (TypedIntegerSource.kernel output Bool.true candidate base X Y sx sy N tasks tiles B)
      alloc s Y base (fun i => taskIndex (s.pids 0) B i * sy)
      (fun i => taskIndex (s.pids 0) B i < N) xs) :=
  IntegerContracts.one_tile output candidate alloc base X Y sx sy N tasks tiles B s xs hd hcov hy hstride hu hinput hvalid hin hout

/-- Complete int32 grid-stride source with arbitrary dimensions, actual trace safety,
finite-flat execution, independent signed-integer zpow and a full raw-memory frame. -/
specification integer_grid_correct (output : TypedSource.FloatingOutput) (candidate : Bool)
    (alloc : FlatAlloc) (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat)
    (s : BlockState) (xs : Fin tiles → Fin B → Int)
    (hd : alloc.Disjoint) (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0)
    (hy : Y ∈ alloc.regions) (hstride : 0 < sy) (hu : s.undef = fun _ _ => 0)
    (hdisj : X ≠ Y) (hgrid : 0 < s.numPids 0)
    (hinput : ∀ j i, taskIndex (s.pids 0+j.val*s.numPids 0) B i < N →
      s.readMemTyped .int X (taskIndex (s.pids 0+j.val*s.numPids 0) B i * sx) = some (xs j i))
    (hvalid : ∀ j i, taskIndex (s.pids 0+j.val*s.numPids 0) B i < N →
      Int32Bits.inRange (xs j i) = Bool.true)
    (hin : ∀ (j : Fin tiles) i, taskIndex (s.pids 0+j.val*s.numPids 0) B i < N →
      taskIndex (s.pids 0+j.val*s.numPids 0) B i * sx < alloc.extent X)
    (hout : ∀ (j : Fin tiles) i, taskIndex (s.pids 0+j.val*s.numPids 0) B i < N →
      taskIndex (s.pids 0+j.val*s.numPids 0) B i * sy < alloc.extent Y) :
    Spec.Real (IntegerContracts.ExactIO (TypedIntegerSource.kernel output Bool.false candidate base X Y sx sy N tasks tiles B)
      alloc s Y base
      (fun p : Fin tiles × Fin B => taskIndex (s.pids 0+p.1.val*s.numPids 0) B p.2 * sy)
      (fun p => taskIndex (s.pids 0+p.1.val*s.numPids 0) B p.2 < N)
      (fun p => xs p.1 p.2)) :=
  IntegerContracts.grid output candidate alloc base X Y sx sy N tasks tiles B s xs hd hcov hy hstride hu hdisj hgrid hinput hvalid hin hout

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor
