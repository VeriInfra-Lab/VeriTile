import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Proofs.RealInputs
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.HostBlock

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.RawReal
open VeriTile Triton AttnFwdTriton
open Verified Verified.PhysicalCells
open VeriTile.Bench.Repairs.LayerNormFwd
set_option maxHeartbeats 1000000

/-- Complete causal kernel result over raw caller memory, source geometry and
six memory windows. Views contain only active address/width equations. The
independent inputs use each source role's own base and strides. -/
theorem whole_real (p : RealProofs.Parameters) (a : Layout) (A : FlatAlloc)
    (base : Role→Nat) (views : Views a A p base)
    (bounds : RegionBounds)
    (windows : ViewBounds.Windows bounds (Relocation.parameters p A.flat) (Relocation.offsets A p))
    (memory : Nat→ℝ) (template : BlockState)
    (stage : p.stage=3) (nonoverlap : OutputLayout.NonOverlapping p)
    (fresh : FreshBytes a (GridFootprint.window A p)) :
    ∃ t, ∃ final : AliasMemory.Physical a.toLayout,
      Nonempty (Kernel.GridLaunchedOrdinary (GridLaunch.kernel A p (HostShape.blockD p.headDim) Bool.false)
        (GridCoordinates.grid p) (AliasMemory.pull a.toLayout (RealBinding.cells a memory) template) t) ∧
      (∀ i : OutputLayout.Coordinate p,
        Host.readReal (final (Host.outputCell a A p base views i))=
          RealProofs.expected p.heads p.context p.headDim i.1.val i.2.1.val i.2.2.1.val i.2.2.2.val (RealBinding.inputs p base memory)) ∧
      (∀ idx : GridIndex (GridCoordinates.grid p),
        Stmt.TraceSafeList bounds (GridLaunch.kernel A p (HostShape.blockD p.headDim) Bool.false).body
          ((AliasMemory.pull a.toLayout (RealBinding.cells a memory) template).withGridIndex idx)) ∧
      (∀ cell : Fin a.capacity,
        (∀ k, GridFootprint.window A p k → a.address k≠cell.val) → final cell=(RealBinding.cells a memory) cell) :=
  Host.launch a A p _ base views (RealBinding.cells a memory) template (RealBinding.inputs p base memory)
    (RealBinding.agreement p a A base views memory template) bounds windows
    (HostShape.blockD_covers _) stage Bool.false nonoverlap fresh

#axiomsClean whole_real
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.RawReal
