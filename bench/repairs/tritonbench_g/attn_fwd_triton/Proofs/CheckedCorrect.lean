import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.CheckedBinding

namespace VeriTile.Bench.Repairs.AttnFwdTriton.Checked
open VeriTile Triton
open Verified Verified.PhysicalCells
open VeriTile.Bench.Repairs.LayerNormFwd
set_option maxHeartbeats 1000000

/-- Direct caller-to-return theorem. The initial memory is the caller memory;
there is no normalization/preparation object or assumed source-access bound. -/
theorem whole_real (p : RealProofs.Parameters) (a : Layout) (A : FlatAlloc)
    (base : Role→Nat) (views : Views a A p base)
    (q k v qs ks : TensorMetadata) (hinput : inputChecks (realGeometry p) q k v qs ks=Bool.true)
    (hl : launchChecks (realGeometry p) qs ks=Bool.true)
    (hp : pointerChecks (realGeometry p) base=Bool.true)
    (bounds : RegionBounds) (storage : Storage p A bounds qs ks)
    (caller : AliasMemory.Physical a.toLayout) (template : BlockState)
    (xs : RealProofs.Inputs)
    (binding : LogicalAgreement.Agree p (Host.logicalInputs a A p caller template) xs)
    (stage : p.stage=3) (nonoverlap : OutputLayout.NonOverlapping p)
    (fresh : FreshBytes a (GridFootprint.window A p)) :
    inputChecks (realGeometry p) q k v qs ks=Bool.true ∧
    NativeControl.CudaLaunchable p (HostShape.blockD p.headDim) ∧
    (∀ r n, n<envelope (realGeometry p) → base r%width r=0 ∧ base r+width r*n+width r<2^64) ∧
    ∃ t, ∃ final : AliasMemory.Physical a.toLayout,
      Nonempty (Kernel.GridLaunchedOrdinary (GridLaunch.kernel A p (HostShape.blockD p.headDim) Bool.false)
        (GridCoordinates.grid p) (AliasMemory.pull a.toLayout caller template) t) ∧
      (∀ i : OutputLayout.Coordinate p,
        Host.readReal (final (Host.outputCell a A p base views i))=
          RealProofs.expected p.heads p.context p.headDim i.1.val i.2.1.val i.2.2.1.val i.2.2.2.val xs) ∧
      (∀ idx : GridIndex (GridCoordinates.grid p),
        Stmt.TraceSafeList bounds (GridLaunch.kernel A p (HostShape.blockD p.headDim) Bool.false).body
          ((AliasMemory.pull a.toLayout caller template).withGridIndex idx)) ∧
      (∀ cell : Fin a.capacity,
        (∀ k, GridFootprint.window A p k → a.address k≠cell.val) → final cell=caller cell) := by
  refine ⟨hinput,launchable (realGeometry p) qs ks hl,?_,?_⟩
  · intro r n hn
    have h:=formed_pointer (realGeometry p) base hp r n hn
    exact ⟨h.1,h.2.1⟩
  · exact Host.launch a A p _ base views caller template xs binding bounds
      (checked_windows p A bounds qs ks storage hl) (HostShape.blockD_covers _) stage Bool.false nonoverlap fresh

/-- The source returns its fresh allocation without launching on an empty
logical tensor. The retained unused M allocation has no memory accesses. -/
theorem empty_return (p : RealProofs.Parameters) (caller : Nat→MemCell)
    (h : p.batches*p.heads*p.context*p.headDim=0) :
    IsEmpty (OutputLayout.Coordinate p) ∧ (∀ cell, caller cell=caller cell) :=
  ⟨Host.empty_output p h,fun _=>rfl⟩

#axiomsClean whole_real
#axiomsClean empty_return
end VeriTile.Bench.Repairs.AttnFwdTriton.Checked
