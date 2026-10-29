import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.NativeControl
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.LogicalAgreement

/-! Whole-grid Real contract over one physical input state. Ordinary tensor
metadata connects packed/resolve_neg inputs and the fresh output allocation. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Host
open VeriTile.Triton Memory Relocation PhysicalCells
open VeriTile.Bench.Repairs.LayerNormFwd
noncomputable section
set_option maxHeartbeats 1000000

def logicalInputs (a : Layout) (A : FlatAlloc) (p : Parameters)
    (initial : AliasMemory.Physical a.toLayout) (template : BlockState) : RealProofs.Inputs :=
  LogicalInputs.inputs (AliasMemory.pull a.toLayout initial template) (parameters p A.flat) (offsets A p)

def outputCell (a : Layout) (A : FlatAlloc) (p : Parameters) (base : Role → Nat)
    (v : Views a A p base) (i : OutputLayout.Coordinate p) : Fin a.capacity :=
  ⟨base .output+width .output*OutputLayout.address p i,by
    rw [show OutputLayout.address p i=offset p i.1.val i.2.1.val i.2.2.1.val i.2.2.2.val .output from rfl,
      ←v.address _ _ _ _ .output ⟨i.1.isLt,i.2.1.isLt,i.2.2.1.isLt,i.2.2.2.isLt⟩]
    exact a.bounded _ (v.live _ _ _ _ .output ⟨i.1.isLt,i.2.1.isLt,i.2.2.1.isLt,i.2.2.2.isLt⟩)⟩

def readReal (cell : MemCell) : ℝ :=
  match cell.readAs .real with
  | some (some x) => x
  | _ => 0

/-- Complete launch success, physical output readback, flat trace safety, and
the full physical typed-cell complement. Logical input agreement is precisely
the ordinary coordinate-preserving resolve_neg/materialization contract. -/
theorem launch (a : Layout) (A : FlatAlloc) (p : Parameters) (BD : Nat)
    (base : Role → Nat) (views : Views a A p base)
    (initial : AliasMemory.Physical a.toLayout) (template : BlockState)
    (xs : RealProofs.Inputs) (agree : LogicalAgreement.Agree p (logicalInputs a A p initial template) xs)
    (bounds : RegionBounds) (windows : ViewBounds.Windows bounds (parameters p A.flat) (offsets A p))
    (hD : p.headDim≤BD) (hstage : p.stage=3) (reciprocal : Bool)
    (nonoverlap : OutputLayout.NonOverlapping p)
    (fresh : FreshBytes a (GridFootprint.window A p)) :
    ∃ t, ∃ final : AliasMemory.Physical a.toLayout,
      Nonempty (Kernel.GridLaunchedOrdinary (GridLaunch.kernel A p BD reciprocal) (GridCoordinates.grid p)
        (AliasMemory.pull a.toLayout initial template) t) ∧
      (∀ i : OutputLayout.Coordinate p,
        readReal (final (outputCell a A p base views i)) =
          RealProofs.expected p.heads p.context p.headDim i.1.val i.2.1.val i.2.2.1.val i.2.2.2.val xs) ∧
      (∀ idx : GridIndex (GridCoordinates.grid p),
        Stmt.TraceSafeList bounds (GridLaunch.kernel A p BD reciprocal).body
          ((AliasMemory.pull a.toLayout initial template).withGridIndex idx)) ∧
      (∀ cell : Fin a.capacity,
        (∀ k, GridFootprint.window A p k → a.address k≠cell.val) → final cell=initial cell) := by
  obtain ⟨t,final,ht,hr,hframe⟩ := PhysicalCells.launch a A p BD initial template hD hstage reciprocal nonoverlap fresh
  refine ⟨t,final,ht,?_,GridLaunch.safe A bounds _ p BD reciprocal windows,hframe⟩
  intro i
  have hp := active_read a A p base views final i.1.val i.2.1.val i.2.2.1.val i.2.2.2.val .output
    ⟨i.1.isLt,i.2.1.isLt,i.2.2.1.isLt,i.2.2.2.isLt⟩
  have hr' := hr i
  change readReal (AliasMemory.read a.toLayout final (key A p i.1.val i.2.1.val i.2.2.1.val i.2.2.2.val .output)) =
    GridLaunch.expected A (AliasMemory.pull a.toLayout initial template) p i at hr'
  rw [hp] at hr'
  exact hr'.trans (LogicalAgreement.expected p _ xs agree _ _ _ _
    i.1.isLt i.2.1.isLt i.2.2.1.isLt i.2.2.2.isLt)

/-- No logical output coordinate exists on the host's empty fast path. -/
theorem empty_output (p : Parameters) (h : p.batches*p.heads*p.context*p.headDim=0) :
    IsEmpty (OutputLayout.Coordinate p) := by
  refine ⟨fun i => ?_⟩
  have hB := i.1.isLt
  have hH := i.2.1.isLt
  have hN := i.2.2.1.isLt
  have hD := i.2.2.2.isLt
  have hp := Nat.mul_pos (Nat.mul_pos (Nat.mul_pos (by omega : 0<p.batches)
    (by omega : 0<p.heads)) (by omega : 0<p.context)) (by omega : 0<p.headDim)
  omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Host

