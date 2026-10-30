import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.GridFootprint
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.FlatCausal
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.FlatTraceSafety

/-! Whole-grid successful Real execution and full shared-memory frame. Every
program starts from the same input memory; fresh nonoverlapping output layout
gives unique ownership. The empty grid writes no cells. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.GridLaunch
open VeriTile.Triton Memory Relocation GridCoordinates GridFootprint
noncomputable section
open scoped Classical
set_option maxHeartbeats 1500000
set_option linter.unusedSimpArgs false

def kernel (A : FlatAlloc) (p : Parameters) (BD : Nat) (reciprocal : Bool) : Kernel where
  inputs := [A.flat]
  outputs := [A.flat]
  body := A.flattenStmts (Verified.code 128 64 BD p reciprocal)

def expected (A : FlatAlloc) (s : BlockState) (p : Parameters) (i : OutputLayout.Coordinate p) : ℝ :=
  RealProofs.expected p.heads p.context p.headDim i.1.val i.2.1.val i.2.2.1.val i.2.2.2.val
    (LogicalInputs.inputs s (parameters p A.flat) (offsets A p))

theorem one (A : FlatAlloc) (s : BlockState) (p : Parameters) (BD : Nat)
    (hD : p.headDim ≤ BD) (hstage : p.stage=3) (reciprocal : Bool)
    (hn : OutputLayout.NonOverlapping p) (idx : GridIndex (grid p)) :
    ∃ f : Kernel.ExecFrame (kernel A p BD reciprocal) (s.withGridIndex idx),
      f.writes = writes A s p BD idx ∧
      ∀ i, Output.active (cfg A s p idx) 128 BD i = Bool.true →
        f.final.readMem A.flat (Output.address (cfg A s p idx) 128 BD i) =
          RealProofs.expected p.heads p.context p.headDim (cfg A s p idx).z (cfg A s p idx).h
            ((cfg A s p idx).startM*128+i.1.val) i.2.1.val
            (LogicalInputs.inputs s (parameters p A.flat) (offsets A p)) := by
  obtain ⟨t,ht,hr,hf⟩ := FlatCausal.executes A (s.withGridIndex idx) p BD hD hstage reciprocal
  let f : Kernel.ExecFrame (kernel A p BD reciprocal) (s.withGridIndex idx) := {
    final := t
    writes := writes A s p BD idx
    h_exec := ht
    h_writeWithin := by
      intro r o hnot
      symm
      apply hf r o
      by_cases he : r=A.flat
      · right
        intro i hi hie
        exact hnot ⟨he,i,hi,hie.symm⟩
      · exact Or.inl he }
  refine ⟨f,rfl,?_⟩
  intro i hi
  exact hr i hi (OutputLayout.active_injective (cfg A s p idx) 128 BD (program A s p idx) hn)

def frames (A : FlatAlloc) (s : BlockState) (p : Parameters) (BD : Nat)
    (hD : p.headDim ≤ BD) (hstage : p.stage=3) (reciprocal : Bool)
    (hn : OutputLayout.NonOverlapping p) : Kernel.GridFrames (kernel A p BD reciprocal) (grid p) s :=
  fun idx => Classical.choose (one A s p BD hD hstage reciprocal hn idx)

theorem frames_writes (A : FlatAlloc) (s : BlockState) (p : Parameters) (BD : Nat)
    (hD : p.headDim ≤ BD) (hstage : p.stage=3) (reciprocal : Bool)
    (hn : OutputLayout.NonOverlapping p) (idx : GridIndex (grid p)) :
    (frames A s p BD hD hstage reciprocal hn idx).writes = writes A s p BD idx :=
  (Classical.choose_spec (one A s p BD hD hstage reciprocal hn idx)).1

theorem launch (A : FlatAlloc) (s : BlockState) (p : Parameters) (BD : Nat)
    (hD : p.headDim ≤ BD) (hstage : p.stage=3) (reciprocal : Bool)
    (hn : OutputLayout.NonOverlapping p) :
    ∃ t, (Nonempty (Kernel.GridLaunchedOrdinary (kernel A p BD reciprocal) (grid p) s t)) ∧
      (∀ i : OutputLayout.Coordinate p,
        t.readMem A.flat (OutputLayout.address p i+A.base p.Out)=expected A s p i) ∧
      (∀ r o, ¬window A p (r,o) → t.mem r o=s.mem r o) := by
  let fs := frames A s p BD hD hstage reciprocal hn
  have hdis : Kernel.GridWritesDisjoint fs := by
    intro a b hne
    rw [show (fs a).writes = writes A s p BD a from frames_writes A s p BD hD hstage reciprocal hn a,
      show (fs b).writes = writes A s p BD b from frames_writes A s p BD hD hstage reciprocal hn b]
    exact disjoint A s p BD hn a b hne
  refine ⟨Kernel.mergeFrames (grid p) s fs,⟨fs,hdis,rfl⟩,?_,?_⟩
  · intro i
    let idx := index p i
    have hw : (fs idx).writes (A.flat,OutputLayout.address p i+A.base p.Out) := by
      rw [show (fs idx).writes = writes A s p BD idx from frames_writes A s p BD hD hstage reciprocal hn idx]
      exact ⟨rfl,lane p BD hD i,lane_active A s p BD hD i,lane_address A s p BD hD i⟩
    have hm := Kernel.mergeFrames_mem_written hdis idx A.flat (OutputLayout.address p i+A.base p.Out) hw
    have hr := (Classical.choose_spec (one A s p BD hD hstage reciprocal hn idx)).2
      (lane p BD hD i) (lane_active A s p BD hD i)
    rw [lane_address A s p BD hD i] at hr
    simp only [BlockState.readMem,hm]
    change (fs idx).final.readMem A.flat (OutputLayout.address p i+A.base p.Out) = _
    change (fs idx).final.readMem A.flat (OutputLayout.address p i+A.base p.Out) = _ at hr
    rw [hr]
    change RealProofs.expected _ _ _ (cfg A s p (index p i)).z (cfg A s p (index p i)).h
      ((cfg A s p (index p i)).startM*128+(lane p BD hD i).1.val) i.2.2.2.val _ = _
    rw [index_batch,index_head,lane_row]
    rfl
  · intro r o hnot
    apply Kernel.mergeFrames_mem_unwritten
    rintro ⟨idx,hw⟩
    rw [show (fs idx).writes = writes A s p BD idx from frames_writes A s p BD hD hstage reciprocal hn idx] at hw
    exact hnot (within A s p BD idx (r,o) hw)

theorem safe (A : FlatAlloc) (bounds : RegionBounds) (s : BlockState) (p : Parameters) (BD : Nat) (reciprocal : Bool)
    (hv : ViewBounds.Windows bounds (parameters p A.flat) (offsets A p)) :
    ∀ idx : GridIndex (grid p), Stmt.TraceSafeList bounds (kernel A p BD reciprocal).body (s.withGridIndex idx) := by
  intro idx
  have hp := program A s p idx
  exact FlatTraceSafety.safe A bounds (s.withGridIndex idx) p BD reciprocal hv ⟨hp.batch,hp.head,hp.row⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.GridLaunch
