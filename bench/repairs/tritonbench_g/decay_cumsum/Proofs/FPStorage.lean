import bench.repairs.tritonbench_g.decay_cumsum.Proofs.FPTileStable
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.PackedStorage
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.Run

namespace VeriTile.Bench.Repairs.DecayCumsum.FPStorage
open VeriTile.Triton FP.Structural FP.ProfiledRewrite Geometry Grid PackedStorage
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

abbrev Heap (α : Type) := Nat → Cell α
abbrev OutputIndex (h : Call) (kind : Kind) (w : Work h) :=
  {r : RegionName // r∈Roles.outputs kind} × Fin w.1.span × Fin h.BK

def entry (h : Call) (kind : Kind) (storage : Placement h kind) (w : Work h) (heap : Heap α) : State α where
  mem := lift storage w.1 heap
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then w.2.feature else if ax=1 then w.2.chunk else w.2.row
  numPids := fun ax => if ax=0 then h.features else if ax=1 then w.1.count else h.rows
  undef := fun d _ _ => defaultValue d

def execute (M : Algebra α) (h : Call) (kind : Kind) (storage : Placement h kind) (scale : ℝ)
    (w : Work h) (heap : Heap α) : Option (Heap α) :=
  (FP.Structural.exec (engine M) (Kernels.source kind h w.1 scale) (entry h kind storage w heap)).map
    (fun v => commit storage w.1 heap v.mem)

def outputAddress (h : Call) (kind : Kind) (storage : Placement h kind) (w : Work h)
    (i : OutputIndex h kind w) : Nat :=
  address storage i.1.val (Geometry.offset h w.1 w.2 i.2.1.val i.2.2.val)

def expected (M : Algebra α) (h : Call) (kind : Kind) (storage : Placement h kind) (scale : ℝ)
    (heap : Heap α) (w : Work h) (i : OutputIndex h kind w) : Cell α :=
  FPTile.expected M kind h w.1 w.2 scale (entry h kind storage w heap) i.1.val i.2.1.val i.2.2

theorem roles_outputs (kind : Kind) : FPTile.outputs kind=Roles.outputs kind := by cases kind <;> rfl
theorem roles_inputs (kind : Kind) : FPTile.inputs kind=Roles.inputs kind := by cases kind <;> rfl

theorem inside (h : Call) (kind : Kind) (storage : Placement h kind) (w : Work h) (i : OutputIndex h kind w) :
    footprint h kind storage w (outputAddress h kind storage w i) :=
  ⟨i.1.val,Roles.output_member kind i.1.val i.1.property,i.2.1,i.2.2,rfl⟩

theorem complete (M : Algebra α) (h : Call) (kind : Kind) (storage : Placement h kind) (scale : ℝ)
    (w : Work h) (member : w∈schedule h) (heap : Heap α) :
    ∃ final, execute M h kind storage scale w heap=some final ∧
      (∀ i, final (outputAddress h kind storage w i)=expected M h kind storage scale heap w i) ∧
      (∀ addr, ¬footprint h kind storage w addr → final addr=heap addr) := by
  have phase := schedule_valid h w member
  have positive := (phase_positive h w.1 phase).2.1
  obtain ⟨v,he,output,frame⟩ := FPTile.complete M kind h w.1 positive w.2 scale (entry h kind storage w heap)
    (by simp [entry]) (by simp [entry]) (by simp [entry])
  refine ⟨commit storage w.1 heap v.mem,?_,?_,?_⟩
  · simp [execute,he]
  · intro i
    have bounds : shift h w.1+relative h w.1 w.2 i.2.1.val i.2.2.val<h.cells := by
      rw [relocated]
      exact offset_inside h w.1 phase w.2 i.2.1 i.2.2
    unfold outputAddress
    rw [←relocated,commit_output h kind storage w.1 heap v.mem i.1.val i.1.property _ bounds]
    exact output i.1.val (by simpa [roles_outputs] using i.1.property) i.2.1 i.2.2
  · intro addr outside
    apply commit_frame h kind storage w heap v.mem
    · simpa only [TileFrame,FPMemory.Frame,FPMemory.footprint,relative,FPPrefix.off,roles_outputs] using frame
    · exact fun hw => outside (writes_footprint h kind storage w addr hw)

theorem expected_stable (M : Algebra α) (h : Call) (kind : Kind) (storage : Placement h kind) (scale : ℝ)
    (w : Work h) (member : w∈schedule h) (a b : Heap α)
    (stable : ∀ addr, footprint h kind storage w addr → a addr=b addr) (i : OutputIndex h kind w) :
    expected M h kind storage scale a w i=expected M h kind storage scale b w i := by
  apply FPTileStable.expected_stable M kind h w.1 (phase_positive h w.1 (schedule_valid h w member)).2.1
  intro r input t ht j
  change a (address storage r (shift h w.1+FPPrefix.off h w.1 w.2 t j))=
    b (address storage r (shift h w.1+FPPrefix.off h w.1 w.2 t j))
  apply stable
  refine ⟨r,Roles.input_member kind r (by simpa [roles_inputs] using input),⟨t,ht⟩,j,?_⟩
  congr 1
  exact relocated h w.1 w.2 t j.val

theorem grid_complete (M : Algebra α) (h : Call) (kind : Kind) (storage : Placement h kind) (scale : ℝ) (heap : Heap α) :
    ∃ final, Run.run (execute M h kind storage scale) (schedule h) heap=some final ∧
      (∀ w∈schedule h, ∀ i, final (outputAddress h kind storage w i)=expected M h kind storage scale heap w i) ∧
      (∀ addr, (∀ w∈schedule h, ¬footprint h kind storage w addr) → final addr=heap addr) := by
  exact Run.complete (execute M h kind storage scale) (OutputIndex h kind) (outputAddress h kind storage) id
    (expected M h kind storage scale) (footprint h kind storage) (inside h kind storage) (schedule h)
    (schedule_nodup h) (fun a ha b hb => disjoint h kind storage a b ha hb) (complete M h kind storage scale)
    (expected_stable M h kind storage scale) heap

#axiomsClean grid_complete
end
end VeriTile.Bench.Repairs.DecayCumsum.FPStorage
