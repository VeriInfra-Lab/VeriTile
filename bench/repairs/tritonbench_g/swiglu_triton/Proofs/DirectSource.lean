import bench.repairs.tritonbench_g.swiglu_triton.Proofs.DirectTile
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.DirectPhysical

namespace VeriTile.Bench.Repairs.SwigluTriton.DirectSource
open VeriTile.Triton FP.Structural
open Memory DirectPhysical
noncomputable section
open Classical
set_option maxHeartbeats 6000000

def region (backward : Bool) : Role → RegionName
  | .a => "A" | .b => "B" | .third => if backward then "DC" else "C"

theorem region_injective (backward : Bool) : Function.Injective (region backward) := by
  intro a b same
  cases backward <;> cases a <;> cases b <;> try rfl
  all_goals simp [region] at same

theorem output_region (backward : Bool) (ch : Channel) (on : enabled backward ch) :
    region backward (outputRole ch)=outputRegion ch := by
  cases backward <;> cases ch <;> try rfl
  all_goals simp [enabled] at on

/-- The actual chunk slice shifts each symbolic base by first*block words.
All loaded cells are reads of this very physical memory, including in-place
A/B. No independent entry array or assumed kernel output is introduced. -/
def state {M α β : Type} [Inhabited α] (ops : Ops M α β) (backward : Bool)
    (p : Layout) (first : Nat) (s : M) : State α where
  mem := fun reg offset => .mk .real (
    if reg="A" then ops.read s ((p.allocation .a).key (first*p.cols+offset))
    else if reg="B" then ops.read s ((p.allocation .b).key (first*p.cols+offset))
    else if reg=region backward .third then ops.read s ((p.allocation .third).key (first*p.cols+offset))
    else default)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun dtype _ _ => defaultValue dtype

theorem state_read {M α β : Type} [Inhabited α] (ops : Ops M α β) (backward : Bool)
    (p : Layout) (first : Nat) (s : M) (role : Role) (offset : Nat) :
    (state ops backward p first s).mem (region backward role) offset=
      Cell.mk .real (ops.read s ((p.allocation role).key (first*p.cols+offset))) := by
  cases backward <;> cases role <;> simp [state,region]

theorem row_values {M α β : Type} [Inhabited α] (ops : Ops M α β) (algebra : Algebra α)
    (backward : Bool) (p : Layout) (first pid : Nat) (s : M)
    (ch : Channel) (on : enabled backward ch) (col : Nat) (active : col<p.cols)
    (bounded : p.cols≤p.block) :
    (DirectTile.sourceRun algebra backward p.block p.cols pid (state ops backward p first s)).val.mem
      (outputRegion ch) (pid*p.cols+col)=
      Cell.mk .real (DirectPhysical.value ops (DirectTile.scalar algebra) backward p s ch (first+pid) col) := by
  rw [DirectTile.readback algebra backward p.block p.cols pid (state ops backward p first s) ch on
    (⟨⟨col,lt_of_lt_of_le active bounded⟩,PUnit.unit⟩) active]
  cases backward <;> cases ch <;> try simp [enabled] at on
  all_goals simp [state,region,DirectPhysical.value,Layout.key,Nat.add_mul,Nat.add_assoc,DirectTile.scalar]

/-- Complete physical-memory refinement of the full source row. It includes
every valid private backing cell, not merely the active output values. -/
theorem row_refines {M α β : Type} [Inhabited α] (ops : Ops M α β) (algebra : Algebra α)
    (backward : Bool) (p : Layout) (valid : Valid p) (first pid : Nat)
    (rowBound : first+pid<p.rows) (s : M) :
    ∃ t,run algebra (directSource backward p.block p.cols).surfaceBody
        (DirectTile.rowState pid (state ops backward p first s))=some t ∧
      ∀ role offset,first*p.cols+offset<(p.allocation role).words →
        t.mem (region backward role) offset=Cell.mk .real
          (ops.read (DirectPhysical.row ops (DirectTile.scalar algebra) backward p (first+pid) s)
            ((p.allocation role).key (first*p.cols+offset))) := by
  refine ⟨_,(DirectTile.sourceRun algebra backward p.block p.cols pid (state ops backward p first s)).property,?_⟩
  intro role offset backed
  by_cases hit : ∃ ch,enabled backward ch ∧ role=outputRole ch ∧
      ∃ col,col<p.cols ∧ offset=pid*p.cols+col
  · obtain ⟨ch,on,same,col,active,equal⟩ := hit
    subst role offset
    rw [output_region backward ch on,row_values ops algebra backward p first pid s ch on col active valid.cols]
    have keyEq : ((p.allocation (outputRole ch)).key (first*p.cols+(pid*p.cols+col)))=
        p.key (outputRole ch) (first+pid) col := by simp [Layout.key,Nat.add_mul,Nat.add_assoc]
    rw [keyEq,DirectPhysical.row_values ops (DirectTile.scalar algebra) backward p valid (first+pid) s rowBound ch on col active]
  · have structural : DirectTile.Outside backward p.block p.cols pid (region backward role) offset := by
      intro ch on
      by_cases different : region backward role≠outputRegion ch
      · exact Or.inl different
      · right
        intro lane active equal
        have sameRole : role=outputRole ch := region_injective backward
          ((not_not.mp different).trans (output_region backward ch on).symm)
        exact hit ⟨ch,on,sameRole,lane.1.val,active,equal⟩
    rw [DirectTile.frame algebra backward p.block p.cols pid (state ops backward p first s) _ _ structural,
      state_read]
    congr 1
    symm
    apply DirectPhysical.row_frame
    intro ch on col active
    by_cases sameRole : role=outputRole ch
    · subst role
      apply Allocation.indices_disjoint
      intro words
      have sameOffset : offset=pid*p.cols+col := by
        simp only [Nat.add_mul] at words
        omega
      exact hit ⟨ch,on,rfl,col,active,sameOffset⟩
    · exact Allocation.spans_separate _ _ (valid.separate role (outputRole ch) sameRole) _ _
        (Allocation.key_contains _ _ backed)
        (key_backed p valid (outputRole ch) (first+pid) col rowBound active)

#axiomsClean row_refines
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.DirectSource
