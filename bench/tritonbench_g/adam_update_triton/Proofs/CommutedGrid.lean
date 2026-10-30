import bench.tritonbench_g.adam_update_triton.Proofs.CommutedTile
import bench.tritonbench_g.swiglu_triton.Proofs.Grid
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.CommutedGrid
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Grid (row_unique)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

def runPrograms {α : Type} [Inhabited α] (sem : Tile.Semantics α) (params : Scalars α)
    (total chunk block : Nat) : List Nat → State α → Option (State α)
  | [],s => some s
  | pid::rest,s =>
    (run sem.backend (commutedSource total chunk block).surfaceBody (CommutedTile.programState params pid s)).bind
      (runPrograms sem params total chunk block rest)

theorem program_frame {α : Type} [Inhabited α] (sem : Tile.Semantics α) (params : Scalars α)
    (total chunk block written observed col : Nat) (s : State α) (region : RegionName)
    (lane : col<block) (different : observed≠written) :
    (CommutedTile.sourceRun sem params total chunk block written s).val.mem region (observed*block+col)=
      s.mem region (observed*block+col) := by
  apply CommutedTile.frame
  intro output
  right
  intro i _ equal
  exact different (row_unique block observed col written i.1.val lane i.1.isLt equal).1

/-- The programs may run in any permutation. Every program loads all three
entry words before either store, and distinct programs have disjoint windows. -/
theorem programs_result {α : Type} [Inhabited α] (sem : Tile.Semantics α) (params : Scalars α)
    (total chunk block : Nat) (pids : List Nat) (unique : pids.Nodup) (s : State α) :
    ∃ final,runPrograms sem params total chunk block pids s=some final ∧
      (∀ output pid col,pid∈pids → col<block → pid*block+col<chunk →
        final.mem (CommutedTile.outputRegion output) (pid*block+col)=Cell.mk .real
          (CommutedTile.scalar sem params output ((s.mem "P" (pid*block+col)).read .real)
            ((s.mem "G" (pid*block+col)).read .real) ((s.mem "M" (pid*block+col)).read .real))) ∧
      (∀ region address,(∀ pid∈pids,CommutedTile.Outside chunk block pid region address) →
        final.mem region address=s.mem region address) := by
  induction pids generalizing s with
  | nil =>
    refine ⟨s,rfl,?_,?_⟩
    · intro output pid col member; simp at member
    · intro region address outside; rfl
  | cons head rest ih =>
    obtain ⟨notMember,uniqueRest⟩ := List.nodup_cons.mp unique
    let middle := (CommutedTile.sourceRun sem params total chunk block head s).val
    obtain ⟨final,ran,values,frame⟩ := ih uniqueRest middle
    refine ⟨final,?_,?_,?_⟩
    · simp only [runPrograms,(CommutedTile.sourceRun sem params total chunk block head s).property,Option.bind_some]
      exact ran
    · intro output pid col member lane active
      rcases List.mem_cons.mp member with same | tail
      · subst pid
        have keep : final.mem (CommutedTile.outputRegion output) (head*block+col)=
            middle.mem (CommutedTile.outputRegion output) (head*block+col) := by
          apply frame
          intro other member otherOutput
          right
          intro i _ equal
          have same := (row_unique block head col other i.1.val lane i.1.isLt equal).1
          exact notMember (same ▸ member)
        rw [keep]
        exact CommutedTile.readback sem params total chunk block head s output ⟨⟨col,lane⟩,PUnit.unit⟩ active
      · have different : pid≠head := fun same => notMember (same ▸ tail)
        rw [values output pid col tail lane active]
        rw [program_frame sem params total chunk block head pid col s "P" lane different,
          program_frame sem params total chunk block head pid col s "G" lane different,
          program_frame sem params total chunk block head pid col s "M" lane different]
    · intro region address outside
      rw [frame region address (fun pid member => outside pid (List.mem_cons_of_mem head member))]
      exact CommutedTile.frame sem params total chunk block head s region address
        (outside head List.mem_cons_self)

def Outside (chunk : Nat) (region : RegionName) (address : Nat) : Prop :=
  (region≠"P" ∧ region≠"M") ∨ chunk≤address

theorem whole {α : Type} [Inhabited α] (sem : Tile.Semantics α) (params : Scalars α)
    (total chunk block : Nat) (positive : 0<block) (schedule : List Nat)
    (complete : schedule.Perm (List.range ((chunk+block-1)/block))) (s : State α) :
    ∃ final,runPrograms sem params total chunk block schedule s=some final ∧
      (∀ output i,i<chunk → final.mem (CommutedTile.outputRegion output) i=Cell.mk .real
        (CommutedTile.scalar sem params output ((s.mem "P" i).read .real)
          ((s.mem "G" i).read .real) ((s.mem "M" i).read .real))) ∧
      (∀ region address,Outside chunk region address → final.mem region address=s.mem region address) := by
  obtain ⟨final,ran,values,frame⟩ := programs_result sem params total chunk block schedule
    (complete.nodup_iff.mpr List.nodup_range) s
  refine ⟨final,ran,?_,?_⟩
  · intro output i active
    have member : i/block∈schedule := by
      apply complete.mem_iff.mpr
      apply List.mem_range.mpr
      have h := Nat.mod_lt i positive
      have div : i/block*block+i%block=i := by simpa only [Nat.mul_comm] using Nat.div_add_mod i block
      apply (Nat.lt_div_iff_mul_lt (by omega)).mpr
      omega
    have lane := Nat.mod_lt i positive
    have split : i/block*block+i%block=i := by simpa only [Nat.mul_comm] using Nat.div_add_mod i block
    simpa only [split] using values output (i/block) (i%block) member lane (by omega)
  · intro region address outside
    apply frame
    intro pid member output
    rcases outside with ⟨p,m⟩ | above
    · left
      cases output <;> assumption
    · right
      intro i active
      omega

#axiomsClean programs_result
#axiomsClean whole
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.CommutedGrid
