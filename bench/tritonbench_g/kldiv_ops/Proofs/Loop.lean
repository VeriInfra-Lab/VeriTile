import bench.tritonbench_g.kldiv_ops.Proofs.Chunk

namespace VeriTile.Bench.Repairs.KldivOps.Program
open VeriTile.Triton FP.Structural FP.ProfiledRewrite Tile
noncomputable section
attribute [local irreducible] Tile.chunkRun Tile.prefixRun
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

abbrev Memory (α : Type) := RegionName→Nat→Cell α

def memoryState {α : Type} [Inhabited α] (mem : Memory α) : State α where
  mem := mem
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

/-- Independent scalar/chunk recurrence, without invoking the source evaluator.
The FP reduction remains the original ordered vector operation. -/
def advance {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red row start : Nat) (eps : α) (data : Memory α×α) : Memory α×α :=
  (chunkMemory M p flag red row start eps (memoryState data.1),
   nextCarry M p flag red eps data.2 row start (memoryState data.1))

def loopModel {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red row : Nat) (eps : α) (start : Nat) (data : Memory α×α) : Memory α×α :=
  if h : 0<p.block ∧ start<p.cols then
    loopModel M p flag red row eps (start+p.block) (advance M p flag red row start eps data)
  else data
termination_by p.cols-start
decreasing_by omega

theorem chunk_as_advance {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red row start : Nat) (eps carry : α) (s : State α) (h : Registers p red row eps carry s) :
    (chunkRun M p flag red row start eps carry s h).val.mem=
      (advance M p flag red row start eps (s.mem,carry)).1 ∧
    Registers p red row eps (advance M p flag red row start eps (s.mem,carry)).2
      (chunkRun M p flag red row start eps carry s h).val := by
  constructor
  · exact chunk_memory M p flag red row start eps carry s h
  · exact chunk_registers M p flag red row start eps carry s h

/-- Symbolic all-iteration execution, including no iterations at N=0. No
successful-execution or whole-output premise is supplied. -/
theorem loop_complete {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red row : Nat) (eps : α) (start : Nat) (carry : α) (s : State α)
    (h : Registers p red row eps carry s) (hb : 0<p.block) :
    ∃ final, rangeInt (engine M) "i" (start : Int) (p.cols : Int) p.block (forwardBody p flag red) s=some final ∧
      final.mem=(loopModel M p flag red row eps start (s.mem,carry)).1 ∧
      Registers p red row eps (loopModel M p flag red row eps start (s.mem,carry)).2 final := by
  by_cases active : start<p.cols
  · let next := (chunkRun M p flag red row start eps carry s h).val
    have he := (chunkRun M p flag red row start eps carry s h).property
    obtain ⟨hm,hr⟩ := chunk_as_advance M p flag red row start eps carry s h
    obtain ⟨final,runFinal,memFinal,regsFinal⟩ := loop_complete M p flag red row eps
      (start+p.block) (advance M p flag red row start eps (s.mem,carry)).2 next hr hb
    have hn : next.mem=(advance M p flag red row start eps (s.mem,carry)).1 := hm
    have statePair : (next.mem,(advance M p flag red row start eps (s.mem,carry)).2)=
        advance M p flag red row start eps (s.mem,carry) := by rw [hn]
    refine ⟨final,?_,?_,?_⟩
    · rw [rangeInt,if_neg (by omega),if_pos (by exact_mod_cast active),he]
      simpa only [Option.bind_some,←Int.natCast_add] using runFinal
    · rw [loopModel,dif_pos ⟨hb,active⟩]
      simpa only [statePair] using memFinal
    · rw [loopModel,dif_pos ⟨hb,active⟩]
      simpa only [statePair] using regsFinal
  · refine ⟨s,?_,?_,?_⟩
    · rw [rangeInt,if_neg (by omega),if_neg (by exact_mod_cast active)]
    · rw [loopModel,dif_neg (by tauto)]
    · rw [loopModel,dif_neg (by tauto)]
      exact h
termination_by p.cols-start
decreasing_by omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Program
