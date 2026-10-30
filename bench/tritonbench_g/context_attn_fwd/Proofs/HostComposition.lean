import VeriTile.Triton.Launch.Composition
import VeriTile.Meta.StatementAudit

/-! A finite-grid assembly lemma. Successful program executions are constructed
by the caller from its value theorem; only address ownership is a premise. -/
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.HostComposition
open VeriTile.Triton
noncomputable section

variable {ι : Type} (g : Grid)

def writes (active : GridIndex g → ι → Prop)
    (address : GridIndex g → ι → MemCellAddr) (idx : GridIndex g) : WriteFootprint :=
  fun cell => ∃ i, active idx i ∧ address idx i = cell

def allWrites (active : GridIndex g → ι → Prop)
    (address : GridIndex g → ι → MemCellAddr) : WriteFootprint :=
  fun cell => ∃ idx, writes g active address idx cell

theorem assemble (k : Kernel) (initial : BlockState)
    (active : GridIndex g → ι → Prop) (address : GridIndex g → ι → MemCellAddr)
    (expected : GridIndex g → ι → ℝ)
    (programs : ∀ idx, ∃ final,
      exec k (initial.withGridIndex idx) = some final ∧
      (∀ i, active idx i → final.readMem (address idx i).1 (address idx i).2 = expected idx i) ∧
      BlockState.WriteWithin (writes g active address idx) (initial.withGridIndex idx) final)
    (ownership : ∀ idx jdx i j, active idx i → active jdx j →
      address idx i = address jdx j → idx = jdx) :
    ∃ final, Nonempty (Kernel.GridLaunchedOrdinary k g initial final) ∧
      (∀ idx i, active idx i → final.readMem (address idx i).1 (address idx i).2 = expected idx i) ∧
      BlockState.WriteWithin (allWrites g active address) initial final := by
  let frames : Kernel.GridFrames k g initial := fun idx => {
    final := Classical.choose (programs idx)
    writes := writes g active address idx
    h_exec := (Classical.choose_spec (programs idx)).1
    h_writeWithin := (Classical.choose_spec (programs idx)).2.2 }
  have hd : Kernel.GridWritesDisjoint frames := by
    intro idx jdx hne cell hi hj
    obtain ⟨i, hi, hei⟩ := hi
    obtain ⟨j, hj, hej⟩ := hj
    exact hne (ownership idx jdx i j hi hj (hei.trans hej.symm))
  refine ⟨Kernel.mergeFrames g initial frames, ⟨⟨frames, hd, rfl⟩⟩, ?_, ?_⟩
  · intro idx i hi
    have hm := Kernel.mergeFrames_mem_written hd idx (address idx i).1 (address idx i).2
      (show (frames idx).writes (address idx i) from ⟨i, hi, rfl⟩)
    have ho := (Classical.choose_spec (programs idx)).2.1 i hi
    simpa only [BlockState.readMem, hm] using ho
  · exact Kernel.mergeFrames_writeWithin frames

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.HostComposition
