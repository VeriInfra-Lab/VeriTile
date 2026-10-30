import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.GridIO
import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.IOSafety
open VeriTile Triton FP.Structural Whole IOBinding
open VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
variable {α β : Type} [Inhabited β]

theorem source_bounds (M : Program → Algebra β) (c : Call) (m : Metadata c) (scale : ℝ)
    (work : List Program) (encode : RegionName → Nat → α → β) (observe : β → α)
    (xs : Interface.Inputs α c) (s : State β) (v : Valid c m)
    (h : Reads c m xs encode s) (bounds : RegionName → Nat)
    (hb : (Interface.io M c m scale work encode observe).InBounds bounds xs)
    (pid : Program) (hp : pid∈programs c.grid) :
    SourceSafety.Bounds (c.parameters scale) (start c.grid pid s) bounds := by
  obtain ⟨hbatch,hhead,hblock⟩ := (mem_programs c.grid pid.1 pid.2.1 pid.2.2).mp hp
  have hl := length_bound c m v pid.1 hbatch
  have hgroup := grouped_head c m v pid.2.1 hhead
  constructor
  · exact hb.2.1 ⟨2,by simp [Interface.io,Interface.inputs]⟩ (⟨pid.1,hbatch⟩,PUnit.unit) trivial
  · exact hb.2.1 ⟨3,by simp [Interface.io,Interface.inputs]⟩ (⟨pid.1,hbatch⟩,PUnit.unit) trivial
  · intro _ d
    exact hb.2.1 ⟨0,by simp [Interface.io,Interface.inputs]⟩
      (⟨pid.1,hbatch⟩,⟨pid.2.1,hhead⟩,d,PUnit.unit) trivial
  · intro i hi
    have hn := (active_eq h v scale pid hbatch i).mp hi
    have hlb := v.locationBounds pid.1 hbatch (c.maxLength-m.len pid.1+(pid.2.2*32+i.1.val))
      (by omega)
    simp only [windowIndex,length_eq h scale pid hbatch]
    exact hb.2.1 ⟨4,by simp [Interface.io,Interface.inputs]⟩
      (⟨_,hlb⟩,PUnit.unit) trivial
  · intro i hi d
    have hn := (active_eq h v scale pid hbatch i).mp hi
    have hp := (v.pages ⟨pid.1,hbatch⟩ _ hn).2
    rw [keyOffset,page_eq h v scale pid hbatch i hn]
    exact hb.2.1 ⟨1,by simp [Interface.io,Interface.inputs]⟩
      (⟨_,hp⟩,⟨pid.2.1/(c.heads/c.kvHeads),hgroup⟩,d,PUnit.unit) trivial
  · intro i hi
    have hn := (active_eq h v scale pid hbatch i).mp hi
    have hnMax : pid.2.2*32+i.1.val<c.maxLength := by omega
    rw [offset_eq h scale pid hbatch i]
    exact hb.2.2.1 ⟨0,by simp [Interface.io,Interface.outputs]⟩
      (⟨pid.1,hbatch⟩,⟨pid.2.1,hhead⟩,⟨_,hnMax⟩,PUnit.unit) hn

theorem schedule (M : Program → Algebra β) (c : Call) (m : Metadata c) (scale : ℝ)
    (allWork : List Program) (encode : RegionName → Nat → α → β) (observe : β → α)
    (xs : Interface.Inputs α c) (v : Valid c m) (bounds : RegionName → Nat)
    (hb : (Interface.io M c m scale allWork encode observe).InBounds bounds xs)
    (work : List Program) (within : ∀ pid∈work, pid∈programs c.grid)
    (s : State β) (h : Reads c m xs encode s) :
    ScheduleSafe bounds (work.map (Interface.invocation M c scale)) s := by
  induction work generalizing s with
  | nil => trivial
  | cons pid tail ih =>
    change (Interface.invocation M c scale pid).Safe bounds s ∧ _
    constructor
    · exact SourceSafety.source_safe (M pid) (c.parameters scale) (start c.grid pid s) bounds
        (source_bounds M c m scale allWork encode observe xs s v h bounds hb pid (within pid (by simp)))
    · intro next ran
      change exec (sourceBackend (M pid)) (source (c.parameters scale)) (start c.grid pid s) = some next at ran
      have eqNext := Option.some.inj
        (ran.symm.trans (SourceExecution.execution (M pid) (c.parameters scale) (start c.grid pid s)).property)
      have preserved : ReadsEq (c.parameters scale) next s := by
        intro r hr
        funext o
        rw [eqNext,SourceCorrectness.execution_memory]
        exact SourceCorrectness.written_frame _ _ _ r o (Or.inr (Or.inl hr))
      exact ih (fun pid member => within pid (by simp [member])) next (h.preserve scale preserved)

#axiomsClean schedule
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.IOSafety
