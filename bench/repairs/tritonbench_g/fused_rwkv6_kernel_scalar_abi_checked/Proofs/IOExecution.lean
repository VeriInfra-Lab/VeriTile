import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.SourceCells

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Interface.Execution
open VeriTile Triton FP.Structural FusedRwkv6Kernel Source SourceCells
noncomputable section
attribute [local instance] Classical.propDecidable
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

def OutputAt (c : Shape) (r : RegionName) (b : Nat) : Prop :=
  (r="o" ∧ b<c.NK*c.Z*c.T*c.V) ∨ (c.final=Bool.true ∧ r="ht" ∧ b<c.Z*c.K*c.V)

theorem output_at (c : Shape) (r : FusedRwkv6Kernel.Region) (b : Nat) :
    OutputAt c (regionName r) b ↔ Output c r b := by
  cases r <;> simp [OutputAt,Output,regionName]

theorem invocation_program (M : Algebra α) (c : Shape) (scale : α)
    (p : FusedRwkv6Kernel.Program) (s : State α) :
    program ((invocation M c scale p).initial s)=p := by
  cases p
  simp [program,invocation,Invocation.initial,bindRegisters,State.setReg]

theorem invocation_scalar (M : Algebra α) (c : Shape) (scale : α)
    (p : FusedRwkv6Kernel.Program) (s : State α) :
    ((invocation M c scale p).initial s).regs .real [] "scale"=some (fun _ => scale) := by
  simp [invocation,Invocation.initial,bindRegisters,State.setReg]

theorem one (M : Algebra α) (c : Shape) (plan : Plans c)
    (reduction : ReductionMatches M c plan) (scale : α)
    (p : FusedRwkv6Kernel.Program) (valid : Valid c p) (s : State α) :
    ∃ t, (invocation M c scale p).run s=some t ∧
      words t.mem=FusedRwkv6Kernel.runProgram (ops M) c plan p scale (words s.mem) ∧
      (∀ r b,RealCell (t.mem (regionName r) b) ↔
        RealCell (s.mem (regionName r) b) ∨ Writes c p r b) ∧
      (∀ r b,¬OutputAt c r b → t.mem r b=s.mem r b) := by
  let start := (invocation M c scale p).initial s
  obtain ⟨t,run,mem⟩ := source_memory M c start scale (invocation_scalar M c scale p s)
  refine ⟨t,run,?_,?_,?_⟩
  · funext r b
    change (t.mem (regionName r) b).read .real=_
    rw [mem,program_model M c start scale plan (reduction start scale)]
    change FusedRwkv6Kernel.runProgram (ops M) c plan (program start) scale (memory start) r b=_
    rw [invocation_program]
    rfl
  · intro r b
    rw [mem,final_tag]
    change RealCell (s.mem (regionName r) b) ∨ Writes c (program start) r b ↔ _
    rw [invocation_program]
  · intro r b outside
    rw [mem]
    by_cases ho : r="o"
    · subst r
      change finalMemory M c start scale (regionName .o) b=s.mem "o" b
      rw [final_frame]
      · rfl
      · rw [invocation_program]
        exact fun h => outside ((output_at c .o b).mpr (writes_output c p valid h))
    by_cases ht : r="ht"
    · subst r
      change finalMemory M c start scale (regionName .ht) b=s.mem "ht" b
      rw [final_frame]
      · rfl
      · rw [invocation_program]
        exact fun h => outside ((output_at c .ht b).mpr (writes_output c p valid h))
    · rw [untracked_frame M c start scale r b ho ht]
      rfl

theorem grid (M : Algebra α) (c : Shape) (plan : Plans c)
    (reduction : ReductionMatches M c plan) (scale : α)
    (order : List FusedRwkv6Kernel.Program) (valid : ∀ p∈order,Valid c p) (s : State α) :
    ∃ t, runSchedule (order.map (invocation M c scale)) s=some t ∧
      words t.mem=FusedRwkv6Kernel.runGrid (ops M) c plan scale order (words s.mem) ∧
      (∀ r b,RealCell (t.mem (regionName r) b) ↔
        RealCell (s.mem (regionName r) b) ∨ ∃ p∈order,Writes c p r b) ∧
      (∀ r b,¬OutputAt c r b → t.mem r b=s.mem r b) := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl,rfl,by simp,by intros;rfl⟩
  | cons p rest ih =>
    obtain ⟨mid,hm,wm,tm,fm⟩ := one M c plan reduction scale p (valid p (by simp)) s
    obtain ⟨t,ht,wt,tt,ft⟩ := ih (fun q hq => valid q (List.mem_cons_of_mem _ hq)) mid
    refine ⟨t,?_,?_,?_,?_⟩
    · simp only [List.map_cons,runSchedule,hm,Option.bind_some,ht]
    · rw [wt,wm]
      rfl
    · intro r b
      rw [tt,tm]
      simp [List.mem_cons,or_assoc,or_and_right,exists_or]
    · intro r b outside
      exact (ft r b outside).trans (fm r b outside)

#axiomsClean grid
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Interface.Execution
