import bench.tritonbench_g.chunk_linear_attn.Proofs.PairOutput
import bench.tritonbench_g.chunk_linear_attn.Proofs.PairOutputAccess

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.PairOutput
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 3000000
variable {α : Type} [Inhabited α]

theorem range_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (bounds : RegionName → Nat) (pid : PID) (input : Memory α) (s : State α)
    (ctx : OutputReady p pid input s) (a : OutputAcc p α) (ha : HasOutput p a s)
    (hh : Head p pid s) (capacity : StageSafety.OutputCapacity p bounds)
    (head : pid.z < p.B*p.H) (time : pid.y < p.NT) :
    Safety.rangeSafe (engine (withDot M D)) bounds "i_k" 0 (keyBlocks p) 1 (parts p scale).body s := by
  let invariant := fun (_ : Nat) (t : State α) =>
    OutputReady p pid input t ∧ ∃ values, HasOutput p values t ∧ Head p pid t
  apply Safety.rangeSafe_of_invariant _ _ _ _ _ _ invariant
  · intro n _ t ⟨ready,values,available,offset⟩
    exact body_safe M D p scale bounds pid input t ready offset values available n capacity head time
  · intro n _ t ⟨ready,values,available,offset⟩ u ran
    let actual := outputBody M D p scale pid input t n values ready available
    rw [body_run M D p scale pid input t n ready offset] at ran
    have same : actual.val=u := Option.some.inj (actual.property.symm.trans ran)
    subst u
    have next := outputBody_ready M D p scale pid input t n values ready available
    exact ⟨next.1,_,next.2,body_head M D p scale pid input t n values ready available offset⟩
  · exact ⟨ctx,a,ha,hh⟩

theorem source_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (bounds : RegionName → Nat) (s : State α) (capacity : StageSafety.OutputCapacity p bounds)
    (head : s.pids 2 < p.B*p.H) (time : s.pids 1 < p.NT) :
    TraceSafe (engine (withDot M D)) bounds (forwardOHoisted p scale) s := by
  let pid : PID := ⟨s.pids 0,s.pids 1,s.pids 2⟩
  have ready := entry_ready M D p scale s
  obtain ⟨mid,loopRan,midReady,midValues,midHead⟩ :=
    loop M D p scale pid s.mem (entry M D p scale s) ready.1 ready.2.1 ready.2.2
  have h0 : (parts p scale).start=.constNat 0 := rfl
  have hn : (parts p scale).stop=.div .nat .nil
    (.sub .nat .nil (.add .nat .nil (.constNat p.K) (.constNat p.BK)) (.constNat 1)) (.constNat p.BK) := rfl
  have h1 : (parts p scale).stride=.constNat 1 := rfl
  let loopStmt := ComputeStmt.forRangeDyn "i_k" (parts p scale).start (parts p scale).stop
    (parts p scale).stride (parts p scale).body
  have loopStep : step (engine (withDot M D)) loopStmt (entry M D p scale s)=some mid := by
    simpa [loopStmt,step,h0,hn,h1,evalOp_unfold,Option.bind,numeric,bop,keyBlocks] using loopRan
  have loopSafe : Safety.stepSafe (engine (withDot M D)) bounds loopStmt (entry M D p scale s) := by
    have safe := range_safe M D p scale bounds pid s.mem (entry M D p scale s) ready.1 _ ready.2.1
      ready.2.2 capacity head time
    simpa [loopStmt,Safety.stepSafe_unfold,Safety.safeOp_unfold,h0,hn,h1,evalOp_unfold,
      Option.bind,numeric,bop,keyBlocks] using safe
  unfold TraceSafe
  rw [decomposition,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨entry_safe M D p scale bounds s,?_⟩
  intro t ran
  have same : entry M D p scale s=t := Option.some.inj ((entry_run M D p scale s).symm.trans ran)
  subst t
  apply Safety.traceSafeList_cons_of_step loopSafe loopStep
  rw [after]
  exact StageSafety.output_finish_safe M D p scale bounds pid s.mem mid midReady _ midValues capacity head

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.PairOutput
