import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.PairOutputAddress

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.PairOutput
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem entry_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts p scale).before s := by
  apply StageSafety.pure_trace
  simp [parts,forwardOHoisted,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD]
  all_goals
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,StageSafety.safeBlockPtr2]
    simp [Safety.safeOp_unfold]

theorem body_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (bounds : RegionName → Nat) (pid : PID) (input : Memory α) (s : State α)
    (ctx : OutputReady p pid input s) (qhead : Head p pid s) (a : OutputAcc p α) (ha : HasOutput p a s)
    (kb : Nat) (capacity : StageSafety.OutputCapacity p bounds)
    (head : pid.z < p.B * p.H) (time : pid.y < p.NT) :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts p scale).body
      (s.setReg "i_k" .nat [] (fun _ => kb)) := by
  have original := StageSafety.output_body_safe M D p scale bounds pid input s ctx a ha kb capacity head time
  have splitOriginal := (Safety.traceSafeList_append (engine (withDot M D)) bounds
    ((outputParts p scale).body.take 2) ((outputParts p scale).body.drop 2)
    (s.setReg "i_k" .nat [] (fun _ => kb))).mp
    (by simpa only [List.take_append_drop] using original)
  rw [←List.take_append_drop 2 (parts p scale).body,Safety.traceSafeList_append]
  constructor
  · apply StageSafety.pure_trace
    simp [parts,forwardOHoisted,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD]
    all_goals
      simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,StageSafety.safeBlockPtr2]
      simp [Safety.safeOp_unfold]
  · intro t ran
    rw [first_run M D p scale pid input s kb ctx qhead] at ran
    rw [body_suffix]
    exact splitOriginal.2 t ran


#axiomsClean body_safe

end
end VeriTile.Bench.Repairs.ChunkLinearAttn.PairOutput
