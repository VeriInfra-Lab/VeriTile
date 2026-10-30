import bench.tritonbench_g.layer_norm_welfold.Proofs.ReadableSyntax

namespace VeriTile.Bench.Repairs.LayerNormWelfold.Readable
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem phase_run (M : Algebra α) (phase : Bool)
    (s t : State α) (X : RegionName) (S D B k : Nat) (rt : ReductionTree B)
    (values : Fin B → α) (mean : α) (h : Ready s t values mean) :
    ∃ final, run M (phaseBody phase X S D B rt) (t.setReg "k" .nat [] (fun _ => k))=some final ∧
      Ready s final (advance M phase s X D B k mean values) mean := by
  cases phase <;>
    simp [phaseBody,code,SourceCode.rowKernel,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.bop,
      FP.Structural.natLt,FP.Structural.numericLt,Region.cast,
      ofFloat,toFloat,h.values,h.mean,h.row,h.mem]
  all_goals constructor <;> simp [State.setReg,advance,loaded,zero,FP.Structural.bop,ofFloat,toFloat,
    Fin.eq_zero,h.mem,h.pids,h.undef,h.row,h.mean]
  all_goals first | rfl | (funext i; first | (split <;> simp_all [Fin.eq_zero]) | simp [FP.Structural.bop])
  all_goals try dsimp [FP.Structural.bop]
  all_goals try simp_all
  all_goals norm_num [FP.Structural.bop]
  all_goals rename_i i hi
  all_goals simp only [if_neg (Nat.not_lt_of_ge hi)]

theorem phase_loop (M : Algebra α) (phase : Bool) (s t : State α)
    (X : RegionName) (S D B start count : Nat) (rt : ReductionTree B)
    (values : Fin B → α) (mean : α) (h : Ready s t values mean) :
    ∃ final, FP.Structural.loop M "k" start (start+count) (phaseBody phase X S D B rt) t=some final ∧
      Ready s final (recurrence M phase s X D B mean start count values) mean := by
  induction count generalizing start t values with
  | zero => exact ⟨t,by simp [FP.Structural.loop],h⟩
  | succ count ih =>
    obtain ⟨u,hu,hp⟩ := phase_run M phase s t X S D B start rt values mean h
    obtain ⟨v,hv,hq⟩ := ih u (start+1) _ hp
    refine ⟨v,?_,hq⟩
    rw [FP.Structural.loop,if_pos (by omega),hu]
    simpa only [Option.bind_some,show start+Nat.succ count=(start+1)+count by omega] using hv

#axiomsClean phase_run
#axiomsClean phase_loop
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Readable
