import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.SourceProgram
namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.SourceExecution
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def bodyExecution {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // run (sourceBackend M) (body p)
      ((preExecution M p s).val.setReg "start_mark" .int [] (fun _ => 0)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [body, source, ComputeKernel.surfaceBody]
  try simp only [eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
  simp [preExecution, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    evalOffsets, store, numeric, bop, natLt, numericEq, ComputeDType.eraseDType,
    ofFloat, toFloat, FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, sourceBackend, integralFloorDiv, integralMod]
  rfl

/-- Both branches use the source's exact dynamic 0/1 bound. -/
def execution {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec (sourceBackend M) (source p) s = some t} := by
  by_cases hb : blockActive p s
  · refine ⟨(bodyExecution M p s).val, ?_⟩
    change run (sourceBackend M) (source p).surfaceBody s = _
    rw [source_body, run_append]
    conv_lhs => rw [(preExecution M p s).property]
    simp only [Option.bind_some, run, step, evalOp_unfold, preExecution_mask,
      hb, ↓reduceIte, Option.bind_some, bind, pure, Option.bind, Option.pure_def]
    rw [FP.Structural.rangeInt]
    simp only [Nat.one_ne_zero, ↓reduceIte, Int.zero_lt_one, Int.zero_add]
    conv_lhs => rw [(bodyExecution M p s).property]
    simp only [Option.bind_some, bind, Option.bind]
    rw [FP.Structural.rangeInt]
    simp [run]
  · refine ⟨(preExecution M p s).val, ?_⟩
    change run (sourceBackend M) (source p).surfaceBody s = _
    rw [source_body, run_append]
    conv_lhs => rw [(preExecution M p s).property]
    simp only [Option.bind_some, run, step, evalOp_unfold, preExecution_mask,
      hb, ↓reduceIte, Option.bind_some, bind, pure, Option.bind, Option.pure_def]
    rw [FP.Structural.rangeInt]
    simp [run]

#axiomsClean bodyExecution
#axiomsClean execution
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.SourceExecution
