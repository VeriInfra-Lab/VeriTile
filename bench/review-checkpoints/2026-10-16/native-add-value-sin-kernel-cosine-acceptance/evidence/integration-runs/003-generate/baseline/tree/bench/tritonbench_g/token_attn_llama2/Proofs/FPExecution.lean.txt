import bench.tritonbench_g.token_attn_llama2.Proofs.FPProgram
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
namespace VeriTile.Bench.TritonBenchG.TokenAttnLlama2.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def bodyExecution {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // run M (body p)
      ((preExecution M p s).val.setReg "start_mark" .nat [] (fun _ => 0)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [body, source, typed, ComputeKernel.surfaceBody]
  try simp only [eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
  simp [preExecution, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    evalOffsets, store, numeric, bop, natLt, numericEq, ComputeDType.eraseDType,
    ofFloat, toFloat, FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod]
  rfl

/-- Both branches use the source's exact dynamic 0/1 bound. -/
def execution {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (source p) s = some t} := by
  by_cases hb : blockActive p s
  · refine ⟨(bodyExecution M p s).val, ?_⟩
    change run M (source p).surfaceBody s = _
    rw [source_body, run_append]
    conv_lhs => rw [(preExecution M p s).property]
    simp only [Option.bind_some, run, step, evalOp_unfold, preExecution_mask,
      hb, ↓reduceIte, Option.bind_some, bind, pure, Option.bind, Option.pure_def]
    rw [FP.Structural.range]
    simp only [Nat.one_ne_zero, ↓reduceIte, Nat.zero_lt_one, Nat.zero_add]
    conv_lhs => rw [(bodyExecution M p s).property]
    simp only [Option.bind_some, bind, Option.bind]
    rw [FP.Structural.range]
    simp [run]
  · refine ⟨(preExecution M p s).val, ?_⟩
    change run M (source p).surfaceBody s = _
    rw [source_body, run_append]
    conv_lhs => rw [(preExecution M p s).property]
    simp only [Option.bind_some, run, step, evalOp_unfold, preExecution_mask,
      hb, ↓reduceIte, Option.bind_some, bind, pure, Option.bind, Option.pure_def]
    rw [FP.Structural.range]
    simp [run]

#axiomsClean bodyExecution
#axiomsClean execution
end
end VeriTile.Bench.TritonBenchG.TokenAttnLlama2.FPValue
