import bench.tritonbench_g.token_attn_llama2.Proofs.FPTypedSource
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def prelude (p : Parameters) := (source p).surfaceBody.take 13

def body (p : Parameters) : List ComputeStmt :=
  match (source p).surfaceBody.drop 13 with
  | [.forRangeDyn _ _ _ _ code] => code
  | _ => []

theorem source_body (p : Parameters) : (source p).surfaceBody =
    prelude p ++ [.forRangeDyn "start_mark" (.constNat 0) (.ref .nat [] "block_mask")
      (.constNat 1) (body p)] := by rfl

def preExecution {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // run M (prelude p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [prelude, source, typed, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, numeric, bop, natLt, numericEq, ComputeDType.eraseDType,
    ofFloat, toFloat, FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod]
  rfl

@[simp] theorem preExecution_mem {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    (preExecution M p s).val.mem = s.mem := by simp [preExecution]

@[simp] theorem preExecution_mask {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    (preExecution M p s).val.regs .nat [] "block_mask" =
      some (fun _ => if blockActive p s then 1 else 0) := by
  simp [preExecution, blockActive, length, Region.cast]
  rfl

#axiomsClean source_body
#axiomsClean preExecution
#axiomsClean preExecution_mask
end
end VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
