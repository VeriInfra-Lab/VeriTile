import bench.tritonbench_g.rbe_triton_transform.Proofs.Bridge
import bench.tritonbench_g.rbe_triton_transform.Kernels

namespace VeriTile.Bench.Repairs.RbeTritonTransform.Selected
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- The selected execution is the shared repaired source, with its documented
mathematical index projection. No second complete kernel is maintained here. -/
abbrev typed := VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.typed

def source (p : Source.Parameters) : ComputeKernel :=
  typed .fp32 p.input p.output p.rows p.columns p.inputBatchStride p.inputRowStride
    p.inputColumnStride p.outputBatchStride p.outputRowStride p.outputColumnStride
    p.startToken p.theta p.frequencyDimension p.blockRows p.blockColumns

def carrierTypes (ss : List ComputeStmt) : List (RegName × AlgDType) :=
  ss.filterMap fun statement =>
    let binding := match statement with
      | .alg (.assign dtype _ name _) => some (name,dtype)
      | .assign dtype _ name _ => some (name,dtype)
      | _ => none
    binding.filter fun entry => ["M","K","start_token_position","pid_batch","pid",
      "pid_m","pid_n","offs_m","offs_n","start_block","freqs","tks"].contains entry.1

theorem carrier_types (p : Source.Parameters) :
    carrierTypes (source p).surfaceBody=
      [("M",.nat),("K",.nat),("start_token_position",.int),("pid_batch",.nat),
       ("pid",.nat),("pid_m",.nat),("pid_n",.nat),("offs_m",.nat),("offs_n",.nat),
       ("start_block",.int),("freqs",.nat),("tks",.int)] := by rfl

def storeTypes (ss : List ComputeStmt) : List AlgDType :=
  ss.filterMap fun statement => match statement with
    | .store dtype _ _ _ _ => some dtype
    | _ => none

theorem store_types (p : Source.Parameters) :
    storeTypes (source p).surfaceBody=[.fp32,.fp32] := by rfl

def execution {α : Type} [Inhabited α] (M : Algebra α) (p : Source.Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (source p) s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [source,typed,VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.typed,FP.Structural.exec,ComputeKernel.surfaceBody,FP.Structural.run,
    FP.Structural.step,FP.Structural.evalExpr,FP.Structural.evalComputeOp,
    FP.Structural.evalOp_unfold,FP.Structural.evalOffsets,FP.Structural.store,
    FP.Structural.numeric,FP.Structural.bop,FP.Structural.natLt,FP.Structural.numericLt,FP.Structural.numericEq,
    ComputeDType.eraseDType,FP.Structural.ofFloat,FP.Structural.toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransform.Selected
