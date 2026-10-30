import bench.tritonbench_g.chunk_gate_recurrence.Proofs.KernelProgram

/-! Construct successful execution of both complete initialization prefixes.
The witnesses are interpreter states built from the entry memory; no register,
output equation, or successful source execution is an assumption. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelPrefix
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def forward (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool) (u : State α) :
    {v : State α // run (engine M) (before .forward c s seed) u=some v} := by
  cases seed
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [before,parts,source,Kernels.forward,ComputeKernel.surfaceBody,
      VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
      beq_self_eq_true,↓reduceIte]
    simp (config := {maxSteps:=150000}) [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      FP.Structural.store,integralFloorDiv,integralMod]
    try rfl

def backward (M : Algebra α) (c : Shape) (s : SeedStrides) (seed : Bool) (u : State α) :
    {v : State α // run (engine M) (before .backward c s seed) u=some v} := by
  refine ⟨?_,?_⟩
  swap
  simp only [before,parts,source,Kernels.backward,
    TritonBenchG.ChunkGateRecurrence.TypedSource.backward,
    TritonBenchG.ChunkGateRecurrence.TypedSource.specializeBackward,
    TritonBenchG.ChunkGateRecurrence.TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,
    String.reduceEq,beq_self_eq_true,↓reduceIte]
  simp (config := {maxSteps:=150000}) [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    FP.Structural.store,integralFloorDiv,integralMod]
  try rfl

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelPrefix
