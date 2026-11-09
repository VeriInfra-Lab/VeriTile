import bench.tritonbench_g.matmul_triton_autotune.Proofs.Sources.Mathematical_FPReference
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
theorem prelude_profiles (p : Parameters) (st : Storage) : prelude p st=prelude p .single := by
  cases st <;> rfl

def preludeExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) :
    {t : State α // run (backend M D) (prelude p st) s=some t} := by
  refine ⟨?_, ?_⟩
  swap
  rw [prelude_profiles]
  simp [prelude,source,typed,Pasted.originalKernel,dtype,ComputeKernel.surfaceBody,run,step,evalExpr,
       evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
       FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod]
  rfl


/-- Exact signed quotient emitted for the source's cdiv expression. -/
def loopStop (p : Parameters) : Int :=
  if (p.K : Int)+((BK p : Int)-1)<0 then -((1-(BK p : Int)+ -(p.K : Int))/(BK p : Int))
  else ((p.K : Int)+((BK p : Int)-1))/(BK p : Int)

theorem loop_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (st : Storage) (s : State α) :
    run M [loopStatement p st] s=rangeInt M "k" 0 (loopStop p) 1 (body p st) s := by
  cases st <;>
    simp [loopStatement,body,source,typed,Pasted.originalKernel,dtype,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,integralFloorDiv,numeric,
      numericLt,bop,Broadcast.leftIndex,Broadcast.rightIndex,loopStop]
  all_goals exact Option.bind_fun_some _

#axiomsClean loop_run

#axiomsClean preludeExecution
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.FPValue
