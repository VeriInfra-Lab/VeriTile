import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Model
namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def preludeExecution (P : Source.Arithmetic α α) (p : Parameters) (s : State α) :
    {t : State α // run (backend P) (prelude p) s=some t} := by
  by_cases hm : p.problem.M%128=0 <;>
    by_cases hn : p.problem.N%256=0 <;>
    by_cases hk : p.problem.K%p.problem.profile.blockK=0 <;>
    by_cases hs : (s.pids 0 : Int) < ((ceil p.problem.M 128 * ceil p.problem.N 256)%p.problem.SMS : Nat)
  all_goals
    simp [ceil,hm,hn] at hs
    refine ⟨?_,?_⟩
    swap
    rw [prelude_raw]
    simp [rawPrelude,TypedSource.kernel,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,numericLt,ComputeDType.eraseDType,
      ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
      integralFloorDiv,integralMod,backend,hm,hn,hk,hs,Broadcast.leftIndex,Broadcast.rightIndex]
    first
    | rfl
    | simp only [if_neg (not_lt_of_ge hs)]
      simp [Option.bind,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
      rfl
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
