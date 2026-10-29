import bench.tritonbench_g.triton_attention.Proofs.FPBackwardInvariant
namespace VeriTile.Bench.TritonBenchG.TritonAttention.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
def backwardPrologueResult (p : BackwardParameters) (s : State α) : State α :=
  let t := s.setReg "off_hz" .nat [] (fun _ => s.pids 0)
  let t := t.setReg "off_z" .nat [] (fun _ => s.pids 0/p.heads)
  let t := t.setReg "off_h" .nat [] (fun _ => s.pids 0%p.heads)
  let t := t.setReg "stride_qz_2d" .nat [] (fun _ => p.batchStride/p.head)
  let t := t.setReg "stride_qh_2d" .nat [] (fun _ => p.headStride/p.head)
  let t := t.setReg "k_tile_ptr" .blockPtr [p.blockM,p.head] (fun _ => backwardPointer p s p.k 0)
  let t := t.setReg "v_tile_ptr" .blockPtr [p.blockM,p.head] (fun _ => backwardPointer p s p.v 0)
  let t := t.setReg "dk_tile_ptr" .blockPtr [p.blockM,p.head] (fun _ => backwardPointer p s p.dk 0)
  t.setReg "dv_tile_ptr" .blockPtr [p.blockM,p.head] (fun _ => backwardPointer p s p.dv 0)

theorem backwardPrologueExecution (rest : Algebra α) (P : Primitives α) (p : BackwardParameters) (s : State α) :
    run (backend rest P) (backwardPrologue p) s=some (backwardPrologueResult p s) := by
  simp [backwardPrologue,backwardSource,backwardTyped,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    FP.Structural.evalOp,numeric,evalOffsets,backend,backwardPrologueResult,
    backwardPointer,backwardHeadRow,integralFloorDiv,integralMod,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind]
  delta FP.Structural.bop
  simp only [Broadcast.leftIndex,Broadcast.rightIndex,Nat.div_one]

theorem backwardPrologueInvariant (p : BackwardParameters) (s : State α) :
    BackwardOuterInvariant p s 0 s.mem (backwardPrologueResult p s) := by
  constructor
  · constructor <;> simp [backwardPrologueResult]
  all_goals simp [backwardPrologueResult]

#axiomsClean backwardPrologueExecution
#axiomsClean backwardPrologueInvariant
end
end VeriTile.Bench.TritonBenchG.TritonAttention.FPValue
