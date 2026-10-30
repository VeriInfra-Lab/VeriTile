import bench.tritonbench_g.lightning_attention.Proofs.IntraModel
import bench.tritonbench_g.lightning_attention.Proofs.ForwardBody
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 24000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
def intraExecution (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec (engine (withDot M D)) (intra p) s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [intra,ComputeKernel.surfaceBody,FP.Structural.exec,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store]
  rfl
theorem intraExecution_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (s : State α) :
    (intraExecution M D p s).val.mem=intraMemory M D p (s.pids 0) (s.pids 1) s.mem := by
  unfold intraMemory writeRows writeTrans scatter ChunkDeltaFwd.scatter
    intraDQ intraDKT intraDV intraDS intraST intraQ intraK intraV intraG
  simp [intraExecution,masked_scatter_memory,pair_eta,
    read,zero,add,cast,dot,FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    Nat.add_assoc]
  norm_num
theorem intraProgram (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec (engine (withDot M D)) (intra p) s=some t ∧
      t.mem=intraMemory M D p (s.pids 0) (s.pids 1) s.mem ∧ Inputs t.mem s.mem := by
  refine ⟨(intraExecution M D p s).val,(intraExecution M D p s).property,intraExecution_memory M D p s,?_⟩
  rw [intraExecution_memory]
  exact intraMemory_inputs _ _ _ _ _ _
#axiomsClean intraExecution_memory
#axiomsClean intraProgram
end
end VeriTile.Bench.Repairs.LightningAttention
