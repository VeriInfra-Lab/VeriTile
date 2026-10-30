import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Kernels
import bench.repairs.tritonbench_g.fused_rotary_embedding.Kernels
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_regs)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def algebra {α : Type} (rest : Algebra α) (p : Ops α) : Algebra α :=
  { rest with
    fp32Load:=id
    cast:=fun _ _ _=>id
    binary:=fun tag dtype op=>match op with
      | .mul=>p.mul | .sub=>p.sub | .add=>p.add | _=>rest.binary tag dtype op }

/-- Constructive execution of every statement in the shared quotation, with
the actual Q stores before the conditional K/V loads and cache writes. -/
def execution {α : Type} [Inhabited α] (rest : Algebra α) (ops : Ops α)
    (p : Parameters) (s : State α) :
    {final : State α // FP.Structural.exec (algebra rest ops) (kernel p) s=some final} := by
  have headNonneg : ¬(s.pids 0 : Int)<0 := by omega
  have signedLeader :
      ((s.pids 0 : Int)-(p.KV_GROUP_NUM : Int)*((s.pids 0 : Int)/(p.KV_GROUP_NUM : Int))=0) ↔
        s.pids 0%p.KV_GROUP_NUM=0 := by
    rw [←Int.emod_def,←Int.natCast_emod,Int.natCast_eq_zero]
  by_cases leader : s.pids 0%p.KV_GROUP_NUM=0
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp (maxSteps := 1000000) [kernel,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
      evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,Option.bind,
      algebra,numericLt,natLt,numericEq,bop,numeric,store,VeriTile.Triton.Region.cast,
      ComputeDType.eraseDType,ofFloat,toFloat,integralFloorDiv,integralMod,
      headNonneg,signedLeader,leader]
    rfl

#axiomsClean execution
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
