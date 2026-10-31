import bench.tritonbench_g.triton_matmul.Proofs.SourceBodyExecution
import bench.tritonbench_g.triton_matmul.Proofs.SourceOperands
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 1800000
set_option linter.unusedSimpArgs false

theorem bodyContext {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α) (k : Nat) (a : CIndex p→α)
    (ctx : Context p initial current) (ha : HasAcc p a current) :
    let t:=(bodyExecution M D p st initial current k a ctx ha).val
    Context p initial t ∧ HasAcc p (update M D p st initial k a) t := by
  rcases ctx.metadata with ⟨hm,hn,hk,ham,hak,hbk,hbn,hcm,hcn⟩
  cases st <;> constructor
  all_goals try (constructor <;> simp [bodyExecution,Metadata,hm,hn,hk,ham,hak,hbk,hbn,hcm,hcn,
    ctx.memory,ctx.metadata,ctx.offk,ctx.am,ctx.bn,ctx.mm,ctx.nn,ctx.blocks,
    bop,Broadcast.leftIndex,Broadcast.rightIndex])
  all_goals simp [HasAcc,bodyExecution,update,dtype,backend,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals apply congrArg₂ (fun (av : AIndex p→α) (bv : BIndex p→α)=>
    D.apply (some .fp32) _ _ .real {} (batch := []) av bv a)
  all_goals funext i
  · simpa only [VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue.read,dtype,BK] using aValue_signed M p .half initial k i
  · simpa only [VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue.read,dtype,BK] using bValue_signed M p .half initial k i
  · simpa only [VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue.read,dtype,BK] using aValue_signed M p .e4m3 initial k i
  · simpa only [VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue.read,dtype,BK] using bValue_signed M p .e4m3 initial k i

#axiomsClean bodyContext
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
