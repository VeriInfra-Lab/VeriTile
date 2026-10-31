import bench.tritonbench_g.vector_addition.Proofs.Mathematical_TypedHostValues

/-! Public source-host outcome contract on the complete seventeen-by-seventeen
dtype domain. Success comes from actual existing grid executions. Its value
domain is explicit; dtype admission does not imply native value-totality. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.TypedHostCorrect
open VeriTile.Triton DTypeScope TypedHost
open HostWrapper (Tensor Result Run Event Failure parameters successTrace cudaGuard add)
noncomputable section
set_option maxHeartbeats 6000000
local instance : Inhabited TypedHost.Word := ⟨FP.Value.zero .fp32⟩

def rejectedTrace (x y output : Tensor) : List Event :=
  [.emptyLike x.region, .cudaCheck x.isCUDA y.isCUDA output.isCUDA,
   .numel output.region output.numel,
   .launch output.numel 1024 (ValueProofs.gridCount (parameters x y output))]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.TypedHostCorrect
