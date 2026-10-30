import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.PublicIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section

theorem logical_implements (c : Shape) (scale : ℝ) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw) :
    (logicalRealIO c scale fw pw bw).ImplementsOn (logicalZeroDQ c) (logicalFormula c scale) :=
  (logical_iff c scale fw pw bw).2 (real_implements c scale fw pw bw hf hp hb)

#axiomsClean logical_implements
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.PublicIO
