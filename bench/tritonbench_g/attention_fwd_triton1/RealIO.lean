import bench.tritonbench_g.attention_fwd_triton1.IO
import bench.tritonbench_g.attention_fwd_triton1.Proofs.RealOutput

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.RealIO
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section

/-- Independent full attention sum and prefix-history matrix. -/
def formula (c : Shape) (values : KernelIO.Values ℝ c) : ChannelValues ℝ (KernelIO.outputs c)
  | ⟨0,_⟩,i=>MathTarget.output c (KernelIO.logicalInputs c values) (KernelIO.realScale c)
      (KernelIO.mainHead c i.1.val) (KernelIO.mainRow c i.1.val) (KernelIO.mainColumn c i.1.val)
  | ⟨1,_⟩,i=>MathTarget.history c (KernelIO.logicalInputs c values)
      (KernelIO.historyHead c i.1.val) (KernelIO.historyChunk c i.1.val)
      (KernelIO.historyRow c i.1.val) (KernelIO.historyColumn c i.1.val)

#specNonCircular formula avoiding [FP.Structural.exec,FP.Structural.run,SharedGrid.result,Reference.output,Reference.carry]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.RealIO
