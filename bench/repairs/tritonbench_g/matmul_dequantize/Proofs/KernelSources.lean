import bench.repairs.tritonbench_g.matmul_dequantize.Kernels
import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Host

namespace VeriTile.Bench.Repairs.MatmulDequantize.KernelSources
open VeriTile.Triton
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
noncomputable section

/-- Exact source identities connect both Real/native grid executions to the
single readable quotation exposed by the shared Kernels.lean facade. -/
theorem gptq (p : Host.First.Parameters) :
    VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.source p =
      KernelSource.gptq p.input p.output p.scale p.weight p.zero
        p.rows p.columns p.inner p.inputRowStride p.inputInnerStride
        p.weightInnerStride p.weightColumnStride p.outputRowStride p.outputColumnStride
        p.scaleGroupStride p.scaleColumnStride p.zeroGroupStride p.zeroColumnStride
        p.groupSize p.noGroups p.blockRows p.blockColumns p.blockInner p.groupRows := rfl

theorem s2 (p : FPTypedSecond.Parameters) : FPTypedSecond.source p =
    KernelSource.s2 p.input p.output p.scale p.weight p.zero
      p.rows p.columns p.inner p.inputRowStride p.inputInnerStride
      p.weightInnerStride p.weightColumnStride p.outputRowStride p.outputColumnStride
      p.scaleGroupStride p.scaleColumnStride p.zeroGroupStride p.zeroColumnStride
      p.groupSize p.blockRows p.blockColumns p.blockInner p.groupRows p.steps := rfl

theorem dequant16 (p : Host.D16.Parameters) :
    VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.source p =
      KernelSource.dequant16 p.weight p.scale p.zero p.output
        p.rows p.columns p.groupSize p.weightRowStride p.weightColumnStride
        p.scaleRowStride p.scaleColumnStride p.zeroRowStride p.zeroColumnStride
        p.outputRowStride p.outputColumnStride p.blockRows p.blockColumns := rfl

theorem dequant32 (p : Dequant32.Parameters) : Dequant32.source p =
    KernelSource.dequant32 p.weight p.scale p.zero p.output
      p.rows p.columns p.groupSize p.weightRowStride p.weightColumnStride
      p.scaleRowStride p.scaleColumnStride p.zeroRowStride p.zeroColumnStride
      p.outputRowStride p.outputColumnStride p.blockRows p.blockColumns := rfl

#axiomsClean gptq
#axiomsClean s2
#axiomsClean dequant16
#axiomsClean dequant32
end
end VeriTile.Bench.Repairs.MatmulDequantize.KernelSources
