import bench.tritonbench_g.int8_matmul_quantization.Sources.Kernels
import bench.tritonbench_g.int8_matmul_quantization.Sources.Mathematical_QuantizeSource

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelSourceBridge
open VeriTile.Triton
namespace Q
abbrev Previous := TritonBenchG.Int8MatmulQuantization.FPQuantize.Parameters

def previous (p : KernelSource.QuantizeParameters) : Previous where
  input := p.input
  output := p.output
  scales := p.scales
  rows := p.rows
  columns := p.columns
  inputRowStride := p.inputRowStride
  inputColumnStride := p.inputColumnStride
  outputRowStride := p.outputRowStride
  outputColumnStride := p.outputColumnStride
  scaleStride := p.scaleStride
  blockRows := p.blockRows
  blockColumns := p.blockColumns
  steps := p.steps

/-- The complete producer quotation is exactly the source already used by
the old execution fragments. This permits direct proof reuse; it does not
import the old finite-conversion InputDefined premise into the new domain. -/
theorem source_identity (p : KernelSource.QuantizeParameters) :
    KernelSource.quantize p = TritonBenchG.Int8MatmulQuantization.FPQuantize.source (previous p) := rfl

#axiomsClean source_identity
end Q
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelSourceBridge
