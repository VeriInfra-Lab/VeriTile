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

end Q
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelSourceBridge
