import bench.repairs.tritonbench_g.token_attn_reduceV.Source

namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked
open VeriTile.Triton TokenAttnReduceV
abbrev source := TokenAttnReduceV.source
abbrev kernel := TokenAttnReduceV.kernel

/-- Full affine-address candidate: compute uniform P origin scalarly,
removing one vector addition, with every FP32 operation and mask retained. -/
abbrev affineSource := TokenAttnReduceV.affineSource

def affineKernel (p : Parameters) : ComputeKernel :=
  affineSource p.prob p.value p.output p.tokens p.request p.start p.length
    p.requestStride p.startStride p.lengthStride
    p.requestBatchStride p.requestTokenStride p.probHeadStride p.probTokenStride
    p.valueBatchStride p.valueHeadStride p.valueDimStride
    p.outputBatchStride p.outputHeadStride p.outputDimStride p.kvGroup p.width p.block

end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked
