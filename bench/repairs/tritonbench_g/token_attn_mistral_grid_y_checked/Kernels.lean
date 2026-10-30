import bench.repairs.tritonbench_g.token_attn_mistral.Source

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked
open VeriTile.Triton TokenAttnMistral
abbrev source := TokenAttnMistral.source
abbrev kernel := TokenAttnMistral.kernel

/-- Existing complete gathered-attention source with its uniform probability
address origin computed scalarly. This removes one vector addition from affine
address formation; all masks and floating computations remain in source order. -/
abbrev affineSource := TokenAttnMistral.affineSource

def affineKernel (p : Parameters) : ComputeKernel :=
  affineSource p.prob p.value p.output p.tokens p.request p.unusedStart p.length p.start p.attentionLength
    p.requestStride p.sequenceStride p.attentionStartStride p.attentionLengthStride
    p.requestBatchStride p.requestTokenStride p.probHeadStride p.probTokenStride
    p.valueBatchStride p.valueHeadStride p.valueDimStride
    p.outputBatchStride p.outputHeadStride p.outputDimStride p.kvGroup p.window p.width p.block

end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked
