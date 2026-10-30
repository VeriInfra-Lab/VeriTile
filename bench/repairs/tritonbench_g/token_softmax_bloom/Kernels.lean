import bench.repairs.tritonbench_g.token_softmax_llama.Kernels

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom
open VeriTile.Triton
/-- Both selected Python bodies are identical; their manifests check them
independently against this shared authoritative raw quotation. -/
abbrev source := TokenSoftmaxLlama.Strided.source
abbrev indexCommutedSource := TokenSoftmaxLlama.KernelSource.indexCommutedSource
end VeriTile.Bench.Repairs.TokenSoftmaxBloom
