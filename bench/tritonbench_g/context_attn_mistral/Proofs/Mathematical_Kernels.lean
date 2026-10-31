import bench.tritonbench_g.context_attn_mistral.Kernels

/-! Compatibility names for the shared repaired source and retained dtype
specializations. Complete bodies live only in Kernels.lean. -/
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral
open VeriTile.Triton
inductive Storage where
  | fp16 | bf16 | fp32
  deriving DecidableEq

def original (storage : Storage) := match storage with
  | .fp32 => VeriTile.Bench.Repairs.ContextAttnMistralKernel.kernel
  | .fp16 => VeriTile.Bench.Repairs.ContextAttnMistralKernel.storageProfile .fp16
  | .bf16 => VeriTile.Bench.Repairs.ContextAttnMistralKernel.storageProfile .bf16

def scaleFirst (storage : Storage) := match storage with
  | .fp32 => VeriTile.Bench.Repairs.ContextAttnMistralKernel.scaleFirst32
  | .fp16 => VeriTile.Bench.Repairs.ContextAttnMistralKernel.storageScaleFirst .fp16
  | .bf16 => VeriTile.Bench.Repairs.ContextAttnMistralKernel.storageScaleFirst .bf16

end VeriTile.Bench.TritonBenchG.ContextAttnMistral
