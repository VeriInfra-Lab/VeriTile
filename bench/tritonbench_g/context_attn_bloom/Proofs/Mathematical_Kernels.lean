import bench.tritonbench_g.context_attn_bloom.Kernels

/-! Compatibility names for the shared repaired source and retained dtype
specializations. Complete bodies live only in Kernels.lean. -/
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom
open VeriTile.Triton
inductive Storage where
  | fp16 | bf16 | fp32
  deriving DecidableEq

def original (storage : Storage) := match storage with
  | .fp32 => VeriTile.Bench.Repairs.ContextAttnBloomSource.kernel
  | .fp16 => VeriTile.Bench.Repairs.ContextAttnBloomSource.storageProfile .fp16
  | .bf16 => VeriTile.Bench.Repairs.ContextAttnBloomSource.storageProfile .bf16

def scaleFirst (storage : Storage) := match storage with
  | .fp32 => VeriTile.Bench.Repairs.ContextAttnBloomSource.scaleFirst32
  | .fp16 => VeriTile.Bench.Repairs.ContextAttnBloomSource.storageScaleFirst .fp16
  | .bf16 => VeriTile.Bench.Repairs.ContextAttnBloomSource.storageScaleFirst .bf16

end VeriTile.Bench.TritonBenchG.ContextAttnBloom
