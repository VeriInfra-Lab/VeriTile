import VeriTile.Triton.Source
import bench.tritonbench_g.context_attn_llama.Proofs.Sources.Selected

/-! Storage compatibility views instantiate the same copied bodies. -/
namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama
open VeriTile.Triton
noncomputable section
def original (storage : Storage) := match storage with
  | .fp16 => VeriTile.Bench.Repairs.ContextAttnLlama.Bodies.originalBody .fp16
  | .bf16 => VeriTile.Bench.Repairs.ContextAttnLlama.Bodies.originalBody .bf16
  | .fp32 => VeriTile.Bench.Repairs.ContextAttnLlama.SelectedSource.original
def scaleFirst (storage : Storage) := match storage with
  | .fp16 => VeriTile.Bench.Repairs.ContextAttnLlama.Bodies.scaleFirstBody .fp16
  | .bf16 => VeriTile.Bench.Repairs.ContextAttnLlama.Bodies.scaleFirstBody .bf16
  | .fp32 => VeriTile.Bench.Repairs.ContextAttnLlama.SelectedSource.scaleFirst
end
end VeriTile.Bench.TritonBenchG.ContextAttnLlama
