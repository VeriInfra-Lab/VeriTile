import VeriTile.Triton.Source
import bench.tritonbench_g.context_attn_fwd.Proofs.Sources.Selected

/-! Storage compatibility views instantiate the same copied bodies. -/
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd
open VeriTile.Triton
noncomputable section
def original (storage : Storage) := match storage with
  | .fp16 => VeriTile.Bench.Repairs.ContextAttnFwd.Bodies.originalBody .fp16
  | .bf16 => VeriTile.Bench.Repairs.ContextAttnFwd.Bodies.originalBody .bf16
  | .fp32 => VeriTile.Bench.Repairs.ContextAttnFwd.SelectedSource.original
def scaleFirst (storage : Storage) := match storage with
  | .fp16 => VeriTile.Bench.Repairs.ContextAttnFwd.Bodies.scaleFirstBody .fp16
  | .bf16 => VeriTile.Bench.Repairs.ContextAttnFwd.Bodies.scaleFirstBody .bf16
  | .fp32 => VeriTile.Bench.Repairs.ContextAttnFwd.SelectedSource.scaleFirst
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd
