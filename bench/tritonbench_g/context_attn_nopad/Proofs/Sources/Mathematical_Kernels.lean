import VeriTile.Triton.Source
import bench.tritonbench_g.context_attn_nopad.Proofs.Sources.Selected

/-! Storage compatibility views instantiate the same copied bodies. -/
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad
open VeriTile.Triton
noncomputable section
def original (storage : Storage) := match storage with
  | .fp16 => VeriTile.Bench.Repairs.ContextAttnNopad.Bodies.originalBody .fp16
  | .bf16 => VeriTile.Bench.Repairs.ContextAttnNopad.Bodies.originalBody .bf16
  | .fp32 => VeriTile.Bench.Repairs.ContextAttnNopad.SelectedSource.original
def scaleFirst (storage : Storage) := match storage with
  | .fp16 => VeriTile.Bench.Repairs.ContextAttnNopad.Bodies.scaleFirstBody .fp16
  | .bf16 => VeriTile.Bench.Repairs.ContextAttnNopad.Bodies.scaleFirstBody .bf16
  | .fp32 => VeriTile.Bench.Repairs.ContextAttnNopad.SelectedSource.scaleFirst
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad
