import bench.tritonbench_g.context_attn_llama.Sources.Mathematical_Program
import VeriTile.Triton.Launch.Grid

namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama.Host
open VeriTile.Triton Formula Program
noncomputable section

/-- FP32 launch-argument values of Python `1 / sqrt(D) * 1.4426950408889634`.
The host regression checks all five asserted widths. -/
def scale (d : Nat) : ℝ :=
  if d = 16 then 12102203/33554432 else if d = 32 then 4278775/16777216 else
  if d = 64 then 12102203/67108864 else if d = 128 then 4278775/33554432 else
  12102203/134217728

/-- Actual stride arguments remain independent. Metadata arrays use the
source's raw contiguous int32 ABI. The caller supplies and owns `out`. -/
structure Call where
  layout : Layout
  regions : Regions
  batches : Nat
  maxInput : Nat
  kvHeads : Nat
  start : Nat → Nat
  length : Nat → Nat
  prompt : Nat → Nat
  request : Nat → Nat
  page : Nat → Nat → Nat
  headsPositive : 0 < layout.heads
  blockM : layout.BM = 64 ∨ layout.BM = 128
  blockN : layout.BN = layout.BM
  channels : layout.D ∈ [16,32,64,128,256]

def Call.blocks (c : Call) : Nat := (c.maxInput + c.layout.BM - 1) / c.layout.BM
def Call.grid (c : Call) : Grid := ⟨[c.blocks,c.batches*c.layout.heads,1]⟩
def Call.launchAt (c : Call) (idx : GridIndex c.grid) : Launch :=
  ⟨idx.toPids 0,idx.toPids 1⟩
def Call.output (c : Call) (b h row d : Nat) : Nat :=
  (c.start b+row)*c.layout.outRow+h*c.layout.outHead+d*c.layout.outDim
end
end VeriTile.Bench.TritonBenchG.ContextAttnLlama.Host
