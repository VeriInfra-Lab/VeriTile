import bench.tritonbench_g.context_attn_mistral.Proofs.Program
import VeriTile.Triton.Launch.Grid

namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.Host
open VeriTile.Triton Formula Program
noncomputable section

/-- FP32 launch-argument values of Python `1.0 / (D ** 0.5)` on the
asserted channel widths. The dyadics are checked against Python in the host
regression; no ideal reciprocal square root is substituted for the argument. -/
def scale (d : Nat) : ℝ :=
  if d = 16 then 1/4 else if d = 32 then 11863283/67108864 else
  if d = 64 then 1/8 else 11863283/134217728

/-- Actual stride arguments remain independent. Metadata arrays use the
source's raw contiguous int32 ABI. The caller supplies and owns `out`. -/
structure Call where
  layout : Layout
  regions : Regions
  batches : Nat
  heads : Nat
  maxInput : Nat
  kvHeads : Nat
  start : Nat → Nat
  length : Nat → Nat
  blockM : layout.BM = 128
  blockN : layout.BN = 128
  channels : layout.D ∈ [16,32,64,128]

def Call.blocks (c : Call) : Nat := (c.maxInput + 127) / 128
def Call.grid (c : Call) : Grid := ⟨[c.batches,c.heads,c.blocks]⟩
def Call.launchAt (c : Call) (idx : GridIndex c.grid) : Launch :=
  ⟨idx.toPids 2,idx.toPids 0,idx.toPids 1⟩
def Call.output (c : Call) (b h row d : Nat) : Nat :=
  (c.start b+row)*c.layout.outRow+h*c.layout.outHead+d*c.layout.outDim
end
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.Host
