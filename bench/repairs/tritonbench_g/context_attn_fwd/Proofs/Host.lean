import bench.repairs.tritonbench_g.context_attn_fwd.Kernels
import bench.tritonbench_g.context_attn_fwd.Proofs.HostParameters

namespace VeriTile.Bench.Repairs.ContextAttnFwd.Host
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula
noncomputable section
abbrev Original := VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.Arguments

/-- Same-call host descriptors and actual observed INT32 metadata. -/
structure Call where
  args : Original
  regions : Regions
  start : Nat → Nat
  length : Nat → Nat
  prompt : Nat → Nat
  headsPositive : 0 < args.q.heads

abbrev Call.layout (c : Call) : Layout := VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.layout c.args
abbrev Call.grid (c : Call) : Grid := VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.grid c.args
abbrev Call.batches (c : Call) : Nat := c.args.batch
abbrev Call.maxInput (c : Call) : Nat := c.args.maxInput
abbrev scale (c : Call) : ℝ := VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.scale c.args
abbrev Call.blocks (c : Call) : Nat := VeriTile.Bench.TritonBenchG.ContextAttnFwd.Bounds.ceilBlocks c.maxInput c.layout.BM

def Call.launchAt (c : Call) (idx : GridIndex c.grid) : Launch := ⟨idx.toPids 0,idx.toPids 1⟩
def Call.active (c : Call) (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) : Prop :=
  queryRow c.layout (c.launchAt idx) i.1.val < c.length (batch c.layout (c.launchAt idx))
def Call.output (c : Call) (b h row d : Nat) : Nat :=
  (c.start b+row)*c.layout.outRow+h*c.layout.outHead+d*c.layout.outDim

def Call.offset (c : Call) (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) : Nat :=
  c.output (batch c.layout (c.launchAt idx)) (head c.layout (c.launchAt idx))
    (queryRow c.layout (c.launchAt idx) i.1.val) i.2.1.val

theorem bm_positive (c : Call) : 0<c.layout.BM :=
  VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.block_positive c.args

theorem Call.blockN (c : Call) : c.layout.BN=c.layout.BM := rfl

theorem batch_bound (c : Call) (idx : GridIndex c.grid) :
    batch c.layout (c.launchAt idx)<c.batches := by
  apply (Nat.div_lt_iff_lt_mul c.headsPositive).mpr
  exact (idx ⟨1,by simp [Call.grid,VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.grid,Grid.rank]⟩).isLt

theorem head_bound (c : Call) (idx : GridIndex c.grid) :
    head c.layout (c.launchAt idx)<c.layout.heads := Nat.mod_lt _ c.headsPositive
end
end VeriTile.Bench.Repairs.ContextAttnFwd.Host
