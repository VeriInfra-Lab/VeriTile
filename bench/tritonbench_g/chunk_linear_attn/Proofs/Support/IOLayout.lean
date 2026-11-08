import bench.tritonbench_g.chunk_linear_attn.Proofs.KernelInterface
import VeriTile.Triton.Float.MemoryIOJudgments

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.IOLayout
open VeriTile Triton FP.Structural
noncomputable section

@[reducible] def inputs (d : Host.Dimensions) : List IOChannel :=
  [⟨.real,[d.B*d.H,d.T,d.K],"query"⟩,
   ⟨.real,[d.B*d.H,d.T,d.K],"key"⟩,
   ⟨.real,[d.B*d.H,d.T,d.V],"value"⟩,
   ⟨.real,[d.B*d.H,d.T,d.V],"incoming"⟩,
   ⟨.real,[d.B*d.H,d.K,d.V],"initial"⟩,
   ⟨.real,[d.B*d.H,d.K,d.V],"terminalGradient"⟩]

@[reducible] def outputs (d : Host.Dimensions) : List IOChannel :=
  [⟨.real,[d.B*d.H,d.T,d.V],"output"⟩,
   ⟨.real,[d.B*d.H,d.T,d.K],"queryGradient"⟩,
   ⟨.real,[d.B*d.H,d.T,d.K],"keyGradient"⟩,
   ⟨.real,[d.B*d.H,d.T,d.V],"valueGradient"⟩,
   ⟨.real,[d.B*d.H,d.K,d.V],"final"⟩,
   ⟨.real,[d.B*d.H,d.K,d.V],"initialGradient"⟩]

@[reducible] def scratch (d : Host.Dimensions) : List IOChannel :=
  [⟨.real,[d.B*d.H,d.NT,d.K,d.V],"history"⟩,
   ⟨.real,[d.B*d.H,d.NT,d.K,d.V],"historyGradient"⟩,
   ⟨.real,[keyBlocks (d.parameters Bool.true),d.B*d.H,d.T,d.V],"valuePartials"⟩]

/-- The old native result observes the complete resulting memory, including
all three written intermediates. They remain explicit observable channels. -/
@[reducible] def nativeOutputs (d : Host.Dimensions) : List IOChannel := outputs d ++ scratch d

def buffers : List RegionName := ["query","key","value","incoming","initial","terminalGradient",
  "output","queryGradient","keyGradient","valueGradient","final","initialGradient",
  "history","historyGradient","valuePartials"]

/-- Contiguous logical layouts, the two optional input branches and two
optional endpoint outputs; private intermediates have their full extents. -/
@[reducible] def realMemory (α : Type) (c : KernelInterface.Config) : MemoryIO α where
  inputs := inputs c.dims
  outputs := outputs c.dims
  scratch := scratch c.dims
  buffers := buffers
  read := fun i _ j => TileShape.linearIndex (inputs c.dims)[i].shape j
  write := fun o _ j => TileShape.linearIndex (outputs c.dims)[o].shape j
  scratchWrite := fun k _ j => TileShape.linearIndex (scratch c.dims)[k].shape j
  inputMask := fun i _ _ => (i.val=4 → c.seed=Bool.true) ∧ (i.val=5 → c.terminal=Bool.true)
  outputMask := fun o _ _ => (o.val=4 → c.final=Bool.true) ∧ (o.val=5 → c.seed=Bool.true)
  scratchMask := fun _ _ _ => True

@[reducible] def nativeMemory (α : Type) (c : KernelInterface.Config) : MemoryIO α :=
  { realMemory α c with
    outputs := nativeOutputs c.dims
    write := fun o _ j => TileShape.linearIndex (nativeOutputs c.dims)[o].shape j
    outputMask := fun o _ _ => (o.val=4 → c.final=Bool.true) ∧ (o.val=5 → c.seed=Bool.true) }

end
end VeriTile.Bench.Repairs.ChunkLinearAttn.IOLayout
