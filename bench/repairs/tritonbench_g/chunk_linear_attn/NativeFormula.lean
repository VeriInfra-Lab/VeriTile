import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.IOArray

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.NativeFormula
open VeriTile Triton FP.Structural
noncomputable section

/-- Finite logical FP32 inputs extended only to evaluate the independent
arithmetic reference. Unobserved memory is not constrained by this definition. -/
def input (d : Host.Dimensions) (vals : ChannelValues (BitVec 32) (IOLayout.inputs d))
    (r : RegionName) (offset : Nat) : BitVec 32 :=
  if r="query" then IOArray.extend [d.B*d.H,d.T,d.K] (vals 0) offset
  else if r="key" then IOArray.extend [d.B*d.H,d.T,d.K] (vals 1) offset
  else if r="value" then IOArray.extend [d.B*d.H,d.T,d.V] (vals 2) offset
  else if r="incoming" then IOArray.extend [d.B*d.H,d.T,d.V] (vals 3) offset
  else if r="initial" then IOArray.extend [d.B*d.H,d.K,d.V] (vals 4) offset
  else if r="terminalGradient" then IOArray.extend [d.B*d.H,d.K,d.V] (vals 5) offset
  else 0

/-- All six returned outputs and all three written intermediates, using the
existing independent coordinate tables, FP32/FMA profile and runtime scale word. -/
def native (profile : NativeTerms.FP32Profile) (c : KernelInterface.Config)
    (valid : Host.Valid c.dims) (scale : BitVec 32)
    (vals : ChannelValues (BitVec 32) (IOLayout.inputs c.dims)) :
    ChannelValues (BitVec 32) (IOLayout.nativeOutputs c.dims) :=
  fun o j => by
    have dtype : (IOLayout.nativeOutputs c.dims)[o].dtype = .real := by
      fin_cases o <;> rfl
    rw [dtype]
    exact KernelExecution.nativeTarget profile c.call valid scale (input c.dims vals)
      (IOLayout.nativeOutputs c.dims)[o].buf
      (TileShape.linearIndex (IOLayout.nativeOutputs c.dims)[o].shape j)

#specNonCircular native avoiding [KernelExecution.run,KernelExecution.runNative,
  FP.Structural.exec,runGrid,executeFour,gridMemory,stageMemory,modelFour,completeMemory]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.NativeFormula
