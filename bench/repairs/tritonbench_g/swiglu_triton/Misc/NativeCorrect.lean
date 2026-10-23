import bench.repairs.tritonbench_g.swiglu_triton.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.SwigluTriton
open VeriTile.Triton FP.Structural Native
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term graphAlgebra)
noncomputable section

/-- The actual FP32 source with raw-word input leaves and the selected
sigmoid/FMA instruction observation. Branch masks name only accessed inputs
and enabled outputs; native and Real entries share the same row geometry. -/
def swigluNativeIO (instructions : InstructionSemantics) (backward : Bool)
    (block cols row : Nat) : ProgramIO Word (Term Word) where
  inputs := [⟨.real, [block], "A"⟩, ⟨.real, [block], "B"⟩, ⟨.real, [block], "DC"⟩]
  outputs := [⟨.real, [block], "C"⟩, ⟨.real, [block], "A"⟩, ⟨.real, [block], "B"⟩]
  scratch := []
  buffers := ["A", "B", "DC", "C"]
  read := PublicIO.inputRead block cols row
  inputMask := PublicIO.inputMask backward block cols
  write := PublicIO.outputWrite block cols row
  outputMask := PublicIO.outputMask backward block cols
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [{
    kernel := directSource backward block cols
    backend := graphAlgebra
    pids := fun ax => if ax = 0 then row else 0 }]
  encode := Term.input
  observe := fun _ _ => interpret (compiledPorts instructions)

/-- Source-to-native observation for all FP32 words. The independent ordered
reference retains the backward FMA and sigmoid lowering. Primitive meanings
are explicit; no floating transformation law or deferred experiment is used. -/
theorem swiglu_kernel_native (instructions : InstructionSemantics) (backward : Bool)
    (block cols row : Nat) (_covers : cols ≤ block) :
    (swigluNativeIO instructions backward block cols row).Implements
      (PublicIO.formula (compiledReference instructions) block) := by
  exact PublicIO.implements graphAlgebra Term.input (interpret (compiledPorts instructions))
    (compiledReference instructions) (KernelNative.scalar_lowering instructions)
    backward block cols row

#specNonCircular compiledReference avoiding [directSource, Native.interpret, Native.sourceScalar]
#axiomsClean swiglu_kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton
