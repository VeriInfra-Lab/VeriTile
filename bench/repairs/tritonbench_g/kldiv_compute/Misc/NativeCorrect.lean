import bench.repairs.tritonbench_g.kldiv_compute.Proofs.PublicIO
import bench.repairs.tritonbench_g.kldiv_compute.Proofs.Native

namespace VeriTile.Bench.Repairs.KldivCompute.Kernel.NativeCorrect
open VeriTile.Triton FP.Structural
open Kernel.FPEquiv
open VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs (gridCount)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Full source grid with all FP32 logical words admitted. The primitive
model supplies divide/log/multiply; no numerical rewrite or experiment is used. -/
def kldivNativeIO (ops : Primitives) (p : Parameters) : ProgramIO Word Word where
  inputs := [⟨.real, [p.n], p.x⟩, ⟨.real, [p.n], p.y⟩]
  outputs := [⟨.real, [p.n], p.output⟩]
  scratch := []
  buffers := [p.x, p.y, p.output]
  read := PublicIO.read p
  inputMask := fun _ _ _ => True
  write := PublicIO.write p
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => (List.range (gridCount p)).map fun pid => {
    kernel := KernelDSL.kernel p
    backend := algebra ops
    clearRegisters := Bool.true
    pids := fun ax => if ax = 0 then pid else 0
    numPids := some (fun ax => if ax = 0 then gridCount p else 1) }
  encode := id
  observe := fun _ _ => id

/-- Independent source-ordered divide/log/multiply observation, all outputs,
actual source access safety, and the complete exterior typed-cell frame. -/
theorem native_observation (ops : Primitives) (p : Parameters) (positive : 0 < p.block) :
    (kldivNativeIO ops p).Implements (PublicIO.formula (reference ops) p) := by
  exact PublicIO.implements (algebra ops) p positive (reference ops) (scalar_reference ops)

#specNonCircular reference avoiding [KernelDSL.kernel, Program.scalar, Program.grid]
#specNonCircular Primitives avoiding [KernelDSL.kernel, reference, Program.grid]
#axiomsClean native_observation
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.Kernel.NativeCorrect
