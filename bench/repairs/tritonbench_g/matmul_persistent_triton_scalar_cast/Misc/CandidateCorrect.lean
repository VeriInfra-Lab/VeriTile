import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.Pair
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Public

/-! Optional application of the main source pair through the retained
post-allocation Tensor wrapper and physical-memory materialization. -/
namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Pair
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open Core TensorBinding
noncomputable section
variable {α : Type} [Inhabited α]

def candidateInvoke (P : Source.Arithmetic α α) (p : Parameters) (args : Arguments)
    (initial : State α) (entry : Memory α) : Option (ActualTensor × Memory α) :=
  if p.problem.M=0 ∨ p.problem.N=0 then some (args.output,entry)
  else if p.problem.K=0 then
    some (args.output,Host.zeroMemory P.inputZero p.problem args.descriptors entry)
  else (runSchedule ((List.range p.problem.grid).map (job P p)) initial).map
    (fun final => (args.output,Physical.materialize p args.descriptors entry final))

/-- The actual allocated output object and every physical memory cell agree,
including both no-launch branches. This transports all existing physical
output/input-backing/complement frames without a new numerical premise. -/
theorem invoke_equal (P : Source.Arithmetic α α) (p : Parameters) (args : Arguments)
    (initial : State α) (entry : Memory α) :
    Public.invoke P p args initial entry=candidateInvoke P p args initial entry := by
  have grid : executeGrid P p initial=
      runSchedule ((List.range p.problem.grid).map (job P p)) initial := by
    unfold executeGrid
    rw [← PublicData.schedule_run]
    exact schedule_equal P p _ initial
  unfold Public.invoke candidateInvoke
  rw [grid]

#axiomsClean invoke_equal
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Pair
