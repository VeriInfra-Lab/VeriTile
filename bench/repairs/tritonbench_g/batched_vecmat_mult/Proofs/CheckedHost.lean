import bench.repairs.tritonbench_g.batched_vecmat_mult.Contract
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.FiniteHost

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.CheckedHost
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
noncomputable section
local instance : Inhabited ValueProofs.Word := ⟨FP.Value.zero .fp32⟩

/-- Outer `none` represents the new ValueError before allocation; admitted
arguments retain every ordered allocation, remainder, and setup outcome. -/
def runGraph (allocate : ValueProofs.Host.Allocator) (setup : Whole.Setup)
    (rest : Algebra SourceGraph.Expr) (plans : SourceGraph.PlanSelector)
    (raw : Contract.Arguments) (initial : State ValueProofs.Word) :=
  (Contract.checkInputs raw).map fun a =>
    FiniteHost.runGraph allocate setup rest plans a initial

def runTensor (allocate : ValueProofs.Host.Allocator) (setup : Whole.Setup)
    (rest : Algebra SourceGraph.Expr) (cfg : FP.Config) (plans : SourceGraph.PlanSelector)
    (raw : Contract.Arguments) (initial : State ValueProofs.Word) :=
  (Contract.checkInputs raw).map fun a =>
    FiniteHost.runTensor allocate setup rest cfg plans a initial

@[simp] theorem graph_admitted (allocate setup rest plans raw initial)
    (h : Contract.InputContract raw) :
    runGraph allocate setup rest plans raw initial =
      some (FiniteHost.runGraph allocate setup rest plans (Contract.toArguments raw) initial) := by
  simp [runGraph,Contract.checkInputs,h]

@[simp] theorem tensor_admitted (allocate setup rest cfg plans raw initial)
    (h : Contract.InputContract raw) :
    runTensor allocate setup rest cfg plans raw initial =
      some (FiniteHost.runTensor allocate setup rest cfg plans (Contract.toArguments raw) initial) := by
  simp [runTensor,Contract.checkInputs,h]

theorem graph_rejected (allocate setup rest plans raw initial)
    (h : ¬ Contract.InputContract raw) :
    runGraph allocate setup rest plans raw initial = none := by
  simp [runGraph,Contract.checkInputs,h]

theorem tensor_rejected (allocate setup rest cfg plans raw initial)
    (h : ¬ Contract.InputContract raw) :
    runTensor allocate setup rest cfg plans raw initial = none := by
  simp [runTensor,Contract.checkInputs,h]

#axiomsClean graph_admitted
#axiomsClean tensor_admitted
#axiomsClean graph_rejected
#axiomsClean tensor_rejected
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.CheckedHost
