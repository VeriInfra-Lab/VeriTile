import bench.repairs.tritonbench_g.iv_dependent_matmul.SourceDefinitions

/-! Finite repaired source only: scalar ABI, signed controls, modular pointers,
all five pointer schedules, masked loads/stores and the ordered whole grid.
Numerical interpretations and correctness arguments are under Proofs. -/
namespace VeriTile.Bench.Repairs.IvDependentMatmul
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Request := TritonBenchG.IvDependentMatmul.Whole.Request
abbrev Schedule := TritonBenchG.IvDependentMatmul.Kernels.Schedule
abbrev Graph := TritonBenchG.IvDependentMatmul.FPValue.Graph
abbrev source {α : Type} [Inhabited α] {r : Request} (m : Execution.Arithmetic α r) (g : Graph)
    (ba bb bc pid : Nat) (s : State α) := Execution.program m g ba bb bc pid s
abbrev host {α : Type} [Inhabited α] {r : Request} (m : Execution.Arithmetic α r) (g : Graph)
    (ba bb bc : Nat) (s : State α) := Execution.host m g ba bb bc s
end
end VeriTile.Bench.Repairs.IvDependentMatmul
