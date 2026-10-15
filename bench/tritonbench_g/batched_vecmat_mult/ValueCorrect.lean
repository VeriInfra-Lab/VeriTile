import bench.tritonbench_g.batched_vecmat_mult.Proofs.ValueComplete

/-! Complete original fp32 software-value contract. The original source
executes for all natural model parameters and every initial total typed memory.
No successful-execution premise, output-equality premise, real-rounding law or
scalar algebraic identity is required. Native block/address validity, allocation,
hardware numerical binding and device scheduling are separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

specification batched_vecmat_fp32_machine_correctness
    (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word) :
    ∃ final, FP.Structural.exec (engine rest cfg plans) (source p) initial = some final ∧
      final.mem = expectedMemory cfg plans p initial ∧
      (∀ r o, Outside p initial r o → final.mem r o = initial.mem r o) := by
  exact complete_correct rest cfg plans p initial

theorem memory_independent_of_unused_backend
    (left right : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word) :
    ∃ l r, FP.Structural.exec (engine left cfg plans) (source p) initial = some l ∧
      FP.Structural.exec (engine right cfg plans) (source p) initial = some r ∧ l.mem = r.mem := by
  obtain ⟨l, hl, hml, _⟩ := complete_correct left cfg plans p initial
  obtain ⟨r, hr, hmr, _⟩ := complete_correct right cfg plans p initial
  exact ⟨l, r, hl, hr, hml.trans hmr.symm⟩

#axiomsClean batched_vecmat_fp32_machine_correctness
#axiomsClean memory_independent_of_unused_backend
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
