import bench.tritonbench_g.vector_addition.Proofs.MixedFP32Defined
import bench.tritonbench_g.vector_addition.Proofs.MixedFP32Profiles

/-! Public concrete-software contracts for the 18 ordered integer/bool↔fp32
profiles. Totalized semantics expose every undefined-conversion policy. On
the active defined domain the independent memory result is policy-free.
Native numerical bindings, finite-width addressing, byte views and native
compiler/error/scheduling behavior are not established by these theorems. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.MixedFP32ValueCorrect
open VeriTile.Triton FP.Structural
open ValueProofs (Parameters)
open IntegerSource (Canonical)
open MixedFP32 (Word)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

specification vector_addition_mixed_fp32_total_correctness
    (rest : Algebra Word) (cfg : FP.Config)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Canonical) (p : Parameters) (initial : State Word)
    (positive : 0 < p.block) :
    ∃ final, MixedFP32.runGrid rest cfg us uu integerFirst d.storage p initial = some final ∧
      final.mem = MixedFP32.gridExpectedMemory cfg us uu integerFirst d.storage p initial := by
  exact MixedFP32.grid_correct rest cfg us uu integerFirst d.storage p initial positive

specification vector_addition_mixed_fp32_defined_correctness
    (rest : Algebra Word) (cfg : FP.Config)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Canonical) (p : Parameters) (initial : State Word)
    (positive : 0 < p.block)
    (defined : MixedFP32.OutputDefined cfg integerFirst d.storage p initial) :
    ∃ final, MixedFP32.runGrid rest cfg us uu integerFirst d.storage p initial = some final ∧
      final.mem = MixedFP32.specifiedMemory cfg integerFirst d.storage p initial := by
  exact MixedFP32.defined_grid_correct rest cfg us uu integerFirst d.storage p initial positive defined

#axiomsClean vector_addition_mixed_fp32_total_correctness
#axiomsClean vector_addition_mixed_fp32_defined_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.MixedFP32ValueCorrect
