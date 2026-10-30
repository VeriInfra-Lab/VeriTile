import bench.tritonbench_g.vector_addition.Proofs.Mathematical_MixedSmallDefined
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_MixedSmallProfiles

/-! Public concrete-software contracts for the 36 ordered integer/bool↔fp16/bf16
profiles. Totalized semantics expose every undefined-conversion policy. On
the active defined domain the independent memory result is policy-free.
Native numerical bindings, finite-width addressing, byte views and native
compiler/error/scheduling behavior are not established by these theorems. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.MixedSmallValueCorrect
open VeriTile.Triton FP.Structural
open ValueProofs (Parameters)
open IntegerSource (Canonical)
open MixedSmall (Word FloatStorage)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

specification vector_addition_mixed_small_total_correctness
    (rest : Algebra Word) (cfg : FP.Config) (f : FloatStorage)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Canonical) (p : Parameters) (initial : State Word)
    (positive : 0 < p.block) :
    ∃ final, MixedSmall.runGrid rest cfg f us uu integerFirst d.storage p initial = some final ∧
      final.mem = MixedSmall.gridExpectedMemory cfg f us uu integerFirst d.storage p initial := by
  exact MixedSmall.grid_correct rest cfg f us uu integerFirst d.storage p initial positive

specification vector_addition_mixed_small_defined_correctness
    (rest : Algebra Word) (cfg : FP.Config) (f : FloatStorage)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Canonical) (p : Parameters) (initial : State Word)
    (positive : 0 < p.block)
    (defined : MixedSmall.OutputDefined cfg f integerFirst d.storage p initial) :
    ∃ final, MixedSmall.runGrid rest cfg f us uu integerFirst d.storage p initial = some final ∧
      final.mem = MixedSmall.specifiedMemory cfg f integerFirst d.storage p initial := by
  exact MixedSmall.defined_grid_correct rest cfg f us uu integerFirst d.storage p initial positive defined

#axiomsClean vector_addition_mixed_small_total_correctness
#axiomsClean vector_addition_mixed_small_defined_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.MixedSmallValueCorrect
