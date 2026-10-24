import bench.tritonbench_g.vector_addition.Proofs.ValueSchedules

/-! Complete permutations for the existing 81 integer/bool and 54 mixed
floating profiles. No new dtype pairs are introduced. Actual previous memory
is supplied to each source program, and the result uses independent initial
memory. Native bytes, allocation, interleaving and source/compiler/errors
remain separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.ValueScheduleCorrect
open VeriTile.Triton FP.Structural
open ValueProofs (Parameters gridCount)
open IntegerSource (Canonical)
noncomputable section
local instance : Inhabited MixedFP32.Word := ⟨FP.Value.zero .fp32⟩

specification vector_addition_integer_program_order_correctness
    {α : Type} [Inhabited α] (M : Algebra α) (x y : Canonical)
    (p : Parameters) (initial : State α) (positive : 0 < p.block)
    (jobs : List Nat) (order : jobs.Perm (List.range (gridCount p))) :
    ∃ final, IntegerSchedule.run M x.storage y.storage p initial jobs = some final ∧
      final.mem = IntegerSource.gridExpectedMemory x.storage y.storage p initial := by
  exact IntegerSchedule.complete M x.storage y.storage p initial positive jobs order

specification vector_addition_mixed_fp32_total_program_order_correctness
    (rest : Algebra MixedFP32.Word) (cfg : FP.Config)
    (us : Nat → MixedFP32.Word → Int) (uu : Nat → MixedFP32.Word → Nat)
    (integerFirst : Bool) (d : Canonical)
    (p : Parameters) (initial : State MixedFP32.Word) (positive : 0 < p.block)
    (jobs : List Nat) (order : jobs.Perm (List.range (gridCount p))) :
    ∃ final, MixedFP32Schedule.run rest cfg us uu integerFirst d.storage p initial jobs = some final ∧
      final.mem = MixedFP32.gridExpectedMemory cfg us uu integerFirst d.storage p initial := by
  exact MixedFP32Schedule.complete rest cfg us uu integerFirst d.storage p initial positive jobs order

specification vector_addition_mixed_fp32_defined_program_order_correctness
    (rest : Algebra MixedFP32.Word) (cfg : FP.Config)
    (us : Nat → MixedFP32.Word → Int) (uu : Nat → MixedFP32.Word → Nat)
    (integerFirst : Bool) (d : Canonical)
    (p : Parameters) (initial : State MixedFP32.Word) (positive : 0 < p.block)
    (jobs : List Nat) (order : jobs.Perm (List.range (gridCount p)))
    (defined : MixedFP32.OutputDefined cfg integerFirst d.storage p initial) :
    ∃ final, MixedFP32Schedule.run rest cfg us uu integerFirst d.storage p initial jobs = some final ∧
      final.mem = MixedFP32.specifiedMemory cfg integerFirst d.storage p initial := by
  obtain ⟨final, he, hm⟩ := MixedFP32Schedule.complete rest cfg us uu integerFirst d.storage p initial positive jobs order
  refine ⟨final, he, hm.trans ?_⟩
  exact MixedFP32.grid_memory_policy_independent cfg us (fun _ _ => 0) uu (fun _ _ => 0)
    integerFirst d.storage p initial defined

specification vector_addition_mixed_small_total_program_order_correctness
    (rest : Algebra MixedSmall.Word) (cfg : FP.Config) (f : MixedSmall.FloatStorage)
    (us : Nat → MixedSmall.Word → Int) (uu : Nat → MixedSmall.Word → Nat)
    (integerFirst : Bool) (d : Canonical)
    (p : Parameters) (initial : State MixedSmall.Word) (positive : 0 < p.block)
    (jobs : List Nat) (order : jobs.Perm (List.range (gridCount p))) :
    ∃ final, MixedSmallSchedule.run rest cfg f us uu integerFirst d.storage p initial jobs = some final ∧
      final.mem = MixedSmall.gridExpectedMemory cfg f us uu integerFirst d.storage p initial := by
  exact MixedSmallSchedule.complete rest cfg f us uu integerFirst d.storage p initial positive jobs order

specification vector_addition_mixed_small_defined_program_order_correctness
    (rest : Algebra MixedSmall.Word) (cfg : FP.Config) (f : MixedSmall.FloatStorage)
    (us : Nat → MixedSmall.Word → Int) (uu : Nat → MixedSmall.Word → Nat)
    (integerFirst : Bool) (d : Canonical)
    (p : Parameters) (initial : State MixedSmall.Word) (positive : 0 < p.block)
    (jobs : List Nat) (order : jobs.Perm (List.range (gridCount p)))
    (defined : MixedSmall.OutputDefined cfg f integerFirst d.storage p initial) :
    ∃ final, MixedSmallSchedule.run rest cfg f us uu integerFirst d.storage p initial jobs = some final ∧
      final.mem = MixedSmall.specifiedMemory cfg f integerFirst d.storage p initial := by
  obtain ⟨final, he, hm⟩ := MixedSmallSchedule.complete rest cfg f us uu integerFirst d.storage p initial positive jobs order
  refine ⟨final, he, hm.trans ?_⟩
  exact MixedSmall.grid_memory_policy_independent cfg f us (fun _ _ => 0) uu (fun _ _ => 0)
    integerFirst d.storage p initial defined

#axiomsClean vector_addition_integer_program_order_correctness
#axiomsClean vector_addition_mixed_fp32_total_program_order_correctness
#axiomsClean vector_addition_mixed_fp32_defined_program_order_correctness
#axiomsClean vector_addition_mixed_small_total_program_order_correctness
#axiomsClean vector_addition_mixed_small_defined_program_order_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.ValueScheduleCorrect
