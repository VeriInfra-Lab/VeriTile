import bench.tritonbench_g.vector_addition.Proofs.PointwiseSchedule
import bench.tritonbench_g.vector_addition.Proofs.IntegerGrid
import bench.tritonbench_g.vector_addition.Proofs.MixedFP32Defined
import bench.tritonbench_g.vector_addition.Proofs.MixedSmallDefined

/-! Instantiate the pointwise scheduling proof with actual existing source
execution theorems. No public premise assumes execution success or correctness.
All original undefined conversion policies and finite-domain conditions remain
explicit. This concerns complete programs, not native instruction interleaving. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition
open VeriTile.Triton FP.Structural
open ValueProofs (Parameters gridCount programState)
noncomputable section

namespace IntegerSchedule

def run {α : Type} [Inhabited α] (M : Algebra α) (x y : IntegerSource.Storage)
    (p : Parameters) (initial : State α) (jobs : List Nat) : Option (State α) :=
  PointwisePrograms.runSchedule
    (fun pid current => FP.Structural.exec (M) (IntegerSource.source x y p)
      (programState p initial current.mem pid)) jobs initial

theorem result_local {α : Type} [Inhabited α] (x y : IntegerSource.Storage)
    (p : Parameters) : PointwisePrograms.Pointwise (IntegerSource.resultCell (α := α) x y p) := by
  intro a b o h
  simp only [IntegerSource.resultCell, h]

theorem complete {α : Type} [Inhabited α] (M : Algebra α) (x y : IntegerSource.Storage)
    (p : Parameters) (initial : State α) (positive : 0 < p.block)
    (jobs : List Nat) (order : jobs.Perm (List.range (gridCount p))) :
    ∃ final, run M x y p initial jobs = some final ∧
      final.mem = IntegerSource.gridExpectedMemory x y p initial := by
  apply PointwisePrograms.complete_schedule (IntegerSource.resultCell x y p)
    (result_local x y p) p
    (fun pid current => FP.Structural.exec (M) (IntegerSource.source x y p)
      (programState p initial current.mem pid)) ?_ initial positive jobs order
  intro pid current
  obtain ⟨final, he, hm⟩ := IntegerSource.complete_correct M x y p (programState p initial current.mem pid)
  refine ⟨final, he, hm.trans ?_⟩
  funext r o
  simpa only [programState, ↓reduceIte, PointwisePrograms.oneMemory] using
    (IntegerSource.expected_cell x y p (programState p initial current.mem pid) r o)

#axiomsClean result_local
#axiomsClean complete
#guard_msgs (drop info) in
#auditModuleAxioms
end IntegerSchedule

namespace MixedFP32Schedule
local instance : Inhabited MixedFP32.Word := ⟨FP.Value.zero .fp32⟩

def run (rest : Algebra MixedFP32.Word) (cfg : FP.Config)
    (us : Nat → MixedFP32.Word → Int) (uu : Nat → MixedFP32.Word → Nat)
    (integerFirst : Bool) (d : IntegerSource.Storage)
    (p : Parameters) (initial : State MixedFP32.Word) (jobs : List Nat) : Option (State MixedFP32.Word) :=
  PointwisePrograms.runSchedule
    (fun pid current => FP.Structural.exec (MixedFP32.backend rest cfg us uu) (MixedFP32.source integerFirst d p)
      (programState p initial current.mem pid)) jobs initial

theorem result_local (cfg : FP.Config)
    (us : Nat → MixedFP32.Word → Int) (uu : Nat → MixedFP32.Word → Nat)
    (integerFirst : Bool) (d : IntegerSource.Storage)
    (p : Parameters) : PointwisePrograms.Pointwise (MixedFP32.resultCell cfg us uu integerFirst d p) := by
  intro a b o h
  simp only [MixedFP32.resultCell, MixedFP32.sumValue, h]

theorem complete (rest : Algebra MixedFP32.Word) (cfg : FP.Config)
    (us : Nat → MixedFP32.Word → Int) (uu : Nat → MixedFP32.Word → Nat)
    (integerFirst : Bool) (d : IntegerSource.Storage)
    (p : Parameters) (initial : State MixedFP32.Word) (positive : 0 < p.block)
    (jobs : List Nat) (order : jobs.Perm (List.range (gridCount p))) :
    ∃ final, run rest cfg us uu integerFirst d p initial jobs = some final ∧
      final.mem = MixedFP32.gridExpectedMemory cfg us uu integerFirst d p initial := by
  apply PointwisePrograms.complete_schedule (MixedFP32.resultCell cfg us uu integerFirst d p)
    (result_local cfg us uu integerFirst d p) p
    (fun pid current => FP.Structural.exec (MixedFP32.backend rest cfg us uu) (MixedFP32.source integerFirst d p)
      (programState p initial current.mem pid)) ?_ initial positive jobs order
  intro pid current
  obtain ⟨final, he, hm⟩ := MixedFP32.complete_correct rest cfg us uu integerFirst d p (programState p initial current.mem pid)
  refine ⟨final, he, hm.trans ?_⟩
  funext r o
  simpa only [programState, ↓reduceIte, PointwisePrograms.oneMemory] using
    (MixedFP32.expected_cell cfg us uu integerFirst d p (programState p initial current.mem pid) r o)

#axiomsClean result_local
#axiomsClean complete
#guard_msgs (drop info) in
#auditModuleAxioms
end MixedFP32Schedule

namespace MixedSmallSchedule
local instance : Inhabited MixedSmall.Word := ⟨FP.Value.zero .fp32⟩

def run (rest : Algebra MixedSmall.Word) (cfg : FP.Config) (f : MixedSmall.FloatStorage)
    (us : Nat → MixedSmall.Word → Int) (uu : Nat → MixedSmall.Word → Nat)
    (integerFirst : Bool) (d : IntegerSource.Storage)
    (p : Parameters) (initial : State MixedSmall.Word) (jobs : List Nat) : Option (State MixedSmall.Word) :=
  PointwisePrograms.runSchedule
    (fun pid current => FP.Structural.exec (MixedSmall.backend rest cfg f us uu) (MixedSmall.source f integerFirst d p)
      (programState p initial current.mem pid)) jobs initial

theorem result_local (cfg : FP.Config) (f : MixedSmall.FloatStorage)
    (us : Nat → MixedSmall.Word → Int) (uu : Nat → MixedSmall.Word → Nat)
    (integerFirst : Bool) (d : IntegerSource.Storage)
    (p : Parameters) : PointwisePrograms.Pointwise (MixedSmall.resultCell cfg f us uu integerFirst d p) := by
  intro a b o h
  simp only [MixedSmall.resultCell, MixedSmall.sumValue, h]

theorem complete (rest : Algebra MixedSmall.Word) (cfg : FP.Config) (f : MixedSmall.FloatStorage)
    (us : Nat → MixedSmall.Word → Int) (uu : Nat → MixedSmall.Word → Nat)
    (integerFirst : Bool) (d : IntegerSource.Storage)
    (p : Parameters) (initial : State MixedSmall.Word) (positive : 0 < p.block)
    (jobs : List Nat) (order : jobs.Perm (List.range (gridCount p))) :
    ∃ final, run rest cfg f us uu integerFirst d p initial jobs = some final ∧
      final.mem = MixedSmall.gridExpectedMemory cfg f us uu integerFirst d p initial := by
  apply PointwisePrograms.complete_schedule (MixedSmall.resultCell cfg f us uu integerFirst d p)
    (result_local cfg f us uu integerFirst d p) p
    (fun pid current => FP.Structural.exec (MixedSmall.backend rest cfg f us uu) (MixedSmall.source f integerFirst d p)
      (programState p initial current.mem pid)) ?_ initial positive jobs order
  intro pid current
  obtain ⟨final, he, hm⟩ := MixedSmall.complete_correct rest cfg f us uu integerFirst d p (programState p initial current.mem pid)
  refine ⟨final, he, hm.trans ?_⟩
  funext r o
  simpa only [programState, ↓reduceIte, PointwisePrograms.oneMemory] using
    (MixedSmall.expected_cell cfg f us uu integerFirst d p (programState p initial current.mem pid) r o)

#axiomsClean result_local
#axiomsClean complete
#guard_msgs (drop info) in
#auditModuleAxioms
end MixedSmallSchedule

end
end VeriTile.Bench.TritonBenchG.VectorAddition
