import bench.tritonbench_g.vector_addition.Proofs.TypedSchedule
import bench.tritonbench_g.vector_addition.TypedValueCorrect

/-! Public specification for any complete ordering of the 25 represented
floating profiles. The output and frame use the same independent formula as
the ascending-grid contract. No input/output region separation is required.
Arithmetic/casts remain arbitrary; byte views and native concurrency remain
unproved, as do the other original dtype and host branches. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.TypedScheduleCorrect
open VeriTile.Triton FP.Structural
open ValueProofs (Parameters gridCount)
noncomputable section

specification vector_addition_typed_program_order_correctness
    {α : Type} [Inhabited α] (M : Algebra α)
    (x y : TypedFloat.Storage) (p : Parameters) (initial : State α)
    (positive : 0 < p.block) (jobs : List Nat)
    (order : jobs.Perm (List.range (gridCount p))) :
    ∃ final,
      TypedFloat.runSchedule M x.dtype y.dtype (TypedFloat.promote x y).dtype
        p initial jobs = some final ∧
      final.mem = TypedValueCorrect.spec M x y p initial := by
  exact TypedFloat.complete_schedule_correct M x.dtype y.dtype
    (TypedFloat.promote x y).dtype p initial positive jobs order

#axiomsClean vector_addition_typed_program_order_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.TypedScheduleCorrect
