import bench.tritonbench_g.vector_addition.Proofs.DTypeAdmission

/-! Public finite-model statements for the frozen default sm90 dtype gates.
No original Python, JIT, native compiler or GPU correctness is asserted.
Source branch audits and recorded exceptions are separate evidence, not proof
premises. The 28 promotion-success fnuz pairs are rejection-model cases, not
additional value-execution profiles. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.DTypeAdmissionCorrect
open DTypeScope DTypeAdmission

specification vector_addition_sm90_admission_model_partition :
    pairs.length = 289 ∧ pairs.Nodup ∧
    (stageRows .admitted).length = 189 ∧
    (stageRows .signature).length = 64 ∧
    (stageRows .addition).length = 36 := by
  exact ⟨pair_inventory.1, pair_inventory.2, stage_partition_counts⟩

specification vector_addition_fnuz_first_stage_model (x y : DType) :
    (fnuz x = Bool.true → admission x y = .signatureRejected .x x) ∧
    (fnuz x = Bool.false → fnuz y = Bool.true → admission x y = .signatureRejected .y y) :=
  fnuz_first_stage x y

specification vector_addition_sm90_add_rejection_model (x y : DType) :
    admission x y = .additionRejected ↔
      fnuz x = Bool.false ∧ fnuz y = Bool.false ∧ promote x y = none :=
  addition_rejected_iff x y

specification vector_addition_sm90_promotion_stage_cross_model :
    ((pairs.filter fun pair => (promote pair.1 pair.2).isSome).length = 217 ∧
     (pairs.filter fun pair => (promote pair.1 pair.2).isNone).length = 72) ∧
    (crossRows .admitted Bool.true).length = 189 ∧
    (crossRows .addition Bool.false).length = 36 ∧
    (crossRows .signature Bool.false).length = 36 ∧
    (crossRows .signature Bool.true).length = 28 := by
  exact ⟨standalone_promotion_counts, stage_promotion_cross_counts⟩

#axiomsClean vector_addition_sm90_admission_model_partition
#axiomsClean vector_addition_fnuz_first_stage_model
#axiomsClean vector_addition_sm90_add_rejection_model
#axiomsClean vector_addition_sm90_promotion_stage_cross_model
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.VectorAddition.DTypeAdmissionCorrect
