import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Scope
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
set_option linter.unusedSimpArgs false

/- Store payloads computed from the actual per-program carried tile. The
owner maps are those proved complete/injective in Grid; this is a store-order
normalization of that grid, not a numerical postcondition supplied by a caller. -/
theorem sourceCell_partial (a : Arithmetic α) (s : Shape) (p : Plans) (x : Inputs α)
    (seed final : Bool) (kind : PartialKind) (i : Nat) :
    sourceCell a s p x seed final kind i=partialValue a s p x seed final kind i := by
  cases kind <;> simp [sourceCell,partialValue,programO_partial,programQ_partial,
    programK_partial,programV_partial,programForwardCarry_at,ProgramId.key,ProgramId.value,owner_reconstruct]

theorem sourcePhase_normalized (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (seed final : Bool) (kind : PartialKind) (storage count : Nat) (m : Memory α) :
    sourcePhase a s p b seed final kind storage count m=partialPhase a s p b seed final kind storage count m := by
  unfold sourcePhase partialPhase
  rw [funext (sourceCell_partial a s p (b.observe s m) seed final kind)]

theorem sourceForward_normalized (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (alloc : Allocation 3 caller) (seed final : Bool) (m : Memory α) :
    sourceForward a s p b alloc seed final m=forwardRun a s p b alloc seed final m := by
  simp only [sourceForward,sourcePhase_normalized,forwardRun,forwardPartials]

theorem sourceBackward_normalized (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (alloc : Allocation 6 caller) (seed final : Bool) (m : Memory α) :
    sourceBackward a s p b alloc seed final m=backwardRun a s p b alloc seed final m := by
  simp only [sourceBackward,sourcePhase_normalized,backwardRun,backwardPartials]

/- Any permutation of the uniquely owned store cells has the same memory
result. Reads of Q/K/V/seed/do/F stay fixed because all output storage is fresh. -/
theorem fill_permutation (storage : Nat) (value : Nat → α) (left right : List Nat)
    (perm : left.Perm right) (m : Memory α) : fill storage value left m=fill storage value right m := by
  funext address
  simp only [fill_get,perm.mem_iff]

#axiomsClean sourceCell_partial
#axiomsClean sourcePhase_normalized
#axiomsClean sourceForward_normalized
#axiomsClean sourceBackward_normalized
#axiomsClean fill_permutation
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
