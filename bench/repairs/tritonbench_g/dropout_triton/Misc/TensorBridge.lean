import bench.repairs.tritonbench_g.dropout_triton.Proofs.MemoryContract

/-! Optional Tensor input bindings and Python-check correspondence. Primary
mathematical and native kernel entries do not import this bridge. -/
namespace VeriTile.Bench.Repairs.DropoutTriton.TensorBridge
open VeriTile Triton
noncomputable section

/-- Both input observations use the same physical word reader; FP32 data and
int32 truth values are separate interpretations of those same bytes. The
views may be identical or partially overlap, and no separation premise occurs. -/
theorem checked_tensor_reads (read : Nat → BitVec 32)
    (x keep : VeriTile.Torch.FP32.Tensor)
    (lx : VeriTile.Torch.FP32.TensorLaws x) (lk : MemoryContract.Int32Laws keep)
    (h : TensorContract x.metadata keep.metadata) (i : Nat) (hi : i < x.metadata.numel) :
    (⟨read ((VeriTile.Torch.FP32.dataView x lx h.contiguous.1).address i)⟩ : FP.Value .fp32) =
      VeriTile.Torch.FP32.logicalValue (fun a => ⟨read a⟩) x lx i ∧
    (read ((MemoryContract.keepView keep lk h.contiguous.2).address i)).toInt =
      MemoryContract.logicalKeep read keep lk i :=
  ⟨MemoryContract.direct_data_read (fun a => ⟨read a⟩) x keep lx h i hi,
    MemoryContract.direct_keep_read read x keep lk h i hi⟩

theorem python_checks_iff_preconditions (x keep : ArgumentMetadata) (p : ScalarArgument) :
    checkInputs x keep p = .ok () ↔ AdmittedArguments x keep p := checkInputs_iff x keep p

#axiomsClean checked_tensor_reads
#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DropoutTriton.TensorBridge
