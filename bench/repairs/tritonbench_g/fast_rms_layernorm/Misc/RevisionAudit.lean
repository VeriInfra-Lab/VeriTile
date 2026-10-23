import bench.repairs.tritonbench_g.fast_rms_layernorm.Misc.OptionalHostCorrect
import bench.repairs.tritonbench_g.fast_rms_layernorm.Misc.OptionalHostFPEquiv
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.SavedAliasInhabited

/-! New producer-side verification consumer. This is not the old or independent
review Reader. Neither public entry imports this file or the finite witness. -/
namespace VeriTile.Bench.Repairs.FastRmsLayernorm.Revision002Audit
open VeriTile.Triton Host Storage SavedAliasInhabited
noncomputable section

theorem visible_saved_alias_all_outcomes {α β : Type} (rows : Fin 2) (saved : CellHeap α)
    (weight input : Call → CellHeap α → Option (CellHeap α)) (zero : α) (g : Upstream α)
    (encode : α → Fin 4 → β) (entry : Heap β) (w x z : Gate α)
    (resolve savedRead : Except Failure Unit) (addr : Nat) (active : live addr) :
    (Host.backward weight input zero (context rows saved) g live encode entry
      (.ok (prepared rows saved)) w x z resolve savedRead).bytes addr=entry addr :=
  Host.backward_frame weight input zero (context rows saved) g live encode entry
    (.ok (prepared rows saved)) w x z resolve addr active savedRead

#axiomsClean OptionalHostCorrect.wrapper_correct
#axiomsClean OptionalHostFPEquiv.wrapper_equiv
#axiomsClean Host.forward_frame
#axiomsClean Host.backward_frame
#axiomsClean HostStorage.saved_survives
#axiomsClean HostStorage.saved_backing_not_written
#axiomsClean HostStorage.backward_saved_full_frame
#axiomsClean HostStorage.backward_current_saved_readback
#axiomsClean HostStorage.backward_readback
#axiomsClean HostIdentity.gradients_fresh
#axiomsClean HostIdentity.saved_identity
#axiomsClean SavedAliasInhabited.preparation_with_visible_saved
#axiomsClean SavedAliasInhabited.saved_x_valid
#axiomsClean SavedAliasInhabited.saved_w_valid
#axiomsClean SavedAliasInhabited.saved_r_valid
#axiomsClean SavedAliasInhabited.zero_row_saved_x
#axiomsClean SavedAliasInhabited.raw_payload_not_restored
#axiomsClean SavedAliasInhabited.version_error_preserves_current_entry
#axiomsClean visible_saved_alias_all_outcomes
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.Revision002Audit

#axiomsClean VeriTile.Bench.Repairs.FastRmsLayernorm.NativeHost.forward
#axiomsClean VeriTile.Bench.Repairs.FastRmsLayernorm.NativeHost.backward
#axiomsClean VeriTile.Bench.Repairs.FastRmsLayernorm.NativeGraph.reduction_complete
#auditStmt VeriTile.Bench.Repairs.FastRmsLayernorm.NativeGraph.CompilerBinding
#auditStmt VeriTile.Bench.Repairs.FastRmsLayernorm.NativeGraph.Instructions
