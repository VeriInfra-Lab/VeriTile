import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Runtime

namespace VeriTile.Bench.Repairs.L2NormTriton1.Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section

def checkedCall {α : Type} (metadata : TensorMetadata) (input : View) (eps : α) : Call α :=
  ⟨input,eps,metadata.device.index,Bool.true⟩

/-- Actual ordered Python admission precedes the direct-view host program. -/
def checkedExecute {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (a : Source.Adapter α γ) (rt : Runtime M) (metadata : TensorMetadata)
    (input : View) (eps : γ) (s : World M) : Except InputError (Result M View) := do
  checkInputs (.tensor metadata)
  return execute ops a rt (checkedCall metadata input eps) s

theorem checked_domain {M α : Type} (metadata : TensorMetadata) (input : View) (eps : α)
    (s : World M) (h : InputContract metadata) (binding : TensorBinding metadata input)
    (live : input.storage∈s.allocations) : Domain (checkedCall metadata input eps) s :=
  ⟨binding.backed,live,binding.device,direct_input metadata input h binding⟩

theorem checked_execute {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (a : Source.Adapter α γ) (rt : Runtime M) (metadata : TensorMetadata)
    (input : View) (eps : γ) (s : World M) (h : InputContract metadata) :
    checkedExecute ops a rt metadata input eps s=.ok (execute ops a rt (checkedCall metadata input eps) s) := by
  simp [checkedExecute,checkInputs,(checkTensorInputs_iff metadata).mpr h]

#axiomsClean checked_domain
#axiomsClean checked_execute
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.Host
