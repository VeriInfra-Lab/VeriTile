import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.Runtime
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Binding

namespace VeriTile.Bench.Repairs.L2NormTriton2.Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormTriton1.Host (World Result)
noncomputable section

def checkedCall {α : Type} (metadata : TensorMetadata) (input gradient : View) (eps : α) : Call α :=
  ⟨L2NormTriton1.Host.checkedCall metadata input eps,gradient⟩

def checkedExecute {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (a : Source.Adapter α γ) (rt : Runtime M) (x dy : TensorMetadata)
    (input gradient : View) (eps : γ) (s : World M) : Except InputError (Result M View) := do
  checkInputs (.tensor x) (.tensor dy)
  return execute ops a rt (checkedCall x input gradient eps) s

theorem checked_domain {M α : Type} (x dy : TensorMetadata) (input gradient : View) (eps : α)
    (s : World M) (h : InputContract x dy) (bx : TensorBinding x input) (bd : TensorBinding dy gradient)
    (lx : input.storage∈s.allocations) (ld : gradient.storage∈s.allocations) :
    Domain (checkedCall x input gradient eps) s := by
  refine ⟨⟨bx.backed,lx,bx.device⟩,bd.backed,ld,?_,?_,?_⟩
  · change gradient.storage.owner=input.storage.owner
    rw [←bd.device,←bx.device,h.sameDevice]
  · intro useGradient
    cases useGradient
    · exact direct_view x input bx h.contiguous.1 h.noNeg.1 (fun hn=>(h.alignment hn).1)
    · exact direct_view dy gradient bd h.contiguous.2 h.noNeg.2
        (fun hn=>(h.alignment (by rw [h.sameCount];exact hn)).2)
  · change 0<gradient.shape.getLastD 0
    rw [←bd.shape]
    exact h.width.1

theorem checked_execute {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (a : Source.Adapter α γ) (rt : Runtime M) (x dy : TensorMetadata)
    (input gradient : View) (eps : γ) (s : World M) (h : InputContract x dy) :
    checkedExecute ops a rt x dy input gradient eps s=.ok (execute ops a rt (checkedCall x input gradient eps) s) := by
  simp [checkedExecute,checkInputs,(checkTensorInputs_iff x dy).mpr h]

#axiomsClean checked_domain
#axiomsClean checked_execute
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.Host
