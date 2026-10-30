import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.RealPublic
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.SeparateAPI
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.LossPolynomial
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
namespace Public

/-- Only actual forward entry memory, allocation separation and returned Tensor
handle binding occur here. No future cotangent or saved-context premise. -/
theorem forward_call (ops : Ops α) (c : Shape) (_abi : HostABI.LaunchABI c)
    (b : Binding c) (scale : α) (tree : SumTree) (rf seedPresent : Bool)
    (_seedBinding : b.zeroInput .seed= !seedPresent) (entry : Memory α)
    (device : Nat) (returned : Separate.Handle)
    (handle : returned=Separate.outputHandle c b device)
    (returnedFinal : Option Separate.Handle)
    (finalHandle : returnedFinal=if rf then some (Separate.stateHandle c b device) else none) :
    (∀ h t j,h<c.heads → t<c.T → j<c.V →
      Separate.forwardPhysical ops c b scale tree rf entry
        (returned.base+c.valueOffset h t j)=
        Reference.output ops c (inputsOfHeap (b.view ops.zero entry)) scale h t j) ∧
    (∀ result∈returnedFinal,∀ h e j,h<c.heads → e<c.K → j<c.V →
      Separate.forwardPhysical ops c b scale tree rf entry
        (result.base+c.stateOffset h e j)=
        Reference.finalState ops c (inputsOfHeap (b.view ops.zero entry)) h e j) ∧
    (∀ point,(∀ a∈Separate.forwardFootprint c rf,point≠b.address a) →
      Separate.forwardPhysical ops c b scale tree rf entry point=entry point) ∧
    returned.metadata=HostABI.valueMeta c device 0 := by
  subst returned
  have same:=Separate.forward_physical ops c b scale tree rf entry
  refine ⟨?_,?_,Separate.forward_physical_frame ops c b scale tree rf entry,rfl⟩
  · intro h t j hh ht hj
    have hv : (⟨.output,h,0,t,j⟩:Address).Valid c:=⟨hh,rfl,ht,hj⟩
    have eq:=congrFun same ⟨.output,h,0,t,j⟩
    rw [Source.o_value ops c scale tree rf _ h t j hh ht hj] at eq
    rw [b.view_read ops.zero _ _ hv (by simp [inputRegions])] at eq
    simpa only [Binding.address,Address.offset,Separate.outputHandle] using eq
  · intro result member h e j hh he hj
    have flag : rf=Bool.true := by cases rf <;> simp_all
    subst rf
    have resultEq : result=Separate.stateHandle c b device := by symm; simpa [finalHandle] using member
    subst result
    have hv : (⟨.finalState,h,0,e,j⟩:Address).Valid c:=⟨hh,rfl,he,hj⟩
    have eq:=congrFun same ⟨.finalState,h,0,e,j⟩
    rw [Source.afterO,Source.execute_frame ops c scale tree .o _ (Source.o_footprint c) _ _
      (by simp [Phase.writes]),Source.final_value ops c scale tree _ h e j hh he hj] at eq
    rw [b.view_read ops.zero _ _ hv (by simp [inputRegions])] at eq
    simpa only [Binding.address,Address.offset,Separate.stateHandle] using eq

/-- Backward starts from actual later memory. Saved handles/version equality
binds the actual Q/K/V/history reads to the forward call, while both incoming
cotangents may be supplied later. Returned gradient descriptors expose exactly
the physical bases used below; optional initial gradient follows the same run. -/
theorem backward_call (ops : Ops α) (c : Shape) (_abi : HostABI.LaunchABI c)
    (forwardBinding b : Binding c) (scale : α) (tree : SumTree)
    (complete : tree.Complete c.keyBlocks) (rf ri incomingPresent terminalPresent : Bool)
    (_incomingBinding : b.zeroInput .incoming= !incomingPresent)
    (_terminalBinding : b.zeroInput .terminal= !terminalPresent) (before later : Memory α)
    (saved : Separate.SavedContext ops c scale tree rf forwardBinding b before later)
    (device : Nat) (returned : List (Option Separate.Handle))
    (handles : returned=Separate.gradientHandles c b device ri) :
    Separate.BackwardPost ops c
      (Separate.joined (forwardBinding.view ops.zero before) (b.view ops.zero later)) scale tree
      (b.view ops.zero (Separate.backwardPhysical ops c b scale tree ri later)) ∧
    (ri=Bool.true → ∀ h e j,h<c.heads → e<c.K → j<c.V →
      Separate.backwardPhysical ops c b scale tree ri later (b.base .initialGradient+c.stateOffset h e j)=
        ops.cast (Reference.reverseCarry ops c (inputsOfHeap (b.view ops.zero later)) scale h c.chunks e j)) ∧
    (∀ point,(∀ a∈Separate.backwardFootprint c ++ if ri then Source.eventsInitial c else [],point≠b.address a) →
      Separate.backwardPhysical ops c b scale tree ri later point=later point) ∧
    returned.map (Option.map Separate.Handle.base)=
      [some (b.base .dq),some (b.base .dk),some (b.base .dv),
       if ri then some (b.base .initialGradient) else none,none,none,none] := by
  have same:=Separate.backward_physical ops c b scale tree ri later
  refine ⟨?_,?_,Separate.backward_physical_frame ops c b scale tree ri later,?_⟩
  · rw [same]
    exact Separate.backwardAPI_post ops c scale tree complete rf ri _ _ saved.versions
  · intro flag h e j hh he hj
    subst ri
    have hv : (⟨.initialGradient,h,0,e,j⟩:Address).Valid c:=⟨hh,rfl,he,hj⟩
    have eq:=congrFun same ⟨.initialGradient,h,0,e,j⟩
    rw [Separate.backwardAPI_initial ops c scale tree _ h e j hh he hj] at eq
    rw [b.view_read ops.zero _ _ hv (by simp [inputRegions])] at eq
    simpa only [Binding.address,Address.offset] using eq
  · rw [handles]; cases ri <;> rfl

/-- Exact polynomial expansion of the independently defined loss. The first
coefficient is the pairing with all four returned mathematical gradients. -/
theorem loss_gradient_polynomial (c : Shape) (xs : Inputs ℝ) (d : MathTarget.Variation)
    (scale : ℝ) (hasSeed : Bool) (z : ℝ) :
    MathTarget.loss c (MathTarget.perturb xs d z) scale hasSeed=
      MathTarget.loss c xs scale hasSeed+z*MathTarget.gradientPairing c xs d scale hasSeed+
      z^2*MathTarget.secondCoefficient c xs d scale hasSeed+z^3*MathTarget.thirdCoefficient c xs d scale :=
  MathTarget.loss_gradient_polynomial c xs d scale hasSeed z

#axiomsClean forward_call
#axiomsClean backward_call
#axiomsClean loss_gradient_polynomial
end Public
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001

