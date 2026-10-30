import bench.repairs.tritonbench_g.fused_recurrent_delta.ABISource
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Index
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Native
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.ABI
noncomputable section
/-- The selected domain uses independent contiguous FP32 allocations and a
completed ordinary first-order autograd call. Tensor metadata are constructed
below, so arbitrary layouts or expanded/aliased cotangents are not admitted. -/
def Admitted (c : Call) : Prop :=
  Index.Safe c.shape (flags c) ∧ (c.defaultBeta=Bool.true → c.vectorBeta=Bool.false) ∧
    (c.terminalCotangent=Bool.true → c.outputFinal=Bool.true) ∧
    (∃ q : Rat,c.scale.launch.toRat?=some q)
theorem flags_normal (c : Call) (hc : Admitted c) :
    ((flags c).initialGrad=Bool.true → (flags c).initial=Bool.true) ∧
    ((flags c).terminalGrad=Bool.true → (flags c).final=Bool.true) ∧
    ((flags c).defaultBeta=Bool.true → (flags c).vectorBeta=Bool.false) := by
  constructor
  · intro hi
    change (c.initial && c.initialRequiresGrad)=Bool.true at hi
    cases hh:c.initial <;> simp_all [flags]
  · exact ⟨hc.2.2.1,hc.2.1⟩
theorem seven_backward_slots (c : Call) : (backwardReturn c).length=7 := by simp [backwardReturn]
theorem source_return_regions (c : Call) :
    (backwardReturn c).map (Option.map Tensor.allocation)=backwardReturns (flags c) := by
  cases hi:c.initial <;> cases hg:c.initialRequiresGrad <;> simp [backwardReturn,backwardReturns,flags,tensorMeta,hi,hg]
theorem forward_return_regions (c : Call) :
    ((forwardReturn c).1.allocation, (forwardReturn c).2.map Tensor.allocation)=forwardReturns (flags c) := by
  cases hf:c.outputFinal <;> simp [forwardReturn,forwardReturns,flags,tensorMeta,hf]
theorem default_beta_ones (c : Call) (hc : Admitted c) (m : Memory Native.Word)
    (b t : Nat) (hb : b<c.shape.BH) (ht : t<c.shape.T) (hf : c.defaultBeta=Bool.true) :
    prepareBeta c.shape (flags c) Native.graphOps (Native.graphMemory m) ⟨.beta,b,t,0⟩=Native.Expr.one := by
  simp [prepareBeta,flags,write,hf,hb,ht,Native.graphOps]
theorem single_key_grid (c : Call) (hc : Admitted c) : ceil c.shape.K c.shape.BK=1 :=
  Geometry.nk_one c.shape hc.1.positive.2.2.2.1

#axiomsClean flags_normal
#axiomsClean seven_backward_slots
#axiomsClean source_return_regions
#axiomsClean forward_return_regions
#axiomsClean default_beta_ones
#axiomsClean single_key_grid
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.ABI
