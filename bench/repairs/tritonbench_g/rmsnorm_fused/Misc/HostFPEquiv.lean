import bench.repairs.tritonbench_g.rmsnorm_fused.Kernels
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.Contract
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.Numerical
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.NativeIndex

namespace VeriTile.Bench.Repairs.RmsnormFused.HostFPEquiv
open VeriTile Triton FP.Structural Native
noncomputable section

def checked (x weight : TensorInfo) (eps : EpsilonInfo) (R : Reductions) (q : Host.Request)
    (call : Host.Preparation (State Term) (Cell Term) State.mem q) (schedule : List Nat) :=
  if checkTensorInput x weight eps=.ok () then Host.invoke (algebra R) q call schedule else none

/-- The complete symbolic source is evaluated by the selected ordered native
tree/FMA schedule. This structural theorem assumes no scalar numerical equality. -/
theorem wrapper_fp32 (x weight : TensorInfo) (eps : EpsilonInfo)
    (contract : InputContract x weight eps) (I : Instructions) (R : Reductions) (S : Schedule)
    (memory : Physical.Bytes)
    (call : Host.Preparation (State Term) (Cell Term) State.mem (request x weight eps contract))
    (inputObserved : Observes x call.input) (weightObserved : Observes weight call.weight)
    (hbefore : call.before=entry (Physical.load memory)) (schedule : List Nat)
    (all : schedule.Perm (geometry x).grid) :
    ∃ result,checked x weight eps R (request x weight eps contract) call schedule=some result ∧
      result.tensor=call.output ∧ result.tensor.shape=x.shape ∧ result.tensor.dtype="float32" ∧
      result.tensor.device=x.device.index ∧ result.currentDevice=call.callerDevice ∧
      result.launchDevice=(if rows x=0 then none else some x.device.index) ∧
      (∀ row k,row<rows x → k<width x → ∀ i : Fin 4,
        Physical.project I S result.state result.tensor.region (4*(row*width x+k)+i.val)=
          Physical.encode (value I R S (Physical.load memory) call.input.region call.weight.region
            (request x weight eps contract) row k) i) ∧
      (∀ r b,r≠result.tensor.region ∨ 4*x.numel≤b → Physical.project I S result.state r b=memory r b) := by
  obtain ⟨result,he,ident,shape,dtype,device,context,launch,out,frame⟩ :=
    Physical.whole_bytes I R S (request x weight eps contract) memory call hbefore schedule all
  have rowCount := geometry_rows x contract.rank contract.widthPositive
  have size := geometry_size x contract.rank
  refine ⟨result,?_,ident,?_,dtype,device.trans inputObserved.device,context,?_,?_,?_⟩
  · simp [checked,(checkTensorInput_iff x weight eps).mpr contract,he]
  · exact shape.trans (geometry_shape x contract.rank)
  · simpa only [Host.Request.rows,request,rowCount,inputObserved.device] using launch
  · simpa only [Host.Request.rows,Host.Request.columns,request,rowCount] using out
  · simpa only [Host.Request.size,Host.Request.rows,Host.Request.columns,request,size] using frame

abbrev scalar_equivalence := Numerical.whole_equiv
abbrev reached_primitives := Numerical.value_supported
abbrev native_indices := NativeIndex.formed_index
abbrev native_addresses := @NativeIndex.source_pointers
#axiomsClean wrapper_fp32
#print axioms wrapper_fp32
end
end VeriTile.Bench.Repairs.RmsnormFused.HostFPEquiv
