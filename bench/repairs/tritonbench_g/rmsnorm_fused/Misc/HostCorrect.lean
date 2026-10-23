import bench.repairs.tritonbench_g.rmsnorm_fused.Kernels
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.Contract
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.NativeIndex

namespace VeriTile.Bench.Repairs.RmsnormFused.HostCorrect
open VeriTile Triton
noncomputable section

def checked (x weight : TensorInfo) (eps : EpsilonInfo) (q : Host.Request)
    (call : Host.Preparation BlockState MemCell BlockState.mem q) (schedule : List Nat) :=
  if checkTensorInput x weight eps=.ok () then Host.invokeReal q call schedule else none

theorem wrapper_correct (x weight : TensorInfo) (eps : EpsilonInfo)
    (contract : InputContract x weight eps)
    (call : Host.Preparation BlockState MemCell BlockState.mem (request x weight eps contract))
    (inputObserved : Observes x call.input) (weightObserved : Observes weight call.weight)
    (schedule : List Nat) (all : schedule.Perm (geometry x).grid) :
    ∃ result, checked x weight eps (request x weight eps contract) call schedule=some result ∧
      result.tensor=call.output ∧ result.tensor.shape=x.shape ∧ result.tensor.dtype="float32" ∧
      result.tensor.device=x.device.index ∧ result.currentDevice=call.callerDevice ∧
      result.launchDevice=(if rows x=0 then none else some x.device.index) ∧
      (∀ row k,row<rows x → k<width x →
        result.state.readMem result.tensor.region (row*width x+k)=
          Real.formula call.before call.input.region call.weight.region row (width x) eps.value k) ∧
      (∀ r o,r≠result.tensor.region ∨ x.numel≤o → result.state.mem r o=call.before.mem r o) := by
  obtain ⟨result,he,ident,shape,dtype,device,context,launch,out,frame⟩ :=
    Host.whole_real (request x weight eps contract) call schedule all
  have rowCount := geometry_rows x contract.rank contract.widthPositive
  have size := geometry_size x contract.rank
  refine ⟨result,?_,ident,?_,dtype,device.trans inputObserved.device,context,?_,?_,?_⟩
  · simp [checked,(checkTensorInput_iff x weight eps).mpr contract,he]
  · exact shape.trans (geometry_shape x contract.rank)
  · simpa only [Host.Request.rows,request,rowCount,inputObserved.device] using launch
  · simpa only [Host.Request.rows,Host.Request.columns,request,rowCount] using out
  · simpa only [Host.Request.size,Host.Request.rows,Host.Request.columns,request,size] using frame

theorem rejected (x weight : TensorInfo) (eps : EpsilonInfo) (bad : ¬InputContract x weight eps)
    (q : Host.Request) (call : Host.Preparation BlockState MemCell BlockState.mem q) (schedule : List Nat) :
    checked x weight eps q call schedule=none := by
  have rejected : checkTensorInput x weight eps≠.ok () := fun h => bad ((checkTensorInput_iff x weight eps).mp h)
  simp [checked,rejected]

abbrev native_indices := NativeIndex.formed_index
abbrev native_addresses := @NativeIndex.source_pointers
abbrev active_accesses := NativeIndex.active_access
#axiomsClean wrapper_correct
#print axioms wrapper_correct
end
end VeriTile.Bench.Repairs.RmsnormFused.HostCorrect
