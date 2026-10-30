import bench.repairs.tritonbench_g.softmax_flaggems.Contract
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.SoftmaxFlaggems
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid

theorem checkTensor_iff (x : TensorMetadata) : checkTensor x=.ok () ↔ TensorContract x := by
  by_cases empty : x.numel=0
  · simp [checkTensor,TensorContract,empty,ite_eq_iff,and_assoc]
  · have pos : 0<x.numel := Nat.pos_of_ne_zero empty
    simp [checkTensor,TensorContract,empty,pos,ite_eq_iff,and_assoc]

theorem checkForward_iff (x : TensorMetadata) (dim : Int) (dtype : Option VeriTile.Torch.DType) :
    checkForward x dim dtype=.ok () ↔ ForwardContract x dim dtype := by
  unfold checkForward
  cases h : checkTensor x with
  | error e =>
    have bad : ¬TensorContract x := by intro h';have := checkTensor_iff x |>.mpr h';simp_all
    simp [h,ForwardContract,bad]
  | ok u =>
    cases u
    have good := checkTensor_iff x |>.mp h
    simp [h,ForwardContract,good,ite_eq_iff,and_assoc] <;> tauto

theorem checkGradient_iff (x g : TensorMetadata) : checkGradient x g=.ok () ↔ GradientContract x g := by
  unfold checkGradient
  cases h : checkTensor g with
  | error e =>
    have bad : ¬TensorContract g := by intro h';have := checkTensor_iff g |>.mpr h';simp_all
    simp [h,GradientContract,bad]
  | ok u =>
    cases u
    have good := checkTensor_iff g |>.mp h
    simp [h,GradientContract,good,ite_eq_iff,and_assoc]

theorem admitted_axis (x : TensorMetadata) (dim : Int) (dtype : Option VeriTile.Torch.DType)
    (h : ForwardContract x dim dtype) : ∃ ax, normalizeDim x.shape dim=some ax := by
  cases eq : normalizeDim x.shape dim with
  | some a => exact ⟨a,rfl⟩
  | none => exact False.elim ((normalize_none x.shape dim).mp eq ⟨h.2.1,h.2.2.1⟩)

#axiomsClean checkTensor_iff
#axiomsClean checkForward_iff
#axiomsClean checkGradient_iff
#axiomsClean admitted_axis

theorem checkDynamic_iff (x : ArgumentMetadata) (dim : DimArgument) (dtype : Option VeriTile.Torch.DType) :
    checkDynamic x dim dtype=.ok () ↔ DynamicContract x dim dtype := by
  cases dim with
  | other => simp [checkDynamic,DynamicContract]
  | integer d =>
    cases x with
    | nonTensor => simp [checkDynamic,checkForward,DynamicContract]
    | tensor t => simpa [checkDynamic,DynamicContract] using checkForward_iff t d dtype

#axiomsClean checkDynamic_iff
end VeriTile.Bench.Repairs.SoftmaxFlaggems
