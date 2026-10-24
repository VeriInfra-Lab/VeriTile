import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.Host
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.Contract
open VeriTile.Torch
noncomputable section
open Classical

structure TensorInfo extends TensorMetadata where
  isConj : Bool
  object : Nat
  backing : Nat

structure Epsilon where
  pythonFloat : Bool
  finite : Bool
  value : ℝ

inductive Argument | nonTensor | tensor (metadata : TensorInfo)

def width (x : TensorInfo) := x.shape.getLastD 0

def pythonChecks (x w : TensorInfo) (eps : Epsilon) : List Bool :=
  [decide (x.layout=.strided ∧ w.layout=.strided),
   decide (x.dtype=.floating .fp32 ∧ w.dtype=.floating .fp32),
   decide (1≤x.shape.length ∧ w.shape.length=1),
   x.isContiguous && w.isContiguous,
   !(x.isNeg || x.isConj || w.isNeg || w.isConj),
   decide (x.device=w.device) && x.isCuda,
   decide (0<width x ∧ width x≤16384 ∧ w.numel=width x),
   decide (x.numel+16383<2^31),
   eps.pythonFloat && eps.finite && decide (0<eps.value),
   decide (x.numel=0 ∨ (x.dataPtr.toNat%16=0 ∧ w.dataPtr.toNat%16=0))]

def InputContract (x w : TensorInfo) (eps : Epsilon) : Prop :=
  (x.layout=.strided ∧ w.layout=.strided) ∧
  (x.dtype=.floating .fp32 ∧ w.dtype=.floating .fp32) ∧
  (1≤x.shape.length ∧ w.shape.length=1) ∧
  (x.isContiguous=Bool.true ∧ w.isContiguous=Bool.true) ∧
  ((x.isNeg=Bool.false ∧ x.isConj=Bool.false) ∧ w.isNeg=Bool.false) ∧ w.isConj=Bool.false ∧
  (x.device=w.device ∧ x.device.kind="cuda") ∧
  (0<width x ∧ width x≤16384 ∧ w.numel=width x) ∧
  x.numel+16383<2^31 ∧
  (eps.pythonFloat=Bool.true ∧ eps.finite=Bool.true ∧ 0<eps.value) ∧
  (x.numel=0 ∨ (x.dataPtr.toNat%16=0 ∧ w.dataPtr.toNat%16=0))

def Admitted (x w : TensorInfo) (eps : Epsilon) := (pythonChecks x w eps).all id=Bool.true

def check (x w : Argument) (eps : Epsilon) : Bool :=
  match x,w with
  | .tensor x,.tensor w => (pythonChecks x w eps).all id
  | _,_ => Bool.false

theorem python_checks_iff (x w : TensorInfo) (eps : Epsilon) :
    Admitted x w eps ↔ InputContract x w eps := by
  simp [Admitted,pythonChecks,InputContract,TensorMetadata.isCuda,and_assoc]

theorem non_tensor_rejected (w : Argument) (eps : Epsilon) :
    check .nonTensor w eps=Bool.false := by cases w <;> rfl

theorem shape_split (dims : List Nat) (rank : 1≤dims.length) :
    ∃ leading n,dims=leading++[n] ∧ dims.getLastD 0=n := by
  induction dims using List.reverseRecOn with
  | nil => simp at rank
  | append_singleton leading n _ => exact ⟨leading,n,rfl,by simp⟩

theorem request_exists (x w : TensorInfo) (eps : Epsilon) (ok : Admitted x w eps) :
    ∃ q : Host.Request,q.shape=x.shape ∧ q.columns=width x ∧ q.epsilon=eps.value ∧ q.size=x.numel := by
  rcases (python_checks_iff x w eps).mp ok with ⟨_,_,rank,_,_,_,_,widths,fit,ep,_⟩
  obtain ⟨leading,n,shape,last⟩:=shape_split x.shape rank.1
  have numel : x.numel=leading.prod*width x := by
    simp [TensorMetadata.numel,shape,width]
  refine ⟨⟨leading,width x,eps.value,widths.1,widths.2.1,ep.2.2,?_⟩,?_,rfl,rfl,?_⟩
  · simpa [numel] using fit
  · change leading++[width x]=x.shape
    change leading++[x.shape.getLastD 0]=x.shape
    rw [last]
    exact shape.symm
  · exact numel.symm

noncomputable def request (x w : TensorInfo) (eps : Epsilon) (ok : Admitted x w eps) : Host.Request :=
  (request_exists x w eps ok).choose

theorem request_facts (x w : TensorInfo) (eps : Epsilon) (ok : Admitted x w eps) :
    (request x w eps ok).shape=x.shape ∧ (request x w eps ok).columns=width x ∧
    (request x w eps ok).epsilon=eps.value ∧ (request x w eps ok).size=x.numel :=
  (request_exists x w eps ok).choose_spec

#axiomsClean python_checks_iff
#axiomsClean request_facts
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.Contract
