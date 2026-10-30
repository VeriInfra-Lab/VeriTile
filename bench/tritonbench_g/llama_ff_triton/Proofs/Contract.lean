import bench.tritonbench_g.llama_ff_triton.Proofs.Host
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.LlamaFF.Contract
open VeriTile.Torch
noncomputable section
open Classical

structure TensorInfo extends TensorMetadata where
  isConj : Bool
  object : Nat
  backing : Nat
  storageOffset : Nat
  strides : List Nat

inductive Argument | nonTensor | tensor (info : TensorInfo)
def batch (x : TensorInfo) := x.shape[0]?.getD 0
def sequence (x : TensorInfo) := x.shape[1]?.getD 0
def columns (x : TensorInfo) := x.shape[2]?.getD 0
def outputs (w : TensorInfo) := w.shape[0]?.getD 0
def shape (x w : TensorInfo) : Shape := ⟨1,batch x*sequence x,outputs w,columns x⟩
def useByte (w : TensorInfo) : Bool := decide (w.dtype=.other "int8")
def weightBytes (w : TensorInfo) : Nat := if useByte w then 1 else 2

def Layouts (x w v r : TensorInfo) : Prop :=
  x.layout=.strided ∧ w.layout=.strided ∧ v.layout=.strided ∧ r.layout=.strided
def DTypes (x w v r : TensorInfo) : Prop :=
  x.dtype=.other "float16" ∧ (w.dtype=.other "float16" ∨ w.dtype=.other "int8") ∧
    w.dtype=v.dtype ∧ v.dtype=r.dtype
def Ranks (x w v r : TensorInfo) : Prop :=
  x.shape.length=3 ∧ w.shape.length=2 ∧ v.shape.length=2 ∧ r.shape.length=1
def Devices (x w v r : TensorInfo) : Prop :=
  x.device.kind="cuda" ∧ w.device=x.device ∧ v.device=x.device ∧ r.device=x.device
def Dense (t : TensorInfo) : Prop :=
  t.isContiguous=Bool.true ∧ t.storageOffset=0 ∧ t.isNeg=Bool.false ∧ t.isConj=Bool.false
def Shapes (x w v r : TensorInfo) : Prop :=
  0<columns x ∧ 0<outputs w ∧ w.shape=[outputs w,columns x] ∧
    v.shape=[outputs w,columns x] ∧ r.shape=[columns x]
def Dimensions (x w : TensorInfo) : Prop :=
  (shape x w).M<2^31 ∧ (shape x w).N<2^31 ∧ (shape x w).K<2^31
def Grid (x w : TensorInfo) : Prop := tilesM (shape x w)*tilesN (shape x w)≤2^31-1
def Formed (x w : TensorInfo) : Prop :=
  2*((shape x w).M*(shape x w).K+formedK (shape x w))<2^63 ∧
  weightBytes w*((shape x w).N*(shape x w).K+formedK (shape x w))<2^63 ∧
  2*(tilesM (shape x w)*16*(shape x w).N+tilesN (shape x w)*16)<2^63
def Mixed (x w v r : TensorInfo) : Prop :=
  useByte w=Bool.true → x.backing≠w.backing ∧ x.backing≠v.backing ∧ x.backing≠r.backing
def Aligned (x w v r : TensorInfo) : Prop :=
  (shape x w).M=0 ∨ (x.dataPtr.toNat%16=0 ∧ w.dataPtr.toNat%16=0 ∧
    v.dataPtr.toNat%16=0 ∧ r.dataPtr.toNat%16=0)

def pythonChecks (x w v r : TensorInfo) : List Bool :=
  [decide (Layouts x w v r),decide (DTypes x w v r),decide (Ranks x w v r),
   decide (Devices x w v r),decide (Dense x ∧ Dense w ∧ Dense v ∧ Dense r),
   decide (Shapes x w v r),decide (Dimensions x w),decide (Grid x w),
   decide (Formed x w),decide (Mixed x w v r),decide (Aligned x w v r)]
structure InputContract (x w v r : TensorInfo) : Prop where
  layouts : Layouts x w v r
  dtypes : DTypes x w v r
  ranks : Ranks x w v r
  devices : Devices x w v r
  dense : Dense x ∧ Dense w ∧ Dense v ∧ Dense r
  shapes : Shapes x w v r
  dimensions : Dimensions x w
  grid : Grid x w
  formed : Formed x w
  mixed : Mixed x w v r
  aligned : Aligned x w v r

def Admitted (x w v r : TensorInfo) := (pythonChecks x w v r).all id=Bool.true
def check (x w v r : Argument) : Bool :=
  match x,w,v,r with
  | .tensor x,.tensor w,.tensor v,.tensor r => (pythonChecks x w v r).all id
  | _,_,_,_ => Bool.false

theorem python_checks_iff (x w v r : TensorInfo) : Admitted x w v r ↔ InputContract x w v r := by
  simp only [Admitted,pythonChecks,List.all_cons,List.all_nil,Bool.and_eq_true,decide_eq_true_eq,id_eq,and_true]
  constructor
  · rintro ⟨a,b,c,d,e,f,g,h,i,j,k⟩; exact ⟨a,b,c,d,e,f,g,h,i,j,k⟩
  · intro h; exact ⟨h.layouts,h.dtypes,h.ranks,h.devices,h.dense,h.shapes,h.dimensions,h.grid,h.formed,h.mixed,h.aligned⟩

theorem x_shape (ok : InputContract x w v r) : x.shape=[batch x,sequence x,columns x] := by
  have h:=ok.ranks.1
  rcases hs : x.shape with _|⟨a,tail⟩ <;> simp [hs] at h
  rcases tail with _|⟨b,tail⟩ <;> simp at h
  rcases tail with _|⟨c,tail⟩ <;> simp at h
  have ht : tail=[] := by simpa using h
  subst tail
  simp [batch,sequence,columns,hs]

theorem non_tensor_rejected (w v r : Argument) : check .nonTensor w v r=Bool.false := by
  cases w <;> cases v <;> cases r <;> rfl

#axiomsClean python_checks_iff
#axiomsClean x_shape
end
end VeriTile.Bench.Repairs.LlamaFF.Contract
