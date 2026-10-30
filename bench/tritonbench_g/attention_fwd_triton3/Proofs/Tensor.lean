import bench.tritonbench_g.attention_fwd_triton3.Contract
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.TensorBinding
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open Host
noncomputable section

inductive Role where
  | query | key | value | output | center | denominator
  deriving DecidableEq

def region : Role → RegionName
  | .query => "Q" | .key => "K" | .value => "V"
  | .output => "O" | .center => "M" | .denominator => "L"

/-- Actual shared Tensor, together with the additional ordinary Python API
observations queried by this wrapper. -/
structure Payload where
  tensor : Torch.FP32.Tensor
  strides : List Nat
  ordinary : Bool
  conjugate : Bool

def offset : List Nat → List Nat → Nat → Nat
  | _ :: shape,stride :: strides,i => i/shape.prod*stride+offset shape strides (i%shape.prod)
  | _,_,_ => 0

theorem offset_zero (shape strides : List Nat) : offset shape strides 0=0 := by
  induction shape generalizing strides with
  | nil => rfl
  | cons dim shape ih => cases strides <;> simp [offset,ih]

/-- The strided Tensor API relation is only about metadata and storage
coordinates. It contains no kernel result, floating law or execution premise. -/
structure Laws (x : Payload) : Prop where
  tensor : Torch.FP32.TensorLaws x.tensor
  strides : x.strides.length=x.tensor.metadata.shape.length
  physical : ∀ i,i<x.tensor.metadata.numel →
    x.tensor.physicalIndex i=x.tensor.storageOffset+offset x.tensor.metadata.shape x.strides i

def descriptor (x : Payload) : Checked.Descriptor where
  shape := x.tensor.metadata.shape
  strides := x.strides
  device := x.tensor.metadata.device.index
  ordinary := x.ordinary
  cuda := x.tensor.metadata.isCuda
  fp32 := x.tensor.metadata.isFP32 && x.tensor.metadata.isStrided
  resolved := !x.tensor.metadata.isNeg && !x.conjugate
  contiguous := x.tensor.metadata.isContiguous

abbrev Request := Role → Payload

def arguments (r : Request) : Checked.Arguments :=
  ⟨descriptor (r .query),descriptor (r .key),descriptor (r .value),
    descriptor (r .output),descriptor (r .center),descriptor (r .denominator)⟩

theorem element_address (x : Payload) (laws : Laws x) (i : Nat)
    (hi : i<x.tensor.metadata.numel) :
    x.tensor.storage.address (x.tensor.physicalIndex i)=
      x.tensor.dataPtr.toNat+4*offset x.tensor.metadata.shape x.strides i := by
  have zero : 0<x.tensor.metadata.numel := by omega
  have first := laws.tensor.inBounds 0 zero
  rw [laws.physical 0 zero,offset_zero,Nat.add_zero] at first
  have span := x.tensor.storage.spanFits
  have bound : x.tensor.storage.address x.tensor.storageOffset<2^64 := by
    unfold Torch.FP32.Storage.address
    omega
  rw [laws.physical i hi,laws.tensor.pointer zero]
  simp only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound,Torch.FP32.Storage.address]
  omega

theorem logical_read (read : Nat → Torch.FP32.Word) (x : Payload) (laws : Laws x)
    (resolved : x.tensor.metadata.isNeg=Bool.false) (i : Nat)
    (hi : i<x.tensor.metadata.numel) :
    Torch.FP32.logicalValue read x.tensor laws.tensor i=
      read (x.tensor.dataPtr.toNat+4*offset x.tensor.metadata.shape x.strides i) := by
  simp only [Torch.FP32.logicalValue,resolved,Bool.false_eq_true,if_false]
  rw [element_address x laws i hi]

/-- One admission record binds the actual six caller Tensors to the same
checked pointers and allocation used by the execution theorem. -/
structure Admission (c : Call) (A : FlatAlloc) (p : Checked.Pointers A) (r : Request) : Prop where
  checked : Contract.Admission c A p (arguments r)
  laws : ∀ role,Laws (r role)
  member : ∀ role,region role∈A.regions
  pointer : ∀ role,p.byteBase (region role)=(r role).tensor.dataPtr.toNat

theorem admitted_aligned (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : Request) (admission : Admission c A p r) (role : Role) :
    (r role).tensor.dataPtr.toNat%4=0 := by
  have checks := admission.checked.setup
  simp only [Checked.setupChecks,Bool.and_eq_true] at checks
  have pointers := checks.2
  simp only [Checked.pointerChecks,Bool.and_eq_true,List.all_eq_true,decide_eq_true_eq] at pointers
  have aligned := (pointers.2 (region role) (admission.member role)).1
  rwa [admission.pointer role] at aligned

theorem admitted_resolved (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : Request) (admission : Admission c A p r) (role : Role) :
    (r role).tensor.metadata.isNeg=Bool.false := by
  have checks := admission.checked.metadata
  simp only [Checked.metadataChecks,decide_eq_true_eq] at checks
  have member : descriptor (r role)∈[(arguments r).q,(arguments r).k,(arguments r).v,
      (arguments r).o,(arguments r).m,(arguments r).l] := by
    cases role <;> simp [arguments]
  have resolved := (checks.1 _ member).2.2.2.1
  simp only [descriptor,Bool.and_eq_true,Bool.not_eq_true'] at resolved
  exact resolved.1

theorem allocation_address (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : Request) (admission : Admission c A p r) (role : Role) (i : Nat)
    (hi : i<(r role).tensor.metadata.numel) :
    (r role).tensor.storage.address ((r role).tensor.physicalIndex i)=
      4*A.addr (region role) (offset (r role).tensor.metadata.shape (r role).strides i) := by
  rw [element_address _ (admission.laws role) i hi]
  have hb := p.binding (region role) (admission.member role)
  rw [admission.pointer role] at hb
  have reconstruct := Nat.mod_add_div (r role).tensor.dataPtr.toNat 4
  have ha := admitted_aligned c A p r admission role
  simp only [FlatAlloc.addr,hb]
  omega

theorem all_six_inputs (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : Request) (admission : Admission c A p r) (read : Nat → Torch.FP32.Word)
    (role : Role)
    (i : Nat) (hi : i<(r role).tensor.metadata.numel) :
    Torch.FP32.logicalValue read (r role).tensor (admission.laws role).tensor i=
      read (4*A.addr (region role) (offset (r role).tensor.metadata.shape (r role).strides i)) := by
  simp only [Torch.FP32.logicalValue,admitted_resolved c A p r admission role,Bool.false_eq_true,if_false]
  rw [allocation_address c A p r admission role i hi]

#specNonCircular Laws avoiding [FP.Structural.exec,FP.Structural.run]
#specNonCircular Admission avoiding [FP.Structural.exec,FP.Structural.run]
#axiomsClean element_address
#axiomsClean logical_read
#axiomsClean all_six_inputs
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.TensorBinding
