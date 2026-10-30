import VeriTile.Torch.FP32
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.KernelDomain

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.KernelBinding
open VeriTile.Torch VeriTile.Triton FP.Structural
open KernelDomain
noncomputable section
abbrev Word := FP.Value .fp32
abbrev Heap := Nat → Word

/-- Facts of each ordinary CUDA FP32 allocation made by this wrapper. These
bind the actual Tensor/storage/pointer; they are not added Python guards and
contain no kernel result. Physical heap indices below count FP32 words. -/
structure Allocated (shape : List Nat) where
  tensor : FP32.Tensor
  laws : FP32.TensorLaws tensor
  shape_eq : tensor.metadata.shape=shape
  contiguous : tensor.metadata.isContiguous=Bool.true
  noNeg : tensor.metadata.isNeg=Bool.false
  cuda : tensor.metadata.device.kind="cuda"
  aligned : 0<tensor.metadata.numel → tensor.dataPtr.toNat%4=0
  dimensions : ∀ n∈shape,n<2^63
  capacity : tensor.storage.elements*4<2^63

def Allocated.view (x : Allocated shape) : FP32.DataView := FP32.dataView x.tensor x.laws x.contiguous
def Allocated.base (x : Allocated shape) : Nat := x.view.base/4
def Allocated.count (x : Allocated shape) : Nat := x.tensor.metadata.numel

def Allocated.observe (x : Allocated shape) (heap : Nat → α) (i : Nat) : α :=
  heap (x.tensor.storage.address (x.tensor.physicalIndex i)/4)
def Allocated.value (x : Allocated shape) (heap : Heap) (i : Nat) : Word :=
  FP32.logicalValue (fun byte => heap (byte/4)) x.tensor x.laws i

@[simp] theorem Allocated.count_eq (x : Allocated shape) : x.count=shape.prod := by
  simp [Allocated.count,TensorMetadata.numel,x.shape_eq]

theorem Allocated.base_exact (x : Allocated shape) : 4*x.base=x.view.base := by
  have h := FP32.dataView_aligned x.tensor x.laws x.contiguous x.aligned
  exact Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero h)

theorem Allocated.observe_direct (x : Allocated shape) (heap : Nat → α)
    (i : Nat) (inside : i<x.count) : x.observe heap i=heap (x.base+i) := by
  have idx := x.laws.contiguous x.contiguous i inside
  have base := x.base_exact
  have nonempty : x.tensor.metadata.numel≠0 := by change i<x.tensor.metadata.numel at inside; omega
  simp only [Allocated.view,FP32.dataView,FP32.DataView.base,nonempty,if_false,
    FP32.Storage.address] at base
  simp only [Allocated.observe,idx,FP32.Storage.address,Nat.mul_add]
  rw [←Nat.add_assoc,←base]
  congr 1
  omega

theorem Allocated.value_observe (x : Allocated shape) (heap : Heap) (i : Nat) :
    x.value heap i=x.observe heap i := by
  simp [Allocated.value,FP32.logicalValue,x.noNeg,Allocated.observe]

theorem Allocated.direct (x : Allocated shape) (heap : Heap) (i : Nat) (inside : i<x.count) :
    x.value heap i=heap (x.base+i) := by
  rw [x.value_observe,x.observe_direct heap i inside]

theorem Allocated.actual_pointer (x : Allocated shape) (nonempty : 0<x.count) :
    x.tensor.dataPtr=BitVec.ofNat 64 (4*x.base) := by
  rw [x.base_exact]
  exact (FP32.actual_pointer x.tensor x.laws x.contiguous nonempty).symm

theorem Allocated.span (x : Allocated shape) : 4*x.base+x.count*4≤2^64 := by
  rw [x.base_exact]
  simpa [Allocated.view,Allocated.count,FP32.dataView,Nat.mul_comm] using x.view.spanFits

theorem Allocated.extent (x : Allocated shape) : x.count*4<2^63 := by
  by_cases empty : x.count=0
  · rw [empty]
    decide
  · have fit := FP32.contiguous_capacity x.tensor x.laws x.contiguous (Nat.pos_of_ne_zero empty)
    have cap := x.capacity
    change x.tensor.storageOffset+x.count≤x.tensor.storage.elements at fit
    omega

/-- Exact Tensor shapes are indexed in the types. Output separation protects
both complete live input storages. A and B need not be mutually disjoint. -/
structure Binding (a : Arguments) (x : Allocated [a.M,a.K])
    (y : Allocated [a.M,a.N,a.K]) (z : Allocated [a.M,a.N]) : Prop where
  deviceA : x.tensor.metadata.device=z.tensor.metadata.device
  deviceB : y.tensor.metadata.device=z.tensor.metadata.device
  separateA : x.tensor.storage.base/4+x.tensor.storage.elements≤z.base ∨
    z.base+z.count≤x.tensor.storage.base/4
  separateB : y.tensor.storage.base/4+y.tensor.storage.elements≤z.base ∨
    z.base+z.count≤y.tensor.storage.base/4

/-- This constructor derives all signed-int64 limits from the actual Torch
allocation capacities/dimensions, rather than imposing an i32-safe shape. -/
def storage (a : Arguments) (x : Allocated [a.M,a.K])
    (y : Allocated [a.M,a.N,a.K]) (z : Allocated [a.M,a.N]) : Index64.StorageLimits (parameters a) where
  M := x.dimensions a.M (by simp)
  N := y.dimensions a.N (by simp)
  K := x.dimensions a.K (by simp)
  A := by simpa [Nat.mul_assoc] using x.extent
  B := by simpa [Nat.mul_assoc] using y.extent
  output := by simpa [Nat.mul_assoc] using z.extent

def integerAgreement (a : Arguments) (native : LaunchABI a)
    (x : Allocated [a.M,a.K]) (y : Allocated [a.M,a.N,a.K]) (z : Allocated [a.M,a.N]) :
    Whole.IntegerAgreement (parameters a) :=
  Whole.launch_integer_agreement (launch a (fun _ _ => .mk .real (FP.Value.zero .fp32)))
    (storage a x y z) (launch_abi a native _)

/-- Realization of the final typed output cells in the actual output window.
All other physical FP32 words retain their old values. This is a typed-word
memory binding, not a claim about a byte codec. -/
def physical (z : Allocated shape) (before : Nat → α) (values : Nat → α) (addr : Nat) : α :=
  if z.base≤addr ∧ addr<z.base+z.count then values (addr-z.base) else before addr

theorem physical_read (z : Allocated shape) (before : Nat → α) (values : Nat → α)
    (i : Nat) (inside : i<z.count) : physical z before values (z.base+i)=values i := by
  rw [physical,if_pos (show z.base≤z.base+i ∧ z.base+i<z.base+z.count by omega)]
  congr 1
  omega

theorem physical_frame (z : Allocated shape) (before : Nat → α) (values : Nat → α)
    (addr : Nat) (outside : addr<z.base ∨ z.base+z.count≤addr) :
    physical z before values addr=before addr := by
  rw [physical,if_neg (show ¬(z.base≤addr ∧ addr<z.base+z.count) by omega)]

theorem output_observe (z : Allocated shape) (before : Nat → α) (values : Nat → α)
    (i : Nat) (inside : i<z.count) : z.observe (physical z before values) i=values i := by
  rw [z.observe_direct _ i inside,physical_read z before values i inside]

theorem output_value (z : Allocated shape) (before : Heap) (values : Nat → Word)
    (i : Nat) (inside : i<z.count) : z.value (physical z before values) i=values i := by
  rw [z.value_observe,output_observe z before values i inside]

theorem input_storage_frame (z : Allocated shape) (input : FP32.Storage)
    (sep : input.base/4+input.elements≤z.base ∨ z.base+z.count≤input.base/4)
    (before : Nat → α) (values : Nat → α) (i : Nat) (inside : i<input.elements) :
    physical z before values (input.base/4+i)=before (input.base/4+i) := by
  apply physical_frame
  omega

#axiomsClean Allocated.direct
#axiomsClean Allocated.actual_pointer
#axiomsClean storage
#axiomsClean integerAgreement
#axiomsClean output_observe
#axiomsClean input_storage_frame
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.KernelBinding
