import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Safety
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.KernelTensor
open VeriTile.Triton FP.Structural Checked
open Classical
noncomputable section

/-- Actual Tensor handles and observations at the respective kernel calls.
Only metadata and ordinary Tensor coordinate laws occur in this binding. -/
structure Arguments (c : Shape) (U S V : Bool) (runtime : Runtime c U S V) where
  tensor : Role → VeriTile.Torch.FP32.Tensor
  laws : ∀ r, VeriTile.Torch.FP32.TensorLaws (tensor r)
  object : ∀ r, present U S V r=Bool.true → (tensor r).object=(runtime.tensor r).object
  cuda : ∀ r, present U S V r=Bool.true → (tensor r).metadata.isCuda=(runtime.tensor r).cuda
  device : ∀ r, present U S V r=Bool.true → (tensor r).metadata.device.index=(runtime.tensor r).device
  pointer : ∀ r, present U S V r=Bool.true → (tensor r).metadata.dataPtr.toNat=(runtime.tensor r).base
  shape : ∀ r, present U S V r=Bool.true → (tensor r).metadata.shape=(runtime.tensor r).shape
  contiguous : ∀ r, present U S V r=Bool.true → (tensor r).metadata.isContiguous=(runtime.tensor r).contiguous
  negative : ∀ r, present U S V r=Bool.true → (tensor r).metadata.isNeg=(runtime.tensor r).negative
  offset : ∀ r, present U S V r=Bool.true → (tensor r).storageOffset=(runtime.tensor r).storageOffset

def role (r : RegionName) : Role :=
  if r="x" then .x else if r="g" then .g else if r="h0" then .h0
  else if r="o" then .o else if r="ht" then .ht else if r="do" then .incoming
  else if r="dht" then .dht else if r="dx" then .dx else .dg

@[simp] theorem role_name (r : Role) : role (name r)=r := by cases r <;> decide

def recognized (r : RegionName) : Prop :=
  r="x" ∨ r="g" ∨ r="h0" ∨ r="o" ∨ r="ht" ∨ r="do" ∨ r="dht" ∨ r="dx" ∨ r="dg"

def atPointer {α : Type} (t : Arguments c U S V runtime)
    (memory : Role → Nat → α) (r : RegionName) (i : Nat) : α :=
  memory (role r) ((t.tensor (role r)).metadata.dataPtr.toNat+4*i)

/-- A typed view of the actual pointers. The memory parameter may observe
different entry times for forward inputs and backward cotangents. Forward
never reads do/dht; backward never writes its saved inputs. -/
def lift {α : Type} (t : Arguments c U S V runtime)
    (memory : Role → Nat → α) (other : State α) : State α :=
  {other with mem := fun r i => if recognized r then .mk .real (atPointer t memory r i) else other.mem r i}

theorem lift_read {α : Type} [Inhabited α] (t : Arguments c U S V runtime)
    (memory : Role → Nat → α) (other : State α) (r : Role) (i : Nat) :
    Execution.read (lift t memory other) (name r) i=
      memory r ((t.tensor r).metadata.dataPtr.toNat+4*i) := by
  have h : recognized (name r) := by cases r <;> simp [recognized,name]
  change (if recognized (name r) then .mk .real (atPointer t memory (name r) i)
    else other.mem (name r) i).read .real=_
  rw [if_pos h,Cell.read_same]
  simp only [atPointer,role_name]

theorem facts (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (accepted : check c U S V runtime=Bool.true) (r : Role)
    (active : present U S V r=Bool.true) :
    (runtime.tensor r).shape=Checked.shape c r ∧
    (runtime.tensor r).contiguous=Bool.true ∧ (runtime.tensor r).negative=Bool.false := by
  have h := accepted
  simp only [check,Bool.and_eq_true] at h
  have member : r∈roles := by cases r <;> simp [roles]
  have a := List.all_eq_true.mp h.1.1 r member
  simp only [active,↓reduceIte,decide_eq_true_eq] at a
  exact ⟨a.1,a.2.2.2.2.2.2.1,a.2.2.2.2.2.2.2.1⟩

theorem tensor_facts (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (accepted : check c U S V runtime=Bool.true) (t : Arguments c U S V runtime)
    (r : Role) (active : present U S V r=Bool.true) :
    (t.tensor r).metadata.numel=extent c r ∧
    (t.tensor r).metadata.isContiguous=Bool.true ∧ (t.tensor r).metadata.isNeg=Bool.false := by
  have h := facts c U S V runtime accepted r active
  refine ⟨?_,by rw [t.contiguous r active,h.2.1],by rw [t.negative r active,h.2.2]⟩
  by_cases hr : stateRole r=Bool.true <;>
    simp [VeriTile.Torch.TensorMetadata.numel,t.shape r active,h.1,Checked.shape,extent,
      Shape.rows,Shape.slots,Shape.BH,hr,Nat.mul_assoc]

/-- Contiguous windows retain nonzero storage offsets; no zero-offset premise. -/
theorem coordinates (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (accepted : check c U S V runtime=Bool.true) (t : Arguments c U S V runtime)
    (r : Role) (active : present U S V r=Bool.true) (i : Nat) (inside : i<extent c r) :
    (t.tensor r).physicalIndex i=(t.tensor r).storageOffset+i ∧
    (t.tensor r).storage.address ((t.tensor r).physicalIndex i)=
      (t.tensor r).metadata.dataPtr.toNat+4*i := by
  have h := tensor_facts c U S V runtime accepted t r active
  have hi : i<(t.tensor r).metadata.numel := by rw [h.1];exact inside
  have index := (t.laws r).contiguous h.2.1 i hi
  refine ⟨index,?_⟩
  have ptr := (t.laws r).pointer (by omega)
  have cap := VeriTile.Torch.FP32.contiguous_capacity (t.tensor r) (t.laws r) h.2.1 (by omega)
  have span := (t.tensor r).storage.spanFits
  have bound : (t.tensor r).storage.address (t.tensor r).storageOffset<2^64 := by
    dsimp [VeriTile.Torch.FP32.Storage.address];omega
  have pointer : (t.tensor r).metadata.dataPtr.toNat=(t.tensor r).storage.address (t.tensor r).storageOffset := by
    simpa only [VeriTile.Torch.FP32.Tensor.dataPtr,BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound]
      using congrArg BitVec.toNat ptr
  rw [index,pointer]
  simp [VeriTile.Torch.FP32.Storage.address,Nat.mul_add,Nat.add_assoc]

theorem logical_read (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (accepted : check c U S V runtime=Bool.true) (t : Arguments c U S V runtime)
    (memory : Role → Nat → VeriTile.Torch.FP32.Word)
    (r : Role) (active : present U S V r=Bool.true) (i : Nat) (inside : i<extent c r) :
    memory r ((t.tensor r).metadata.dataPtr.toNat+4*i)=
      VeriTile.Torch.FP32.logicalValue (memory r) (t.tensor r) (t.laws r) i := by
  rw [VeriTile.Torch.FP32.logicalValue,(tensor_facts c U S V runtime accepted t r active).2.2,
    (coordinates c U S V runtime accepted t r active i inside).2]
  rfl

/-- Logical observation of a source-produced typed result through the actual
Tensor coordinates. It reads the interpreter's final memory, never a formula. -/
def result {α : Type} [Inhabited α] (t : Arguments c U S V runtime)
    (q : State α) (r : Role) (i : Nat) : α :=
  Execution.read q (name r) ((t.tensor r).physicalIndex i-(t.tensor r).storageOffset)

theorem result_eq {α : Type} [Inhabited α] (c : Shape) (U S V : Bool)
    (runtime : Runtime c U S V) (accepted : check c U S V runtime=Bool.true)
    (t : Arguments c U S V runtime) (q : State α) (r : Role)
    (active : present U S V r=Bool.true) (i : Nat) (inside : i<extent c r) :
    result t q r i=Execution.read q (name r) i := by
  unfold result
  rw [(coordinates c U S V runtime accepted t r active i inside).1,Nat.add_sub_cancel_left]

/-- A per-argument raw-word observation of the executed typed memory. This
is not an independently populated output buffer or a target-formula callback. -/
def resultWords {α : Type} [Inhabited α] (evaluate : α → BitVec 32)
    (t : Arguments c U S V runtime) (q : State α) (r : Role) (address : Nat) :
    VeriTile.Torch.FP32.Word :=
  ⟨evaluate (Execution.read q (name r) ((address-(t.tensor r).metadata.dataPtr.toNat)/4))⟩

theorem result_logical {α : Type} [Inhabited α] (c : Shape) (U S V : Bool)
    (runtime : Runtime c U S V) (accepted : check c U S V runtime=Bool.true)
    (evaluate : α → BitVec 32) (t : Arguments c U S V runtime) (q : State α) (r : Role)
    (active : present U S V r=Bool.true) (i : Nat) (inside : i<extent c r) :
    VeriTile.Torch.FP32.logicalValue (resultWords evaluate t q r) (t.tensor r) (t.laws r) i=
      (⟨evaluate (result t q r i)⟩ : VeriTile.Torch.FP32.Word) := by
  rw [VeriTile.Torch.FP32.logicalValue,(tensor_facts c U S V runtime accepted t r active).2.2,
    (coordinates c U S V runtime accepted t r active i inside).2,
    result_eq c U S V runtime accepted t q r active i inside]
  simp [resultWords]

#axiomsClean coordinates
#axiomsClean logical_read
#axiomsClean result_eq
#axiomsClean result_logical
#specNonCircular Arguments avoiding [Execution.gridRun,state,dx,dg]
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.KernelTensor
