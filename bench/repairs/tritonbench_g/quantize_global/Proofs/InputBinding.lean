import bench.repairs.tritonbench_g.quantize_global.Proofs.Contract
import bench.repairs.tritonbench_g.quantize_global.Proofs.Native
import VeriTile.Torch.FP32
/-! Kernel-only original input binding. The observed finite flag is connected
to the actual input words below. No wrapper run or output property is assumed. -/
namespace QuantizeGlobalRepair.Direct
open VeriTile.Triton FP.Structural
noncomputable section

def inputValues [Inhabited α] (X : RegionName) (s : State α) (i : Nat) : α :=
  (s.mem X i).read .real

abbrev InputTensor := VeriTile.Torch.FP32.Tensor
abbrev InputTensorLaws := VeriTile.Torch.FP32.TensorLaws
abbrev Word := FP.Value .fp32

def view (x : InputTensor) (laws : InputTensorLaws x) (h : InputContract x.metadata finite) :=
  VeriTile.Torch.FP32.dataView x laws h.contiguous

def TensorBinding [Inhabited α] (x : InputTensor) (laws : InputTensorLaws x)
    (h : InputContract x.metadata finite) (read : Nat → Word) (interpret : Word → α)
    (X : RegionName) (initial : State α) : Prop :=
  ∀ i,i<x.metadata.numel → inputValues X initial i=interpret (read ((view x laws h).address i))

theorem original_logical_input [Inhabited α] (x : InputTensor) (laws : InputTensorLaws x)
    (h : InputContract x.metadata finite) (read : Nat → Word) (interpret : Word → α)
    (X : RegionName) (initial : State α) (binding : TensorBinding x laws h read interpret X initial)
    (i : Nat) (hi : i<x.metadata.numel) :
    inputValues X initial i=interpret (VeriTile.Torch.FP32.logicalValue read x laws i) := by
  rw [binding i hi]
  exact congrArg interpret (VeriTile.Torch.FP32.direct_read_eq_logical read x laws h.contiguous h.noNeg i hi)

theorem original_pointer (x : InputTensor) (laws : InputTensorLaws x)
    (h : InputContract x.metadata finite) (positive : 0<x.metadata.numel) :
    BitVec.ofNat 64 (view x laws h).base=x.dataPtr :=
  VeriTile.Torch.FP32.actual_pointer x laws h.contiguous positive

theorem source_input_pointer (x : InputTensor) (laws : InputTensorLaws x)
    (h : InputContract x.metadata finite) (i : Nat) (hi : i<x.metadata.numel) :
    (x.dataPtr+BitVec.ofNat 64 4*BitVec.ofNat 64 i).toNat=(view x laws h).address i := by
  rw [←original_pointer x laws h (by omega)]
  exact pointer_exact (view x laws h).base 4 x.metadata.numel i (by decide) (view x laws h).spanFits hi

/-- Meaning of torch.isfinite(...).all().item() on the original logical input.
Unlike an unconnected Boolean, this API observation refers to the same stored
FP32 words as the public kernel binding. -/
def FiniteObservation (x : InputTensor) (laws : InputTensorLaws x)
    (read : Nat → Word) (observed : Bool) : Prop :=
  0 < x.metadata.numel → (observed = Bool.true ↔ ∀ i, i < x.metadata.numel →
    (VeriTile.Torch.FP32.logicalValue read x laws i).toRat?.isSome)

theorem checked_finite (x : InputTensor) (laws : InputTensorLaws x)
    (observed : Bool) (h : InputContract x.metadata observed) (read : Nat → Word)
    (binding : FiniteObservation x laws read observed) (i : Nat) (hi : i < x.metadata.numel) :
    (VeriTile.Torch.FP32.logicalValue read x laws i).toRat?.isSome := by
  have yes : observed = Bool.true := h.finite.resolve_left (by omega)
  exact (binding (by omega)).mp yes i hi

/-- Construct the regional input snapshot from the actual pointer view.
This does not allocate/copy a runtime Tensor. Other regions retain scratch. -/
def bindInput [Inhabited α] (x : InputTensor) (laws : InputTensorLaws x)
    (h : InputContract x.metadata finite) (read : Nat → Word) (interpret : Word → α)
    (X : RegionName) (scratch : State α) : State α :=
  { scratch with mem := fun r i =>
      if r = X then .mk .real (interpret (read ((view x laws h).address i))) else scratch.mem r i }

theorem bindInput_binding [Inhabited α] (x : InputTensor) (laws : InputTensorLaws x)
    (h : InputContract x.metadata finite) (read : Nat → Word) (interpret : Word → α)
    (X : RegionName) (scratch : State α) :
    TensorBinding x laws h read interpret X (bindInput x laws h read interpret X scratch) := by
  intro i hi
  simp [TensorBinding, inputValues, bindInput]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end QuantizeGlobalRepair.Direct
