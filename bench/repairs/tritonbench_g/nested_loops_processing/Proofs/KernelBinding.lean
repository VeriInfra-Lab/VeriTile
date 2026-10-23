import bench.repairs.tritonbench_g.nested_loops_processing.Contract
import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Execution
import VeriTile.Torch.Tensor
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.KernelBinding
open VeriTile.Triton FP.Structural
open VeriTile.Torch
open Shared.Address Shared.Execution
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section

/-- Exact Python dimension branch; bool values are excluded by the two
actual-int observations before any allocation or launch. -/
theorem dimensions_iff (rowIsInt colIsInt : Bool) (rows cols : Int) :
    Admission.dimensions rowIsInt colIsInt rows cols = Bool.true ↔
      rowIsInt = Bool.true ∧ colIsInt = Bool.true ∧
      0 ≤ rows ∧ rows < 2^63 ∧ 0 ≤ cols ∧ cols < 2^63 ∧ rows*cols ≤ 2^62 := by
  simp [Admission.dimensions, and_assoc]

/-- Ordinary constructor/API observations on a contiguous int32 allocation.
This carries only the pre-launch metadata, never a computed-output premise. -/
structure Buffer (rows cols : Nat) where
  metadata : TensorMetadata
  shape : metadata.shape = [rows, cols]
  dtype : metadata.dtype = .other "int32"
  layout : metadata.layout = .strided
  cuda : metadata.device.kind = "cuda"
  contiguous : metadata.isContiguous = Bool.true
  /-- Actual stride queries, fixed by the arange/reshape or zeros constructor. -/
  stride : Nat → Nat
  stride_zero : stride 0 = cols
  stride_one : stride 1 = 1
  pointerChecked : Admission.buffer (rows*cols) metadata.dataPtr.toNat = Bool.true

def Buffer.dense (b : Buffer rows cols) : Dense (rows*cols) where
  base := b.metadata.dataPtr.toNat
  finite := by
    have h := b.pointerChecked
    simp only [Admission.buffer, decide_eq_true_eq] at h
    simpa [Nat.mul_comm] using h.2
  aligned := by
    have h := b.pointerChecked
    simp only [Admission.buffer, decide_eq_true_eq] at h
    exact h.1

theorem Buffer.numel (b : Buffer rows cols) : b.metadata.numel = rows*cols := by
  simp [TensorMetadata.numel, b.shape]

theorem Buffer.pointer (b : Buffer rows cols) : b.dense.base = b.metadata.dataPtr.toNat := rfl

/-- The wrapper's fresh buffers are bound to one pre-launch physical memory.
Arange and zero contents are constructor API facts, not launch postconditions. -/
structure Binding (api : API Word Store Byte) (rows cols : Nat) (memory : Store) where
  dimensionChecked : Admission.dimensions Bool.true Bool.true (rows : Int) (cols : Int) = Bool.true
  input : Buffer rows cols
  output : Buffer rows cols
  sameDevice : output.metadata.device = input.metadata.device
  fresh : Disjoint output.dense input.dense
  inputArange : ∀ i, i<rows*cols → api.read memory (input.dense.address i) = arangeWord i
  outputZero : ∀ i, i<rows*cols → api.read memory (output.dense.address i) = 0

def Binding.placement {api : API Word Store Byte} {memory : Store}
    (b : Binding api rows cols memory) : Placement (rows*cols) :=
  ⟨b.input.dense, b.output.dense, b.fresh⟩

/-- The actual typed kernel input is obtained from the bound physical words.
Unrelated regions and all control metadata are inherited from the context. -/
def Binding.state {api : API Word Store Byte} {memory : Store}
    (b : Binding api rows cols memory) (context : State α) : State α :=
  { context with mem := fun r i ↦
      if r = "input" then .mk .int ((api.read memory (b.input.dense.address i)).toInt)
      else if r = "output" then .mk .int ((api.read memory (b.output.dense.address i)).toInt)
      else context.mem r i }

@[simp] theorem Binding.input_cell {api : API Word Store Byte} {memory : Store}
    (b : Binding api rows cols memory) (context : State α) (i : Nat) (hi : i<rows*cols) :
    (b.state context).mem "input" i = .mk .int (signedValue (arangeWord i)) := by
  simp [Binding.state, b.inputArange i hi, signedValue]

@[simp] theorem Binding.output_cell {api : API Word Store Byte} {memory : Store}
    (b : Binding api rows cols memory) (context : State α) (i : Nat) (hi : i<rows*cols) :
    (b.state context).mem "output" i = .mk .int 0 := by
  simp [Binding.state, b.outputZero i hi]

/-- Bounds apply to every formed pointer, including inactive masked lanes. -/
theorem Binding.pointer_bytes {api : API Word Store Byte} {memory : Store}
    (b : Binding api rows cols memory) (positive : 0<rows*cols)
    (raw byte : Nat) (small : byte<4) :
    b.input.dense.contains (b.input.dense.address (clamp (rows*cols) raw)+byte) ∧
    b.input.dense.address (clamp (rows*cols) raw)+byte<2^64 ∧
    b.output.dense.contains (b.output.dense.address (clamp (rows*cols) raw)+byte) ∧
    b.output.dense.address (clamp (rows*cols) raw)+byte<2^64 := by
  exact ⟨(all_pointer_bytes b.input.dense positive raw byte small).1,
    (all_pointer_bytes b.input.dense positive raw byte small).2,
    all_pointer_bytes b.output.dense positive raw byte small⟩

end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.KernelBinding

#guard_msgs (drop info) in
#auditModuleAxioms
