import bench.repairs.tritonbench_g.matmul_dequant_int4.Contract
import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.WholeNative

namespace VeriTile.Bench.Repairs.MatmulDequantInt4.Direct
open Host Physical
noncomputable section
open Classical
set_option maxHeartbeats 4000000

abbrev Heap := Nat → Physical.Byte

/-- A view of original physical bytes, with no data movement. -/
def bind (base : Physical.Allocation → Nat) (heap : Heap) : Bytes := fun r off => heap (base r+off)

def dequantBase (x : PackedArguments) (output : Host.Tensor) : Physical.Allocation → Nat
  | .packed .weight => x.weight.base
  | .packed .zero => x.zero.base
  | .value .scale => x.scale.base
  | .value .dequant => output.base
  | _ => 0

def s1Base (x : Host.Arguments) (a : Host.Allocations x) : Physical.Allocation → Nat
  | .packed .weight => x.packed.weight.base
  | .packed .zero => x.packed.zero.base
  | .value .scale => x.packed.scale.base
  | .value .a => x.input.base
  | .value .output => (x.output a.freshOutput).base
  | .value .dequant => a.scratch.base
  | _ => 0

def quantBase (input : Host.Tensor) : Physical.Allocation → Nat
  | .value .weight => input.base
  | _ => 0

def writeSpan (base bytes : Nat) (data : Nat → Byte) (heap : Heap) (address : Nat) : Byte :=
  if base≤address ∧ address<base+bytes then data (address-base) else heap address

theorem writeSpan_byte (base bytes : Nat) (data : Nat → Byte) (heap : Heap)
    (off : Nat) (bound : off<bytes) : writeSpan base bytes data heap (base+off)=data off := by
  simp [writeSpan,show base≤base+off ∧ base+off<base+bytes by omega]

theorem writeSpan_frame (base bytes : Nat) (data : Nat → Byte) (heap : Heap)
    (address : Nat) (outside : ¬(base≤address ∧ address<base+bytes)) :
    writeSpan base bytes data heap address=heap address := by simp [writeSpan,outside]

theorem writeSpan_disjoint (out input : Host.Tensor) (apart : Host.Disjoint out input)
    (bytes : Nat) (fits : bytes≤out.storageBytes) (data : Nat → Byte) (heap : Heap)
    (off : Nat) (bound : off < input.storageBytes) :
    writeSpan out.base bytes data heap (input.base+off)=heap (input.base+off) := by
  apply writeSpan_frame
  unfold Host.Disjoint at apart
  omega

/-- Public schedules instantiate every masked lane of the actual selected
configuration; callers supply no coverage, permutation, or bounds proof. -/
def dequantConfig (choice : Fin 8) : Source.DequantConfig := Source.dequantConfigs[choice.val]'(by
  simpa [Source.dequantConfigs] using choice.isLt)
theorem dequantConfig_mem (choice : Fin 8) : dequantConfig choice∈Source.dequantConfigs :=
  List.getElem_mem _
def dequantGrid (s : Shape) (choice : Fin 8) : Grid s.k s.n :=
  rectangularGrid s.k s.n (dequantConfig choice).bk (dequantConfig choice).bn
    (Source.dequant_valid _ (dequantConfig_mem choice)).1
    (Source.dequant_valid _ (dequantConfig_mem choice)).2.1
def dequantOrder (s : Shape) (choice : Fin 8) := (dequantGrid s choice).schedule
def trialOrders (s : Shape) (choices : List (Fin 8)) := choices.map (dequantOrder s)
def outputGrid (s : Shape) : Grid s.m s.n := rectangularGrid s.m s.n 1 1 (by decide) (by decide)
def outputOrder (s : Shape) := (outputGrid s).schedule
theorem dequant_complete (s : Shape) (choice : Fin 8) : (dequantGrid s choice).Complete (dequantOrder s choice) := List.Perm.refl _
theorem output_complete (s : Shape) : (outputGrid s).Complete (outputOrder s) := List.Perm.refl _

theorem trial_active (s : Shape) (choices : List (Fin 8)) (order : List Work)
    (used : order∈trialOrders s choices) (w : Work) (member : w∈order) : w.Active s.k s.n := by
  obtain ⟨choice,_,rfl⟩ := List.mem_map.mp used
  exact (Grid.schedule_mem _ w).mp member

structure TensorResult where
  tensor : Host.Tensor
  heap : Heap
  launchDevice : Nat
  callerDevice : Nat

def globalDequant (codec : Codec α) (x : PackedArguments) (dtype : Host.DType)
    (a : DequantAllocation x dtype) (one : α) (ops : DequantOps α) (choices : List (Fin 8))
    (chosen : Fin 8) (heap : Heap) (caller : Nat) : TensorResult :=
  let localResult := dequantHost codec x dtype a one ops (trialOrders x.shape choices)
    (dequantOrder x.shape chosen) (bind (dequantBase x a.tensor) heap) caller
  ⟨localResult.tensor,writeSpan a.tensor.base (codec.width*x.shape.k*x.shape.n)
    (localResult.data (.value .dequant)) heap,localResult.launchDevice,localResult.callerDevice⟩

def globalS1 (x : Host.Arguments) (a : Host.Allocations x) (I : Native.Instructions)
    (plan : Library.Selected x.packed.shape) (choices : List (Fin 8)) (chosen : Fin 8)
    (heap : Heap) (caller : Nat) : TensorResult :=
  let localResult := Host.call x a I plan (trialOrders x.packed.shape choices) (dequantOrder x.packed.shape chosen)
    (outputOrder x.packed.shape) (bind (s1Base x a) heap) caller
  ⟨localResult.tensor,writeSpan (x.output a.freshOutput).base (2*x.packed.shape.m*x.packed.shape.n)
    (localResult.data (.value .output))
    (writeSpan a.scratch.base (2*x.packed.shape.k*x.packed.shape.n) (localResult.data (.value .dequant)) heap),
    localResult.launchDevice,localResult.callerDevice⟩

theorem global_dequant_byte (codec : Codec α) (x : PackedArguments) (dtype : Host.DType)
    (a : DequantAllocation x dtype) (one : α) (ops : DequantOps α) (choices : List (Fin 8))
    (chosen : Fin 8) (heap : Heap) (caller k n : Nat) (hk : k<x.shape.k) (hn : n<x.shape.n)
    (i : Fin codec.width) :
    (globalDequant codec x dtype a one ops choices chosen heap caller).heap
      (a.tensor.base+(codec.width*(k*x.shape.n+n)+i.val))=
    codec.encode (dequantValue x.shape ops (load codec (bind (dequantBase x a.tensor) heap)) ⟨k,n⟩) i := by
  change writeSpan _ _ _ _ _=_
  rw [writeSpan_byte]
  · exact Pipeline.dequant_result codec x.shape one ops (trialOrders x.shape choices)
      (dequantGrid x.shape chosen) _ (dequant_complete x.shape chosen) _ k n hk hn i
  · have h := rowMajor_bound _ _ _ _ hk hn
    have := i.isLt
    nlinarith

theorem global_s1_byte (x : Host.Arguments) (a : Host.Allocations x) (I : Native.Instructions)
    (plan : Library.Selected x.packed.shape) (choices : List (Fin 8)) (chosen : Fin 8)
    (heap : Heap) (caller r n : Nat) (hr : r<x.packed.shape.m) (hn : n<x.packed.shape.n) (i : Fin 2) :
    (globalS1 x a I plan choices chosen heap caller).heap
      ((x.output a.freshOutput).base+(2*(r*x.packed.shape.n+n)+i.val))=
    encode16 (S1.reference x.packed.shape (Native.dequant16 I) (Native.library I) plan
      (load halfCodec (bind (s1Base x a) heap)) r n) i := by
  change writeSpan _ _ _ _ _=_
  rw [writeSpan_byte]
  · exact returned_bytes x a I plan (trialOrders x.packed.shape choices)
      (dequantGrid x.packed.shape chosen) (outputGrid x.packed.shape) _ _
      (dequant_complete x.packed.shape chosen) (output_complete x.packed.shape) _ caller r n hr hn i
  · have h := rowMajor_bound _ _ _ _ hr hn
    have := i.isLt
    nlinarith

#axiomsClean writeSpan_byte
#axiomsClean writeSpan_disjoint
#axiomsClean global_dequant_byte
#axiomsClean global_s1_byte
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulDequantInt4.Direct
