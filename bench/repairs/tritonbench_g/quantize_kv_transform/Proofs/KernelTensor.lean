import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.KernelWhole
import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.KernelCompiler
import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.KernelAddresses
import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.KernelStores

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.KernelTensor
open VeriTile.Triton FP.Structural KernelInputs
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Caller Tensor identity and its existing storage window. This mixed-dtype
API view covers FP32, INT32 and INT8 without introducing an allocation. -/
structure BoundTensor where
  object : Nat
  storageObject : Nat
  storageBase : Nat
  storageElements : Nat
  storageOffset : Nat
  metadata : VeriTile.Torch.TensorMetadata
  physicalIndex : Nat → Nat
structure TensorLaws (role : Role) (t : BoundTensor) : Prop where
  span : t.storageBase+elementBytes role*t.storageElements≤2^64
  inBounds : ∀ i,i<t.metadata.numel → t.physicalIndex i<t.storageElements
  pointer : 0<t.metadata.numel →
    t.metadata.dataPtr.toNat=t.storageBase+elementBytes role*t.storageOffset
  contiguous : t.metadata.isContiguous=Bool.true → ∀ i,i<t.metadata.numel →
    t.physicalIndex i=t.storageOffset+i

structure Tensors (m : Inputs) where
  tensor : Role → BoundTensor
  metadata : ∀ role,(tensor role).metadata=m role
  laws : ∀ role,TensorLaws role (tensor role)

def logicalAddress (tensors : Tensors m) (role : Role) (i : Nat) : Nat :=
  (tensors.tensor role).storageBase+elementBytes role*(tensors.tensor role).physicalIndex i

theorem address (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (tensors : Tensors m) (role : Role) (i : Nat) (hi : i<(m role).numel) :
    logicalAddress tensors role i=base m role+elementBytes role*i := by
  have ci : (tensors.tensor role).metadata.isContiguous=Bool.true := by
    rw [tensors.metadata]; exact h.contiguous role
  have bound : i<(tensors.tensor role).metadata.numel := by rw [tensors.metadata]; exact hi
  have pointer := (tensors.laws role).pointer (lt_of_le_of_lt (Nat.zero_le i) bound)
  rw [tensors.metadata] at pointer
  simp only [logicalAddress,(tensors.laws role).contiguous ci i bound,Nat.mul_add]
  rw [←Nat.add_assoc,←pointer]
  rfl

def heapWord (heap : Heap) (address : Nat) : Word :=
  decodeWord (fun byte => heap (address+byte.val))
def logicalWord (heap : Heap) (tensors : Tensors m) (role : Role) (i : Nat) : Word :=
  heapWord heap (logicalAddress tensors role i)

/-- Construct initial source cells directly from all four caller data spans.
The original signed metadata words and both caller-owned outputs are read at
their actual pointers. No output value or success equation is an input. -/
def nativeInitial (m : Inputs) (heap : Heap) (u : State Word) : State Word :=
  { u with mem := fun r i =>
    if r="input" then .mk .real (heapWord heap (base m .input+4*i))
    else if r="destination" then .mk .int (loadDest heap (base m .destination+4*i) : Int)
    else if r="output" then .mk .int (BitVec.ofNat 8 (heap (base m .output+i)).val).toInt
    else if r="scale" then .mk .real (heapWord heap (base m .scale+4*i))
    else u.mem r i }

theorem native_reads (m : Inputs) (w : Nat → MetadataWord) (heap : Heap) (u : State Word) :
    NativeReads m w heap (nativeInitial m heap u) := by
  constructor
  · intro token head column ht hh hc
    simp [nativeInitial,heapWord]
  · intro token ht
    simp [nativeInitial]

theorem input_bound (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (token head column : Nat) (ht : token<(request m w).tokens)
    (hh : head<(request m w).heads) (hc : column<(request m w).columns) :
    inputCell (request m w) token head column<(m .input).numel := by
  have size := sizes m w h .input
  have sources := h.shapes.1
  have bound : (token*(request m w).heads+head)*(request m w).columns+column<
      (request m w).sourceRows*(request m w).heads*(request m w).columns := by
    calc
      _ < (token*(request m w).heads+head+1)*(request m w).columns := by
        simp only [Nat.add_mul,Nat.one_mul]
        exact Nat.add_lt_add_left hc _
      _ ≤ ((token+1)*(request m w).heads)*(request m w).columns := by
        apply Nat.mul_le_mul_right
        simp only [Nat.add_mul,Nat.one_mul]
        omega
      _ ≤ _ := Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ (by omega))
  simp only [elements,elementBytes,Nat.mul_assoc] at size
  simp only [inputCell,Nat.add_mul,Nat.mul_assoc] at bound ⊢
  omega

theorem original_input (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (tensors : Tensors m) (heap : Heap) (token head column : Nat)
    (ht : token<(request m w).tokens) (hh : head<(request m w).heads)
    (hc : column<(request m w).columns) :
    KernelWhole.tensorWords m w heap token head column=
      logicalWord heap tensors .input (inputCell (request m w) token head column) := by
  unfold logicalWord
  rw [address m w h tensors .input _ (input_bound m w h token head column ht hh hc)]
  rfl

theorem output_bounds (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (token head column : Nat) (ht : token<(request m w).tokens)
    (hh : head<(request m w).heads) (hc : column<(request m w).columns) :
    KernelResult.codeOffset (parameters m w) ((request m w).destination token) head column<
      (m .output).numel ∧
    KernelResult.scaleOffset (parameters m w) ((request m w).destination token) head<
      (m .scale).numel := by
  have row := RW.linear_bound (request m w).capacity (request m w).heads
    ((request m w).destination token) head
    ((checked_domain m w h).destination_bound token ht) hh
  have flat := RW.linear_bound ((request m w).capacity*(request m w).heads)
    (request m w).columns (((request m w).destination token)*(request m w).heads+head)
    column row hc
  have codeSize := sizes m w h .output
  have scaleSize := sizes m w h .scale
  simp only [elements,elementBytes,Nat.one_mul] at codeSize scaleSize
  constructor
  · rw [←codeSize]
    simpa [KernelResult.codeOffset,parameters,Nat.add_mul,Nat.mul_add,Nat.mul_assoc,
      Nat.mul_comm,Nat.mul_left_comm] using flat
  · change ((request m w).destination token)*(request m w).heads+head<(m .scale).numel
    omega

/-- Each selected store coordinate is a valid element of its caller Tensor,
whose physical address is exactly the checked data pointer plus its offset. -/
theorem output_addresses (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (tensors : Tensors m) (token head column : Nat) (ht : token<(request m w).tokens)
    (hh : head<(request m w).heads) (hc : column<(request m w).columns) :
    logicalAddress tensors .output
      (KernelResult.codeOffset (parameters m w) ((request m w).destination token) head column)=
      base m .output+KernelResult.codeOffset (parameters m w) ((request m w).destination token) head column ∧
    logicalAddress tensors .scale
      (KernelResult.scaleOffset (parameters m w) ((request m w).destination token) head)=
      base m .scale+4*KernelResult.scaleOffset (parameters m w) ((request m w).destination token) head := by
  have bounds := output_bounds m w h token head column ht hh hc
  constructor
  · simpa [elementBytes] using address m w h tensors .output _ bounds.1
  · exact address m w h tensors .scale _ bounds.2

/-- Selected-object native execution with all codes/scales and the complete
typed exterior frame, from constructed reads of the original caller tensors.
The same result exposes the correspondence to original logical FP32 elements. -/
def native (platform : KernelCompiler.Platform) (rest : Algebra Word)
    (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (binding : KernelCompiler.Binding platform (parameters m w))
    (schedule : Schedule (request m w))
    (tensors : Tensors m) (heap : Heap) (storage : TensorBinding m w heap) (u : State Word) :=
  And.intro (fun role i hi => address m w h tensors role i hi)
  (And.intro (fun token head column ht hh hc => original_input m w h tensors heap token head column ht hh hc)
    (And.intro (fun token head column ht hh hc => output_addresses m w h tensors token head column ht hh hc)
      (KernelStores.native (KernelCompiler.arithmetic platform binding.object) rest m w h schedule heap storage
        (nativeInitial m heap u) (native_reads m w heap u))))

/-- Independent Real Tensor values use an arbitrary Real read function. They
do not pass through finite FP32 words or a byte decoder. -/
def realValues (tensors : Tensors m) (w : Nat → MetadataWord) (read : Nat → ℝ)
    (token head column : Nat) : ℝ :=
  read (logicalAddress tensors .input (inputCell (request m w) token head column))
def realInitial (m : Inputs) (w : Nat → MetadataWord) (read : Nat → ℝ) (u : State ℝ) : State ℝ :=
  { u with mem := fun r i =>
    if r="input" then .mk .real (read (base m .input+4*i))
    else if r="destination" then .mk .int (w i).toInt
    else u.mem r i }

theorem real_reads (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (tensors : Tensors m) (read : Nat → ℝ) (u : State ℝ) :
    RealReads m w (realValues tensors w read) (realInitial m w read u) := by
  constructor
  · intro token head column ht hh hc
    unfold realValues
    rw [address m w h tensors .input _ (input_bound m w h token head column ht hh hc)]
    simp [realInitial,elementBytes]
  · intro token ht
    simp [realInitial]

def mathematical (rest : Algebra ℝ) (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (schedule : Schedule (request m w))
    (tensors : Tensors m) (read : Nat → ℝ) (u : State ℝ) :=
  And.intro (fun role i hi => address m w h tensors role i hi)
    (And.intro (fun token head column ht hh hc => output_addresses m w h tensors token head column ht hh hc)
      (KernelWhole.mathematical rest m w h schedule (realValues tensors w read)
        (realInitial m w read u) (real_reads m w h tensors read u)))

#axiomsClean address
#axiomsClean native_reads
#axiomsClean input_bound
#axiomsClean original_input
#axiomsClean output_bounds
#axiomsClean output_addresses
#axiomsClean native
#axiomsClean real_reads
#axiomsClean mathematical
#specNonCircular TensorLaws avoiding [KernelSource.kernel,FP.Structural.exec,KernelGrid.run]
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.KernelTensor
