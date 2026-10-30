import bench.repairs.tritonbench_g.token_softmax_bloom.Contract
import bench.repairs.tritonbench_g.square_matrix.Proofs.Geometry
import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Admission
import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Stores

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelTensor
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
noncomputable section
set_option maxHeartbeats 4000000

structure Payload where
  tensor : VeriTile.Torch.FP32.Tensor
  laws : VeriTile.Torch.FP32.TensorLaws tensor
  (rowStride colStride : Nat)
  coordinates : ∀ h  <  tensor.metadata.shape.headD 0,
    ∀ t  <  tensor.metadata.shape[1]?.getD 0,
    tensor.physicalIndex (h*(tensor.metadata.shape[1]?.getD 0)+t)=
      tensor.storageOffset+h*rowStride+t*colStride

/-- Int32 metadata uses the same shared Tensor metadata and four-byte storage
geometry. Its raw-word reader is the caller's memory, not a cached copy. -/
structure MetadataTensor where
  object : Nat
  metadata : Metadata
  storage : VeriTile.Torch.FP32.Storage
  (storageOffset stride : Nat)
  pointer : 0  <  metadata.numel → metadata.dataPtr=BitVec.ofNat 64 (storage.address storageOffset)
  bounds : ∀ b  <  metadata.numel,storageOffset+b*stride  <  storage.elements

def Payload.observation (x : Payload) : Observation :=
  ⟨x.tensor.metadata,[x.rowStride,x.colStride],x.tensor.storage,x.tensor.storageOffset,x.tensor.object⟩
def MetadataTensor.observation (x : MetadataTensor) : Observation :=
  ⟨x.metadata,[x.stride],x.storage,x.storageOffset,x.object⟩

structure TensorCall where
  (input output : Payload)
  (start length : MetadataTensor)
  maximum : Nat

def TensorCall.observation (c : TensorCall) : Call :=
  ⟨c.input.observation,c.start.observation,c.length.observation,c.output.observation,c.maximum⟩

theorem shape_two (m : Metadata) (rank : m.shape.length=2) :
    m.shape=[m.shape.headD 0,m.shape[1]?.getD 0] := by
  cases hs : m.shape with
  | nil => simp [hs] at rank
  | cons a rest => cases rest with
    | nil => simp [hs] at rank
    | cons b rest => cases rest with
      | nil => simp [hs]
      | cons c rest => simp [hs] at rank

theorem shape_one (m : Metadata) (rank : m.shape.length=1) :
    m.shape=[m.shape.headD 0] := by
  cases hs : m.shape with
  | nil => simp [hs] at rank
  | cons a rest => cases rest with
    | nil => simp [hs]
    | cons b rest => simp [hs] at rank

theorem payload_pointer (x : Payload) (rank : x.tensor.metadata.shape.length=2)
    (rh : 0 < rows x.observation) (ck : 0 < cols x.observation) :
    x.tensor.metadata.dataPtr.toNat=x.tensor.storage.address x.tensor.storageOffset := by
  have numel : 0 < x.tensor.metadata.numel := by
    unfold VeriTile.Torch.TensorMetadata.numel
    rw [shape_two _ rank]
    simpa [rows,cols,Payload.observation] using Nat.mul_pos rh ck
  have first := x.laws.inBounds 0 numel
  have coords := x.coordinates 0 rh 0 ck
  simp only [Nat.zero_mul,Nat.add_zero] at coords
  rw [coords] at first
  have span := x.tensor.storage.spanFits
  have bound : x.tensor.storage.address x.tensor.storageOffset < 2^64 := by
    unfold VeriTile.Torch.FP32.Storage.address
    omega
  have pointer := x.laws.pointer numel
  change x.tensor.metadata.dataPtr=_ at pointer
  rw [pointer]
  exact Nat.mod_eq_of_lt bound

theorem payload_address (x : Payload) (rank : x.tensor.metadata.shape.length=2)
    (h t : Nat) (hh : h < rows x.observation) (ht : t < cols x.observation) :
    x.tensor.metadata.dataPtr.toNat+4*(h*x.rowStride+t*x.colStride)=
      x.tensor.storage.address (x.tensor.physicalIndex (h*cols x.observation+t)) := by
  rw [payload_pointer x rank (by omega) (by omega)]
  have coordinates := x.coordinates h hh t ht
  change x.tensor.physicalIndex (h*cols x.observation+t)=_ at coordinates
  rw [coordinates]
  simp only [VeriTile.Torch.FP32.Storage.address]
  omega

theorem metadata_pointer (x : MetadataTensor) (rank : x.metadata.shape.length=1)
    (positive : 0 < rows x.observation) :
    x.metadata.dataPtr.toNat=x.storage.address x.storageOffset := by
  have numel : 0 < x.metadata.numel := by
    unfold VeriTile.Torch.TensorMetadata.numel
    rw [shape_one _ rank]
    simpa [rows,MetadataTensor.observation] using positive
  have first := x.bounds 0 numel
  have span := x.storage.spanFits
  have bound : x.storage.address x.storageOffset < 2^64 := by
    simp only [Nat.zero_mul,Nat.add_zero] at first
    unfold VeriTile.Torch.FP32.Storage.address
    omega
  rw [x.pointer numel]
  exact Nat.mod_eq_of_lt bound


abbrev NativeCells (α : Type) := Nat → Cell α
abbrev Word := BitVec 32

def params (c : TensorCall) : Parameters where
  input := "input"
  output := "output"
  start := Region.cast ("start" : RegionName)
  length := Region.cast ("length" : RegionName)
  startStride := c.start.stride
  lengthStride := c.length.stride
  inputHeadStride := c.input.rowStride
  inputTokenStride := c.input.colStride
  outputHeadStride := c.output.rowStride
  outputTokenStride := c.output.colStride
  width := block c.maximum

def observedStart (c : TensorCall) (metadataRead : Nat → Word) (b : Nat) : Int :=
  (metadataRead (c.start.metadata.dataPtr.toNat+4*(b*c.start.stride))).toInt

def observedLength (c : TensorCall) (metadataRead : Nat → Word) (b : Nat) : Int :=
  (metadataRead (c.length.metadata.dataPtr.toNat+4*(b*c.length.stride))).toInt

def start (c : TensorCall) (metadataRead : Nat → Word) (b : Nat) : Nat := (observedStart c metadataRead b).toNat

def length (c : TensorCall) (metadataRead : Nat → Word) (b : Nat) : Nat := (observedLength c metadataRead b).toNat

/-- A read-only pointer interpretation of the actual native readers. Aliased
read allocations retain one address; each source load applies its own dtype.
Output cells come from the supplied caller memory with no allocation or copy. -/
def view {α} [Inhabited α] (c : TensorCall) (read : Nat → α) (metadataRead : Nat → Word)
    (before : NativeCells α) : Grid.Memory α := fun r i =>
  if r="input" then .mk .fp32 (read (c.input.tensor.dataPtr.toNat+4*i))
  else if r="start" then .mk .nat ((metadataRead (c.start.metadata.dataPtr.toNat+4*i)).toInt.toNat)
  else if r="length" then .mk .nat ((metadataRead (c.length.metadata.dataPtr.toNat+4*i)).toInt.toNat)
  else if r="output" then before (c.output.tensor.dataPtr.toNat+4*i)
  else .mk .nat 0

@[simp] theorem count_view {α} [Inhabited α] (c : TensorCall) (read : Nat → α)
    (metadataRead : Nat → Word) (before : NativeCells α) (b h : Nat) :
    count (params c) (Grid.state (view c read metadataRead before) (b,h))=length c metadataRead b := by
  simp [count,params,Grid.state,view,length,observedLength]

@[simp] theorem start_view {α} [Inhabited α] (c : TensorCall) (read : Nat → α)
    (metadataRead : Nat → Word) (before : NativeCells α) (b h : Nat) :
    startIndex (params c) (Grid.state (view c read metadataRead before) (b,h))=start c metadataRead b := by
  simp [startIndex,params,Grid.state,view,start,observedStart]

/-- Logical coordinate read from the original Tensor's physical index. -/
def logicalValue {α} (read : Nat → α) (x : Payload) (h t : Nat) : α :=
  read (x.tensor.storage.address (x.tensor.physicalIndex (h*cols x.observation+t)))

/-- The actual caller cell, native pointer reader and logical Tensor value are
connected independently of source execution or any softmax formula. -/
def InputBinding {α} [Inhabited α] (c : TensorCall) (read : Nat → α)
    (before : NativeCells α) : Prop :=
  ∀ h < rows c.input.observation, ∀ t < cols c.input.observation,
    (before (c.input.tensor.storage.address
      (c.input.tensor.physicalIndex (h*cols c.input.observation+t)))).read .fp32 =
      logicalValue read c.input h t

theorem source_load {α} [Inhabited α] (c : TensorCall) (read : Nat → α)
    (metadataRead : Nat → Word) (before : NativeCells α)
    (binding : InputBinding c read before)
    (rank : c.input.tensor.metadata.shape.length=2)
    (h t : Nat) (hh : h < rows c.input.observation) (ht : t < cols c.input.observation) :
    (view c read metadataRead before "input" (h*c.input.rowStride+t*c.input.colStride)).read .fp32=
      read (c.input.tensor.dataPtr.toNat+4*(h*c.input.rowStride+t*c.input.colStride)) ∧
    read (c.input.tensor.dataPtr.toNat+4*(h*c.input.rowStride+t*c.input.colStride))=
      logicalValue read c.input h t ∧
    logicalValue read c.input h t=
      (before (c.input.tensor.storage.address
        (c.input.tensor.physicalIndex (h*cols c.input.observation+t)))).read .fp32 := by
  refine ⟨by simp [view],?_,(binding h hh t ht).symm⟩
  exact congrArg read (payload_address c.input rank h t hh ht)

/-- Materialize only the source's output-pointer writes into the original native
cell map. The raw source frame proves every other caller cell is unchanged. -/
def project {α} [Inhabited α] (c : TensorCall) (before : NativeCells α)
    (after : Grid.Memory α) : NativeCells α := fun address =>
  if c.output.tensor.dataPtr.toNat ≤ address ∧ (address-c.output.tensor.dataPtr.toNat)%4=0
  then after "output" ((address-c.output.tensor.dataPtr.toNat)/4) else before address

@[simp] theorem project_read {α} [Inhabited α] (c : TensorCall) (before : NativeCells α)
    (after : Grid.Memory α) (i : Nat) :
    project c before after (c.output.tensor.dataPtr.toNat+4*i)=after "output" i := by
  simp [project,Nat.add_sub_cancel_left,Nat.mul_div_right]

#axiomsClean source_load
#axiomsClean project_read
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelTensor
