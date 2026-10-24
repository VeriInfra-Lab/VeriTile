import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Host
import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Admission

namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
open VeriTile.Triton FP.Structural

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

def storageRegion (s : VeriTile.Torch.FP32.Storage) : RegionName :=
  ⟨"torch-storage:" ++ toString s.object⟩

def descriptor (x : Observation) (r : RegionName) (d : Strided.Host.DType) : Strided.Host.Tensor :=
  ⟨x.object,x.metadata.device.index,x.storage.base,x.storageOffset,x.storage.elements,r,d⟩
def matrix (x : Observation) (r : RegionName) : Strided.Host.Matrix :=
  ⟨descriptor x r .fp32,rows x,cols x,rs x,cs x⟩
def vector (x : Observation) (r : RegionName) : Strided.Host.Vector :=
  ⟨descriptor x r .int32,rows x,rs x⟩
def host (c : Call) : Strided.Host.Call :=
  ⟨matrix c.input "input",matrix c.output "output",vector c.start "start",vector c.length "length",c.maximum⟩

/-- Kernel pointer arguments are views of the current physical storage state.
This mathematical interpretation performs no allocation or host/device copy. -/
def view {α : Type} (c : TensorCall) (m : Strided.Grid.Memory α) : Strided.Grid.Memory α := fun r i =>
  if r="input" then m (storageRegion c.input.tensor.storage) (c.input.tensor.storageOffset+i)
  else if r="output" then m (storageRegion c.output.tensor.storage) (c.output.tensor.storageOffset+i)
  else if r="start" then m (storageRegion c.start.storage) (c.start.storageOffset+i)
  else if r="length" then m (storageRegion c.length.storage) (c.length.storageOffset+i)
  else m r i

/-- The physical poststate is the source's writes through its output pointer.
Unaccessed regions and bytes retain the actual caller prestate. -/
def physical {α : Type} [Inhabited α] (c : TensorCall) (before sourceAfter : Strided.Grid.Memory α) : Strided.Grid.Memory α :=
  fun r i => if r=storageRegion c.output.tensor.storage ∧ c.output.tensor.storageOffset  ≤  i then
    sourceAfter "output" (i-c.output.tensor.storageOffset) else before r i

theorem physical_read {α : Type} [Inhabited α] (c : TensorCall) (before after : Strided.Grid.Memory α) (i : Nat) :
    physical c before after (storageRegion c.output.tensor.storage) (c.output.tensor.storageOffset+i)=after "output" i := by
  simp [physical]

theorem physical_frame {α : Type} [Inhabited α] (c : TensorCall) (before after : Strided.Grid.Memory α)
    (frame : Strided.Host.Frame (host c.observation) (view c before) after)
    (r : RegionName) (i : Nat)
    (outside : r≠storageRegion c.output.tensor.storage ∨
      ∀ e∈Strided.Host.outputEvents (host c.observation) (view c before),
      i≠c.output.tensor.storageOffset+Strided.Grid.offset (Strided.Host.params (host c.observation)) (view c before) e.1 e.2) :
    physical c before after r i=before r i := by
  unfold physical
  split
  · rename_i selected
    rcases selected with ⟨rfl,lower⟩
    rw [frame "output" (i-c.output.tensor.storageOffset) (Or.inr (by
      intro e he eq
      rcases outside with bad|off
      · exact bad rfl
      · apply off e he
        omega))]
    simp [view,show c.output.tensor.storageOffset+(i-c.output.tensor.storageOffset)=i by omega]
  · rfl

/-- Literal host source execution paired with the original supplied Tensor. -/
structure TensorResult (α : Type) where
  returned : Option Nat
  output : VeriTile.Torch.FP32.Tensor
  memory : Strided.Grid.Memory α

noncomputable def invoke {α : Type} [Inhabited α] (M : Algebra α) (c : TensorCall)
    (work : List Strided.Grid.PID) (m : Strided.Grid.Memory α) : Option (TensorResult α) :=
  if active c.observation then
    (Strided.Host.invoke M (host c.observation) work (view c m)).map
      (fun result => ⟨result.returned,c.output.tensor,physical c m result.memory⟩)
  else some ⟨none,c.output.tensor,m⟩

theorem checked_layout (c : Call) (h : MetadataContract c) (launched : active c) : Strided.Host.Layout (host c) := by
  rcases h.tensors with ⟨layouts,xd,od,sd,nd,cuda,devices,rank,shape,mrank,mshape,neg⟩
  have guard := h.launch launched
  rcases guard with ⟨batch,heads,width,nonoverlap,rest⟩
  constructor
  · simp [host,matrix,Strided.Host.heads,rows,cols,shape]
  · simp [host,vector,Strided.Host.batches,rows,mshape]
  · intro h hh t ht k hk u hu same
    exact nonoverlap_injective _ _ _ _ h t k u nonoverlap hh ht hk hu same
  · exact ⟨rfl,rfl⟩
  · exact ⟨rfl,rfl⟩
  · have o := devices c.output (by simp [tensors])
    have s := devices c.start (by simp [tensors])
    have n := devices c.length (by simp [tensors])
    simp only [host,matrix,vector,descriptor]
    exact ⟨congrArg VeriTile.Torch.Device.index o.symm,
      congrArg VeriTile.Torch.Device.index (s.trans o.symm),
      congrArg VeriTile.Torch.Device.index (n.trans o.symm)⟩
  · intro r hr
    simp [Strided.Grid.Reads,Strided.Host.params,host,matrix,vector,descriptor,Region.cast] at hr ⊢
    rcases hr with rfl|rfl|rfl  <;> decide

#axiomsClean physical_frame
#axiomsClean checked_layout
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked

namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
open VeriTile.Triton FP.Structural

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

theorem payload_capacity (x : Payload) (rank : x.tensor.metadata.shape.length=2)
    (rh : 0  <  rows x.observation) (ck : 0  <  cols x.observation) :
    Strided.Host.Fits (matrix x.observation "x").toTensor
      (Strided.Host.matrixExtent (rows x.observation) (cols x.observation)
        (rs x.observation) (cs x.observation)) := by
  have numel : x.tensor.metadata.numel=rows x.observation*cols x.observation := by
    unfold VeriTile.Torch.TensorMetadata.numel
    rw [shape_two _ rank]
    simp [rows,cols,Payload.observation]
  have hh : rows x.observation-1  <  rows x.observation := by omega
  have ht : cols x.observation-1  <  cols x.observation := by omega
  have idx := VeriTile.Bench.Repairs.SquareMatrix.flat_lt _ _ _ _ hh ht
  have bounds := x.laws.inBounds _ (numel ▸ idx)
  change x.tensor.physicalIndex ((rows x.observation-1)*(x.tensor.metadata.shape[1]?.getD 0)+(cols x.observation-1))  <  x.tensor.storage.elements at bounds
  rw [x.coordinates _ hh _ ht] at bounds
  simp only [Strided.Host.Fits,Strided.Host.matrixExtent,
    if_neg (show ¬(rows x.observation=0 ∨ cols x.observation=0) by omega)]
  change x.tensor.storageOffset+((rows x.observation-1)*x.rowStride+
    (cols x.observation-1)*x.colStride+1)  ≤  x.tensor.storage.elements
  omega

theorem metadata_capacity (x : MetadataTensor) (rank : x.metadata.shape.length=1)
    (positive : 0  <  rows x.observation) :
    Strided.Host.Fits (vector x.observation "x").toTensor
      (Strided.Host.vectorExtent (rows x.observation) (rs x.observation)) := by
  have numel : x.metadata.numel=rows x.observation := by
    unfold VeriTile.Torch.TensorMetadata.numel
    rw [shape_one _ rank]
    simp [rows,MetadataTensor.observation]
  have bounds := x.bounds (rows x.observation-1) (by omega)
  simp only [Strided.Host.Fits,Strided.Host.vectorExtent,if_neg (Nat.ne_of_gt positive)]
  change x.storageOffset+((rows x.observation-1)*x.stride+1)  ≤  x.storage.elements
  omega

theorem checked_storage (c : TensorCall) (h : MetadataContract c.observation)
    (launched : active c.observation) : Strided.Host.Storage (host c.observation) := by
  rcases h.tensors with ⟨layouts,xd,od,sd,nd,cuda,devices,rank,shape,mrank,mshape,neg⟩
  rcases launched with ⟨batch,heads,tokens,maxlen⟩
  change 0  <  rows c.start.observation at batch
  change 0  <  rows c.input.observation at heads
  change 0  <  cols c.input.observation at tokens
  change 0  <  c.maximum at maxlen
  change c.output.tensor.metadata.shape=c.input.tensor.metadata.shape at shape
  change c.length.metadata.shape=c.start.metadata.shape at mshape
  have outH : rows c.output.observation=rows c.input.observation := congrArg (fun s : List Nat => s.headD 0) shape
  have outK : cols c.output.observation=cols c.input.observation := congrArg (fun s : List Nat => s[1]?.getD 0) shape
  have lenB : rows c.length.observation=rows c.start.observation := congrArg (fun s : List Nat => s.headD 0) mshape
  have input := payload_capacity c.input rank heads tokens
  have output := payload_capacity c.output (by change c.output.tensor.metadata.shape.length=2; rw [shape]; exact rank)
    (by rw [outH]; exact heads) (by rw [outK]; exact tokens)
  have starts := metadata_capacity c.start mrank batch
  have lengths := metadata_capacity c.length (by change c.length.metadata.shape.length=1; rw [mshape]; exact mrank) (by rw [lenB]; exact batch)
  rw [outH,outK] at output
  refine ⟨input,?_,starts,lengths,?_⟩
  · simpa only [host,matrix,Strided.Host.heads,TensorCall.observation,outH,outK] using output
  · intro t ht
    have guard := h.launch ⟨batch,heads,tokens,maxlen⟩
    have sep := guard.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
    have outBound := output
    simp only [Strided.Host.Fits,matrix,descriptor] at outBound
    simp only [host,List.mem_cons,List.not_mem_nil,or_false] at ht
    rcases ht with rfl|rfl|rfl
    · have apart := sep c.observation.input (by simp)
      simp only [Strided.Host.byteAddress,host,matrix,vector,descriptor,Strided.Host.heads,
        TensorCall.observation,Payload.observation,MetadataTensor.observation] at apart outBound ⊢
      rcases apart with l|r
      · exact Or.inl (by omega)
      · exact Or.inr (by omega)
    · have apart := sep c.observation.start (by simp)
      simp only [Strided.Host.byteAddress,host,matrix,vector,descriptor,Strided.Host.heads,
        TensorCall.observation,Payload.observation,MetadataTensor.observation] at apart outBound ⊢
      rcases apart with l|r
      · exact Or.inl (by omega)
      · exact Or.inr (by omega)
    · have apart := sep c.observation.length (by simp)
      simp only [Strided.Host.byteAddress,host,matrix,vector,descriptor,Strided.Host.heads,
        TensorCall.observation,Payload.observation,MetadataTensor.observation] at apart outBound ⊢
      rcases apart with l|r
      · exact Or.inl (by omega)
      · exact Or.inr (by omega)


#axiomsClean payload_capacity
#axiomsClean metadata_capacity
#axiomsClean checked_storage
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
