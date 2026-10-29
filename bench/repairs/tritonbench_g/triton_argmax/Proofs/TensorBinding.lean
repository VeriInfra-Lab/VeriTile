import bench.repairs.tritonbench_g.triton_argmax.Proofs.NativeGraph

namespace VeriTile.Bench.Repairs.TritonArgmax.TensorBinding
open Source Memory HostExecution
noncomputable section

/-- Actual shared FP32 Tensor and ordinary strided coordinate observation. -/
structure Input where
  tensor : VeriTile.Torch.FP32.Tensor
  laws : VeriTile.Torch.FP32.TensorLaws tensor
  strides : List Nat
  coordinates : ∀ i, i < tensor.metadata.numel →
    tensor.physicalIndex i = tensor.storageOffset + Host.viewOffset tensor.metadata.shape strides i

def Input.storage (x : Input) : Allocation :=
  ⟨x.tensor.metadata.device.index,x.tensor.storage.base,4*x.tensor.storage.elements⟩

def Input.host (x : Input) : Host.Tensor :=
  ⟨x.tensor.object,"torch.float32",x.storage,x.tensor.metadata.shape,x.strides,
    x.tensor.storageOffset,x.tensor.metadata.isNeg,fun w => w ^^^ 0x80000000⟩

theorem actual_input (x : Input) (i : Nat) (hi : i < x.tensor.metadata.numel) :
    x.host.offset i = x.tensor.physicalIndex i := (x.coordinates i hi).symm

theorem input_capacity (x : Input) (contiguous : x.tensor.metadata.isContiguous = Bool.true) :
    4*x.tensor.metadata.numel ≤ x.host.dataWindow.bytes := by
  by_cases empty : x.tensor.metadata.numel = 0
  · simp [empty]
  have bound := VeriTile.Torch.FP32.contiguous_capacity x.tensor x.laws contiguous
    (Nat.pos_of_ne_zero empty)
  change 4*x.tensor.metadata.numel ≤ 4*x.tensor.storage.elements-4*x.tensor.storageOffset
  omega

def direct (mem : Primitives M) (before : M) (x : Input)
    (contiguous : x.tensor.metadata.isContiguous = Bool.true)
    (noNeg : x.tensor.metadata.isNeg = Bool.false) : Host.Packed mem before x.host where
  tensor := x.host
  state := before
  shape := rfl
  device := rfl
  capacity := input_capacity x contiguous
  coordinates := by
    intro i hi
    have physical := actual_input x i hi
    have linear := x.laws.contiguous contiguous i hi
    change mem.read 4 before (address x.host.dataWindow 4 i) =
      Host.logicalWord mem before x.host i
    simp only [Host.logicalWord,Input.host,noNeg,Bool.false_eq_true,ite_false]
    have offset : x.tensor.storageOffset + Host.viewOffset x.tensor.metadata.shape x.strides i =
        x.tensor.storageOffset+i := by
      exact physical.trans linear
    simp only [Host.Tensor.dataWindow,Host.Tensor.offset,address,Input.host]
    rw [offset]
    congr 2
    omega
  callerFrame := fun _ _ => rfl
  libraryWrites := []
  libraryFrame := fun _ _ => rfl

/-- Empty/zeros allocations and squeeze views expose their actual Tensor
objects, queried metadata, storage owner, and data_ptr. Fresh storage starts at
offset zero; scalar memset is executed separately by initializeIndices. -/
structure AllocatedTensor where
  object : Nat
  metadata : VeriTile.Torch.TensorMetadata
  storage : Allocation
  dtypeName : String
  width : Nat
  strides : List Nat
  dtype : metadata.dtype = if dtypeName = "torch.float32" then .floating .fp32 else .other dtypeName
  device : metadata.device.kind = "cuda" ∧ metadata.device.index = storage.owner
  pointer : 0 < metadata.numel → metadata.dataPtr = BitVec.ofNat 64 storage.base
  checkedAlignment : 0 < metadata.numel → metadata.dataPtr.toNat % min width 8 = 0
  capacity : width*metadata.numel ≤ storage.bytes
  span : storage.base+storage.bytes ≤ 2^64
  linear : ∀ i, i < metadata.numel → Host.viewOffset metadata.shape strides i = i

def AllocatedTensor.host (x : AllocatedTensor) : Host.Tensor :=
  ⟨x.object,x.dtypeName,x.storage,x.metadata.shape,x.strides,0,Bool.false,id⟩

structure Output where
  allocated : AllocatedTensor
  returned : AllocatedTensor
  sameStorage : returned.storage = allocated.storage
  sameDtype : returned.dtypeName = allocated.dtypeName
  sameWidth : returned.width = allocated.width
  int64 : returned.dtypeName = "torch.int64"
  width : returned.width = 8

def Output.host (o : Output) : Host.Output where
  allocated := o.allocated.host
  returned := o.returned.host
  sameStorage := o.sameStorage
  sameOffset := rfl
  zeroOffset := rfl
  int64 := o.int64
  linearCoordinates := by
    intro i hi
    simpa [AllocatedTensor.host,Host.Tensor.offset] using o.returned.linear i hi

structure GlobalAllocation (x : Input) (o : Checked.Observation) (d : MidFormat)
    (callers : List Allocation) where
  midValue : AllocatedTensor
  midIndex : AllocatedTensor
  output : Output
  directOutput : output.returned = output.allocated
  midShape : midValue.metadata.shape = [Checked.blocks o]
  midDtype : midValue.dtypeName = d.name
  midWidth : midValue.width = d.bytes
  indexShape : midIndex.metadata.shape = [Checked.midBlock o]
  indexDtype : midIndex.dtypeName = "torch.int64"
  indexWidth : midIndex.width = 8
  outputShape : output.returned.metadata.shape = Host.outputShape x.tensor.metadata.shape none o.keepdim
  ownerMid : midValue.storage.owner = x.storage.owner
  ownerIndex : midIndex.storage.owner = x.storage.owner
  ownerOutput : output.returned.storage.owner = x.storage.owner
  midInput : midValue.storage.Disjoint x.host.dataWindow
  indexInput : midIndex.storage.Disjoint x.host.dataWindow
  outputInput : output.returned.storage.Disjoint x.host.dataWindow
  indexMid : midIndex.storage.Disjoint midValue.storage
  outputMid : output.returned.storage.Disjoint midValue.storage
  outputIndex : output.returned.storage.Disjoint midIndex.storage
  callerInput : x.storage ∈ callers
  freshMid : HostExecution.FreshFor midValue.storage callers
  freshIndex : HostExecution.FreshFor midIndex.storage callers
  freshOutput : HostExecution.FreshFor output.returned.storage callers

def GlobalAllocation.buffers (a : GlobalAllocation x o d callers) : GlobalMemory.Buffers :=
  ⟨x.host.dataWindow,a.midValue.storage,a.midIndex.storage,a.output.returned.storage⟩

def GlobalAllocation.setup (mem : Primitives M) (before : M)
    (x : Input) (o : Checked.Observation) (observed : o.input = x.tensor.metadata)
    (h : Checked.GlobalContract o) (g : NativeGraph.GlobalPrograms d o)
    (a : GlobalAllocation x o d callers) : HostExecution.GlobalSetup mem before x.host d where
  packed := direct mem before x (observed ▸ h.1) (observed ▸ h.2.1)
  c := g.config
  p := a.buffers
  output := a.output.host
  inputBuffer := rfl
  outputBuffer := rfl
  size := congrArg VeriTile.Torch.TensorMetadata.numel observed
  layout := by
    have mid := a.midValue.capacity
    have index := a.midIndex.capacity
    have output := a.output.returned.capacity
    change a.midValue.width*a.midValue.metadata.shape.prod ≤ _ at mid
    change a.midIndex.width*a.midIndex.metadata.shape.prod ≤ _ at index
    change a.output.returned.width*a.output.returned.metadata.shape.prod ≤ _ at output
    rw [a.midWidth,a.midShape] at mid
    rw [a.indexWidth,a.indexShape] at index
    rw [a.output.width,a.outputShape,RequestBinding.global_shape_product] at output
    refine ⟨?_,by simpa [VeriTile.Torch.TensorMetadata.numel] using mid,
      by simpa [VeriTile.Torch.TensorMetadata.numel] using index,?_,
      h.2.2.2.2.2.2.2.1,a.midInput,a.indexInput,a.outputInput,a.indexMid,a.outputMid,a.outputIndex⟩
    · change 4*o.input.numel ≤ x.host.dataWindow.bytes
      rw [observed]
      exact input_capacity x (observed ▸ h.1)
    · simpa using output
  freshMid := a.freshMid x.storage a.callerInput
  freshIndex := a.freshIndex x.storage a.callerInput
  freshOutput := a.freshOutput x.storage a.callerInput

/-- Exactly the original dimension branch's ordinary contiguous() call.
It may return its input or allocate. The scalar-coordinate and fresh-region
frame laws are library semantics, independent of the reduction or its outputs. -/
structure ContiguousCall (mem : Primitives M) (before : M) (x : Input)
    (callers : List Allocation) where
  result : Input
  state : M
  sameShape : result.tensor.metadata.shape = x.tensor.metadata.shape
  sameDevice : result.tensor.metadata.device = x.tensor.metadata.device
  contiguous : result.tensor.metadata.isContiguous = Bool.true
  values : ∀ i, i < x.tensor.metadata.numel →
    mem.read 4 state (address result.host.dataWindow 4 i) = Host.logicalWord mem before x.host i
  writes : List Allocation
  fresh : ∀ a, a ∈ writes → HostExecution.FreshFor a callers
  frame : ∀ point, (∀ a, a ∈ writes → ¬a.contains point) → mem.byte state point = mem.byte before point
  callerInput : x.storage ∈ callers

def ContiguousCall.packed (call : ContiguousCall mem before x callers) : Host.Packed mem before x.host where
  tensor := call.result.host
  state := call.state
  shape := call.sameShape
  device := congrArg VeriTile.Torch.Device.index call.sameDevice
  capacity := by
    have h := input_capacity call.result call.contiguous
    simpa only [VeriTile.Torch.TensorMetadata.numel,call.sameShape] using h
  coordinates := call.values
  callerFrame := by
    intro point inside
    apply call.frame
    intro a ha
    exact HostExecution.disjoint_not_contains a x.storage
      (call.fresh a ha x.storage call.callerInput) point inside
  libraryWrites := call.writes
  libraryFrame := call.frame

structure DimensionAllocation (x : Input) (o : Checked.Observation) (ax : Nat)
    (prepared : Input) (callers : List Allocation) where
  output : Output
  keptOutput : o.keepdim = Bool.true → output.returned = output.allocated
  allocatedShape : output.allocated.metadata.shape = x.tensor.metadata.shape.set ax 1
  shape : output.returned.metadata.shape =
    Host.outputShape x.tensor.metadata.shape (some ax) o.keepdim
  owner : output.returned.storage.owner = x.storage.owner
  apart : output.returned.storage.Disjoint prepared.host.dataWindow
  fresh : HostExecution.FreshFor output.returned.storage callers

def DimensionAllocation.buffers
    (a : DimensionAllocation x o ax prepared callers) : DimMemory.Buffers :=
  ⟨prepared.host.dataWindow,a.output.returned.storage⟩

def DimensionAllocation.setup (mem : Primitives M) (before : M)
    (x : Input) (o : Checked.Observation) (observed : o.input = x.tensor.metadata)
    (ax : Nat) (valid : ax < x.tensor.metadata.shape.length)
    (g : NativeGraph.DimensionPrograms) (call : ContiguousCall mem before x callers)
    (a : DimensionAllocation x o ax call.result callers) :
    HostExecution.DimSetup mem before x.host where
  packed := call.packed
  c := g.config o ax
  p := a.buffers
  output := a.output.host
  inputBuffer := rfl
  outputBuffer := rfl
  size := by
    change Checked.outer o ax*Checked.extent o ax*Checked.inner o ax = x.tensor.metadata.shape.prod
    simp only [Checked.outer,Checked.extent,Checked.inner,observed]
    exact RequestBinding.shape_factorization _ ax valid
  layout := by
    have cap := call.packed.capacity
    have inputSize := RequestBinding.shape_factorization x.tensor.metadata.shape ax valid
    have output := a.output.returned.capacity
    change a.output.returned.width*a.output.returned.metadata.shape.prod ≤ _ at output
    rw [a.output.width,a.shape,RequestBinding.dimension_shape_product _ ax valid] at output
    refine ⟨?_,?_,a.apart⟩
    · change 4*(Checked.outer o ax*Checked.extent o ax*Checked.inner o ax) ≤ _
      simpa only [Checked.outer,Checked.extent,Checked.inner,observed,inputSize] using cap
    · simpa only [NativeGraph.DimensionPrograms.config,Checked.outer,Checked.inner,observed]
        using output
  freshOutput := a.fresh x.storage call.callerInput

structure Result (M : Type) where
  returned : AllocatedTensor
  state : M

def globalInvoke (mem : Primitives M) (before : M) (p : NativeGraph.Primitives)
    (adapter : RealScratch.Adapter) (o : Checked.Observation)
    (g : NativeGraph.GlobalPrograms adapter.format o)
    (a : GlobalAllocation x o adapter.format callers) : Except String (Result M) :=
  if Checked.check o then .ok ⟨a.output.returned,
    NativeGraph.global p adapter g mem a.buffers before⟩ else .error "input contract rejected"

def dimensionInvoke (mem : Primitives M) (before : M)
    (o : Checked.Observation) (ax : Nat) (g : NativeGraph.DimensionPrograms)
    (call : ContiguousCall mem before x callers)
    (a : DimensionAllocation x o ax call.result callers) : Except String (Result M) :=
  if Checked.check o then .ok ⟨a.output.returned,
    NativeGraph.dimension g o ax mem a.buffers call.state⟩ else .error "input contract rejected"

#axiomsClean actual_input
#axiomsClean input_capacity
end
end VeriTile.Bench.Repairs.TritonArgmax.TensorBinding
