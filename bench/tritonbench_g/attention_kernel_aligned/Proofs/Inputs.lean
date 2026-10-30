import bench.tritonbench_g.attention_kernel_aligned.Proofs.Contract
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.KernelOnly
open VeriTile Triton FP.Structural Full Whole
noncomputable section
set_option maxHeartbeats 4000000
abbrev Role := Buffer
def roles : List Role := [.query,.key,.value,.bias,.output]
def reads : List Role := [.query,.key,.value,.bias]
def tag : Role → RegionName
  | .query=>"Q" | .key=>"K" | .value=>"V" | .bias=>"Bias" | .output=>"Out"

/-- Original two-byte Tensor metadata and its ordinary coordinate operation. -/
structure Tensor where
  object : Nat
  metadata : Torch.TensorMetadata
  strides : Strides
  (isConj ordinary : Bool)
  (storageObject storageBase storageElements storageOffset : Nat)
  physicalIndex : Nat → Nat → Nat → Nat → Nat

structure Call where
  tensors : Role → Tensor
  (bm bn numWarps numStages nativeColumns : Nat)
  nativeSerial : Bool
  scale : ℝ
  scaleWord : BitVec 32
  (scalarIsFiniteIntOrFloat launchArgumentsAreInts : Bool)
instance : CoeFun Call (fun _=>Role → Tensor) := ⟨Call.tensors⟩
def extent (c : Call) (i : Nat) := (c .query).metadata.shape[i]?.getD 0
def base (c : Call) (r : Role) := (c r).metadata.dataPtr.toNat
def parameters (c : Call) : Parameters where
  q:=tag .query
  k:=tag .key
  v:=tag .value
  bias:=tag .bias
  output:=tag .output
  scale:=c.scale
  scaleWord:=c.scaleWord
  qBatchStride:=(c .query).strides.batch
  qHeadStride:=(c .query).strides.head
  qRowStride:=(c .query).strides.row
  qColStride:=(c .query).strides.dim
  kBatchStride:=(c .key).strides.batch
  kHeadStride:=(c .key).strides.head
  kSeqStride:=(c .key).strides.row
  kDimStride:=(c .key).strides.dim
  vBatchStride:=(c .value).strides.batch
  vHeadStride:=(c .value).strides.head
  vSeqStride:=(c .value).strides.row
  vDimStride:=(c .value).strides.dim
  outBatchStride:=(c .output).strides.batch
  outHeadStride:=(c .output).strides.head
  outRowStride:=(c .output).strides.row
  outColStride:=(c .output).strides.dim
  biasBatchStride:=(c .bias).strides.batch
  biasHeadStride:=(c .bias).strides.head
  biasRowStride:=(c .bias).strides.row
  batches:=extent c 0
  heads:=extent c 1
  length:=extent c 2
  head:=extent c 3
  blockM:=c.bm
  blockN:=c.bn
  biasLast:=64
  biasNumel:=128
  numWarps:=c.numWarps
  numStages:=c.numStages
  nativeColumns:=c.nativeColumns
  nativeSerial:=c.nativeSerial
def shape (p : Parameters) (r : Role) :=
  [p.batches,p.heads,p.length,if r=.bias then 128 else p.head]
def width (p : Parameters) (r : Role) := if r=.bias then 128 else p.head
def Valid (p : Parameters) (r : Role) (b h n d : Nat) :=
  b<p.batches ∧ h<p.heads ∧ n<p.length ∧ d<width p r
def axisOrder (c : Call) : List (Fin 4) :=
  ((List.finRange 4).mergeSort (fun i j=>decide
    (strideAt (c .output).strides i<strideAt (c .output).strides j ∨
      (strideAt (c .output).strides i=strideAt (c .output).strides j ∧
        extentAt (parameters c) i≤extentAt (parameters c) j)))).reverse
def observations (c : Call) : Contract.Observations where
  order:=axisOrder c
  pointer:=base c
  storageBase:=fun r=>(c r).storageBase
  storageBytes:=fun r=>2*(c r).storageElements
  ordinaryCUDA:=Bool.true

/-- The checked metadata, original allocation and launch domain. Scalar/type
classification is an observation boundary; no result or run predicate occurs. -/
structure InputContract (c : Call) : Prop where
  tensors : ∀ r,(c r).ordinary=Bool.true ∧ (c r).metadata.layout=.strided ∧
    (c r).metadata.isCuda=Bool.true ∧ (c r).metadata.device=(c .query).metadata.device ∧
    (c r).metadata.isNeg=Bool.false ∧ (c r).isConj=Bool.false
  shapes : ∀ r,(c r).metadata.shape=shape (parameters c) r
  dtype : ((c .query).metadata.dtype=.other "float16" ∨ (c .query).metadata.dtype=.floating .bf16) ∧
    ∀ r,(c r).metadata.dtype=(c .query).metadata.dtype
  contiguous : (c .bias).metadata.isContiguous=Bool.true
  classified : c.scalarIsFiniteIntOrFloat=Bool.true ∧ c.launchArgumentsAreInts=Bool.true
  scaleDecoded : (Float32Bits.mk c.scaleWord).decodeReal=some c.scale
  scaleRange : |c.scale|≤340282346638528860000000000000000000000
  launch : c.numWarps∈[4,8] ∧ 1≤c.numStages ∧ c.numStages≤4
  resourceGuard : ¬(c.bn=128 ∧ (parameters c).head=128 ∧
    (c.numStages=4 ∨ (c.bm=128 ∧ c.numStages=3)))
  checked : Contract.InputContract (parameters c) (observations c)

def check (c : Call) : Bool := by
  classical
  exact decide (InputContract c)
theorem check_iff (c : Call) : check c=Bool.true ↔ InputContract c := by
  classical
  simp [check]
def guard (c : Call) (hc : InputContract c) : Guard (parameters c) :=
  Contract.guard _ _ hc.checked

/-- Ordinary Tensor API laws, retaining arbitrary singleton strides. -/
structure TensorLaws (c : Call) : Prop where
  capacity : ∀ r,(c r).storageBase+2*(c r).storageElements≤2^64
  indices : ∀ r b h n d,Valid (parameters c) r b h n d →
    (c r).physicalIndex b h n d=(c r).storageOffset+offset (c r).strides b h n d
  inBounds : ∀ r b h n d,Valid (parameters c) r b h n d →
    (c r).physicalIndex b h n d<(c r).storageElements
  pointer : ∀ r,0<(c r).metadata.numel →
    (c r).metadata.dataPtr=BitVec.ofNat 64 ((c r).storageBase+2*(c r).storageOffset)
  biasDim : (c .bias).metadata.isContiguous=Bool.true → Nonempty (parameters c) → (c .bias).strides.dim=1

def logicalAddress (c : Call) (r : Role) (b h n d : Nat) :=
  (c r).storageBase+2*(c r).physicalIndex b h n d
def logical {α : Type} (read : Nat → α) (c : Call) (r : Role) (b h n d : Nat) :=
  read (logicalAddress c r b h n d)
def sourceAddress (c : Call) (r : Role) (b h n d : Nat) :=
  base c r+2*offset (c r).strides b h n d
theorem actual_address (c : Call) (hc : InputContract c) (laws : TensorLaws c)
    (r : Role) (b h n d : Nat) (valid : Valid (parameters c) r b h n d) :
    logicalAddress c r b h n d=sourceAddress c r b h n d := by
  have positive : 0<(c r).metadata.numel := by
    rw [Torch.TensorMetadata.numel,hc.shapes r]
    rcases valid with ⟨hb,hh,hn,hd⟩
    change 0<(parameters c).batches*((parameters c).heads*((parameters c).length*(width (parameters c) r*1)))
    exact Nat.mul_pos (by omega) (Nat.mul_pos (by omega) (Nat.mul_pos (by omega) (by omega)))
  have cap:=laws.capacity r
  have bound:=laws.inBounds r b h n d valid
  rw [laws.indices r b h n d valid] at bound
  have ptr:=laws.pointer r positive
  have small : (c r).storageBase+2*(c r).storageOffset<2^64 := by omega
  simp only [logicalAddress,sourceAddress,base,ptr,BitVec.toNat_ofNat,Nat.mod_eq_of_lt small,
    laws.indices r b h n d valid]
  omega
theorem actual_read {α : Type} (read : Nat → α) (c : Call) (hc : InputContract c)
    (laws : TensorLaws c) (r : Role) (b h n d : Nat) (valid : Valid (parameters c) r b h n d) :
    logical read c r b h n d=read (sourceAddress c r b h n d) :=
  congrArg read (actual_address c hc laws r b h n d valid)
theorem bias_read {α : Type} (read : Nat → α) (c : Call) (hc : InputContract c)
    (laws : TensorLaws c) (b h n d : Nat) (valid : Valid (parameters c) .bias b h n d) :
    read (base c .bias+2*(offset (c .bias).strides b h n 0+d))=logical read c .bias b h n d := by
  have live : Nonempty (parameters c) := ⟨by have:=valid.1;omega,
    by have:=valid.2.1;omega,by have:=valid.2.2.1;omega⟩
  rw [actual_read read c hc laws .bias b h n d valid]
  simp [sourceAddress,offset,laws.biasDim hc.contiguous live]

theorem offset_bound (c : Call) (laws : TensorLaws c) (r : Role) (b h n d : Nat)
    (valid : Valid (parameters c) r b h n d) :
    offset (c r).strides b h n d<(c r).storageElements-(c r).storageOffset := by
  have bound:=laws.inBounds r b h n d valid
  rw [laws.indices r b h n d valid] at bound
  omega

/-- Finite storage geometry is derived from actual Tensor coordinates. -/
def tensorStorage (c : Call) (hc : InputContract c) (laws : TensorLaws c) : TensorStorage (parameters c) where
  sizeQ:=(c .query).storageElements-(c .query).storageOffset
  sizeK:=(c .key).storageElements-(c .key).storageOffset
  sizeV:=(c .value).storageElements-(c .value).storageOffset
  sizeBias:=(c .bias).storageElements-(c .bias).storageOffset
  sizeOut:=(c .output).storageElements-(c .output).storageOffset
  query:=fun b hb h hh n hn d hd=>offset_bound c laws .query b h n d ⟨hb,hh,hn,hd⟩
  key:=fun b hb h hh n hn d hd=>offset_bound c laws .key b h n d ⟨hb,hh,hn,hd⟩
  value:=fun b hb h hh n hn d hd=>offset_bound c laws .value b h n d ⟨hb,hh,hn,hd⟩
  output:=fun b hb h hh n hn d hd=>offset_bound c laws .output b h n d ⟨hb,hh,hn,hd⟩
  bias:=by
    intro b hb h hh n hn d hd
    have live : Nonempty (parameters c) := ⟨by omega,by omega,by omega⟩
    have dim:=laws.biasDim hc.contiguous live
    simpa [offset,bStrides,parameters,dim] using
      (offset_bound c laws .bias b h n d ⟨hb,hh,hn,hd⟩)
  separated:=by intro live;simp [parameters,tag]

def kernel_access (c : Call) (hc : InputContract c) (laws : TensorLaws c)
    (x y block row dim key : Nat) (hy : y<(parameters c).batches*(parameters c).heads)
    (hblock : block<(parameters c).blocks) :=
  Whole.access_bounds (parameters c) (guard c hc) (tensorStorage c hc laws)
    x y block row dim key hy hblock

#axiomsClean check_iff
#axiomsClean actual_address
#axiomsClean actual_read
#axiomsClean bias_read
#axiomsClean kernel_access
#specNonCircular InputContract avoiding [Whole.runPrograms,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.KernelOnly
