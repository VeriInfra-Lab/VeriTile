import bench.tritonbench_g.attention_kernel.Proofs.Contract
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.AttentionKernel.KernelOnly
open VeriTile Triton FP.Structural Whole001 Source Whole Machine
noncomputable section
set_option maxHeartbeats 4000000

inductive Role where | query | key | value | bias | output
  deriving DecidableEq, Repr
def roles : List Role := [.query,.key,.value,.bias,.output]
def reads : List Role := [.query,.key,.value,.bias]
def tag : Role → RegionName
  | .query => "Q" | .key => "K" | .value => "V" | .bias => "Bias" | .output => "Out"

/-- The original two-byte Tensor object and its backing storage observations.
`physicalIndex` is the ordinary four-dimensional Tensor indexing operation. -/
structure Tensor where
  object : Nat
  metadata : VeriTile.Torch.TensorMetadata
  strides : Strides
  (isConj ordinary : Bool)
  (storageObject storageBase storageElements storageOffset : Nat)
  physicalIndex : Nat → Nat → Nat → Nat → Nat

structure Call where
  tensors : Role → Tensor
  (bm bn numWarps numStages : Nat)
  scale : ℝ
  scaleWord : BitVec 32
  (scalarIsFiniteIntOrFloat launchArgumentsAreInts : Bool)
instance : CoeFun Call (fun _ => Role → Tensor) := ⟨Call.tensors⟩

def storage (c : Call) : TypedSource.Storage :=
  if (c .query).metadata.dtype = .floating .bf16 then .bf16 else .fp16
def dtype (c : Call) : VeriTile.Torch.DType :=
  match storage c with | .fp16 => .other "float16" | .bf16 => .floating .bf16
def extent (c : Call) (i : Nat) := (c .query).metadata.shape[i]?.getD 0
def base (c : Call) (r : Role) := (c r).metadata.dataPtr.toNat
def span (c : Call) (r : Role) := ((c r).storageBase,2*(c r).storageElements)
def parameters (c : Call) : Parameters where
  storage := storage c
  Q := tag .query
  K := tag .key
  V := tag .value
  Bias := tag .bias
  Out := tag .output
  q := (c .query).strides
  k := (c .key).strides
  v := (c .value).strides
  bias := (c .bias).strides
  o := (c .output).strides
  batches := extent c 0
  heads := extent c 1
  length := extent c 2
  dim := extent c 3
  bm := c.bm
  bn := c.bn
  scale := c.scale

def descriptor (c : Call) (r : Role) : Descriptor where
  object := (c r).object
  device := (c r).metadata.device.index
  region := tag r
  dtype := storage c
  shape := (c r).metadata.shape
  strides := [(c r).strides.batch,(c r).strides.head,(c r).strides.row,(c r).strides.dim]
  negative := (c r).metadata.isNeg
  conjugate := (c r).isConj
  contiguous := (c r).metadata.isContiguous
  cuda := (c r).metadata.isCuda

open Classical in
def arguments (c : Call) : Contract.Arguments where
  query := descriptor c .query
  key := descriptor c .key
  value := descriptor c .value
  bias := descriptor c .bias
  output := descriptor c .output
  scaleWord := c.scaleWord
  numWarps := c.numWarps
  numStages := c.numStages
  tensors := decide (∀ r,(c r).ordinary=Bool.true)
  scalar := c.scalarIsFiniteIntOrFloat
  integers := c.launchArgumentsAreInts
  strided := decide (∀ r,(c r).metadata.layout=.strided)
  supportedDTypes := decide (∀ r,(c r).metadata.dtype=dtype c)
  readSpans := reads.map (span c)
  outputSpan := span c .output

def pointers (c : Call) : Bases :=
  ⟨base c .query,base c .key,base c .value,base c .bias,base c .output⟩

/-- Same semantic guard on actual observations. There is no launch-success,
result, floating-payload or resource-availability predicate in admission. -/
def InputContract (c : Call) : Prop :=
  Contract.InputContract (parameters c) (pointers c) (arguments c)
def check (c : Call) := Contract.check (parameters c) (pointers c) (arguments c)
theorem check_iff (c : Call) : check c=.ok () ↔ InputContract c :=
  Contract.check_iff _ _ _

def shape (p : Parameters) (r : Role) : List Nat :=
  [p.batches,p.heads,p.length,if r=.bias then 128 else p.dim]
def width (p : Parameters) (r : Role) := if r=.bias then 128 else p.dim
def Valid (p : Parameters) (r : Role) (b h n d : Nat) :=
  b<p.batches ∧ h<p.heads ∧ n<p.length ∧ d<width p r

theorem admitted_shapes (c : Call) (hc : InputContract c) (r : Role) :
    (c r).metadata.shape=shape (parameters c) r := by
  rcases hc.1 with ⟨_,_,_,_,_,_,_,_,_,hs,_⟩
  simp only [arguments,descriptor] at hs
  rcases hs with ⟨hq,hk,hv,ho,hb⟩
  cases r <;> simp_all [shape]

/-- Ordinary storage/stride API laws. These mention no arithmetic result or
execution and impose no input/input separation. -/
structure TensorLaws (c : Call) : Prop where
  capacity : ∀ r,(c r).storageBase+2*(c r).storageElements ≤ 2^64
  indices : ∀ r b h n d,Valid (parameters c) r b h n d →
    (c r).physicalIndex b h n d=(c r).storageOffset+offset (c r).strides b h n d
  inBounds : ∀ r b h n d,Valid (parameters c) r b h n d →
    (c r).physicalIndex b h n d < (c r).storageElements
  pointer : ∀ r,0<(c r).metadata.numel →
    (c r).metadata.dataPtr=BitVec.ofNat 64 ((c r).storageBase+2*(c r).storageOffset)
  biasDim : (c .bias).metadata.isContiguous=Bool.true → Live (parameters c) →
    (c .bias).strides.dim=1

def logicalAddress (c : Call) (r : Role) (b h n d : Nat) :=
  (c r).storageBase+2*(c r).physicalIndex b h n d
def logical {α : Type} (read : Nat → α) (c : Call) (r : Role) (b h n d : Nat) :=
  read (logicalAddress c r b h n d)
def sourceAddress (c : Call) (r : Role) (b h n d : Nat) :=
  base c r+2*offset (c r).strides b h n d

theorem numel_positive (c : Call) (hc : InputContract c) (r : Role) (b h n d : Nat)
    (valid : Valid (parameters c) r b h n d) : 0<(c r).metadata.numel := by
  rw [VeriTile.Torch.TensorMetadata.numel,admitted_shapes c hc r]
  rcases valid with ⟨hb,hh,hn,hd⟩
  change 0<(parameters c).batches*((parameters c).heads*((parameters c).length*(width (parameters c) r*1)))
  exact Nat.mul_pos (by omega) (Nat.mul_pos (by omega) (Nat.mul_pos (by omega) (by omega)))

theorem actual_address (c : Call) (hc : InputContract c) (laws : TensorLaws c)
    (r : Role) (b h n d : Nat) (valid : Valid (parameters c) r b h n d) :
    logicalAddress c r b h n d=sourceAddress c r b h n d := by
  have cap := laws.capacity r
  have bound := laws.inBounds r b h n d valid
  rw [laws.indices r b h n d valid] at bound
  have ptr := laws.pointer r (numel_positive c hc r b h n d valid)
  have small : (c r).storageBase+2*(c r).storageOffset < 2^64 := by omega
  simp only [logicalAddress,sourceAddress,base,ptr,BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt small,laws.indices r b h n d valid]
  omega

theorem actual_read {α : Type} (read : Nat → α) (c : Call) (hc : InputContract c)
    (laws : TensorLaws c) (r : Role) (b h n d : Nat) (valid : Valid (parameters c) r b h n d) :
    logical read c r b h n d=read (sourceAddress c r b h n d) :=
  congrArg read (actual_address c hc laws r b h n d valid)

/-- The bias source uses byte-contiguous columns directly. Only its nontrivial
last dimension forces stride one; no canonical stride is imposed on singleton
batch, head or row dimensions. -/
theorem bias_read {α : Type} (read : Nat → α) (c : Call) (hc : InputContract c)
    (laws : TensorLaws c) (b h n d : Nat) (valid : Valid (parameters c) .bias b h n d) :
    read (base c .bias+2*(offset (c .bias).strides b h n 0+d))=
      logical read c .bias b h n d := by
  have live : Live (parameters c) := ⟨by have:=valid.1;omega,
    by have:=valid.2.1;omega,by have:=valid.2.2.1;omega⟩
  have contiguous : (c .bias).metadata.isContiguous=Bool.true := by
    rcases hc.1 with ⟨_,_,_,_,_,_,_,_,_,_,_,h,_⟩
    exact h
  rw [actual_read read c hc laws .bias b h n d valid]
  simp [sourceAddress,offset,laws.biasDim contiguous live]

#axiomsClean check_iff
#axiomsClean admitted_shapes
#axiomsClean actual_address
#axiomsClean actual_read
#axiomsClean bias_read
#specNonCircular InputContract avoiding [Whole.invoke,FP.Structural.exec,Whole.runPrograms]
end
end VeriTile.Bench.Repairs.AttentionKernel.KernelOnly
