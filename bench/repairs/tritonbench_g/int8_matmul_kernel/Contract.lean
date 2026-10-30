import bench.tritonbench_g.int8_matmul_kernel.Whole001.Kernels

namespace VeriTile.Bench.Repairs.Int8MatmulKernel
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001

/-- Ordinary rank-two strided Tensor classification precedes this descriptor.
The descriptor retains actual storage identity, offset, strides and device. -/
def MetadataContract (r : Request) (aPointer bPointer : Nat) : Prop :=
  r.a.dtype = .fp32 ∧ r.b.dtype = .uint8 ∧
  r.a.cuda = true ∧ r.b.cuda = true ∧ r.a.device = r.b.device ∧
  r.currentDevice = r.a.device ∧ r.packedShape ∧ r.a.contiguous ∧
  r.a.resolved = true ∧ r.b.resolved = true ∧
  (r.a.rows*r.a.cols=0 ∨ aPointer%4=0) ∧
  aPointer+4*(r.shape.m*r.a.strideRow+(r.shape.k+128)*r.a.strideCol)<2^63 ∧
  bPointer+(r.shape.k/4+128)*r.b.strideRow+r.shape.n*r.b.strideCol<2^63

def metadataChecks (r : Request) (aPointer bPointer : Nat) : Bool :=
  decide (r.a.dtype = .fp32 ∧ r.b.dtype = .uint8 ∧
    r.a.cuda = true ∧ r.b.cuda = true ∧ r.a.device = r.b.device ∧
    r.currentDevice = r.a.device ∧ r.packedShape ∧ r.a.contiguous ∧
    r.a.resolved = true ∧ r.b.resolved = true ∧
  (r.a.rows*r.a.cols=0 ∨ aPointer%4=0) ∧
  aPointer+4*(r.shape.m*r.a.strideRow+(r.shape.k+128)*r.a.strideCol)<2^63 ∧
  bPointer+(r.shape.k/4+128)*r.b.strideRow+r.shape.n*r.b.strideCol<2^63)

/-- Exactly the dimensions, strides, fifteen configurations and offset
envelopes checked in Python. Configuration compilation success is separate. -/
def indexChecks (r : Request) : Bool :=
  let s := r.shape
  let b := hostBuffers r 0
  decide (max s.m (max s.n s.k)+512 < 2^31 ∧
    max b.am (max b.ak (max b.bk b.bn))+256 < 2^31 ∧
    (∀ c : Fin 15, s.programs c < 2^31 ∧ (configs c).gm*s.cols c < 2^31) ∧
    s.m*b.am+(s.k+128)*b.ak < 2^31 ∧
    (s.k/4+128)*b.bk+s.n*b.bn < 2^31 ∧
    (s.m+256)*max s.n 1+s.n+256 < 2^31)

/-- Logical meaning of isfinite followed by truncation and signed8 bounds. -/
def castCheck (word : Word) : Bool :=
  match truncateFloat word with
  | none => false
  | some z => decide (-128 ≤ z ∧ z ≤ 127)

def inputChecks (r : Request) (before : Memory Word) : Bool :=
  (List.range r.a.rows).all fun row => (List.range r.a.cols).all fun k =>
    castCheck (before (aAddress (hostBuffers r 0) row k))

structure InputContract (r : Request) (aPointer bPointer : Nat) (before : Memory Word) : Prop where
  metadata : MetadataContract r aPointer bPointer
  indices : SourceBounds r.shape (hostBuffers r 0)
  defined : InputDefined r.shape (hostBuffers r 0) before

inductive InputError where
  | tensors | layout | rank | negativeStride | metadata | indices | conversion | outputAddress
  deriving DecidableEq, Repr

def checkInputs (r : Request) (aPointer bPointer : Nat) (before : Memory Word) : Except InputError Unit :=
  if !metadataChecks r aPointer bPointer then .error .metadata
  else if !indexChecks r then .error .indices
  else if !inputChecks r before then .error .conversion
  else .ok ()

/-- Exact check after allocating the int32 output, before launch. -/
def outputChecks (r : Request) (cPointer : Nat) : Bool :=
  decide (cPointer+4*((r.shape.m+256)*max r.shape.n 1+r.shape.n+256)<2^63)

/-- Actual allocation-base/data-pointer relation only; no formed-address bound. -/
structure OrdinaryPointers (r : Request) (out aPointer bPointer : Nat) where
  aBase : Nat
  bBase : Nat
  aData : aPointer=aBase+4*r.a.offset
  bData : bPointer=bBase+r.b.offset

/-- The first three Python guards classify the actual dynamic objects before
any two-dimensional Tensor descriptor or element observation is accessed. -/
inductive Arguments where
  | nonTensors | nonStrided | nonMatrix | negativeStrides
  | tensors (request : Request) (aPointer bPointer : Nat)

def checkArguments (args : Arguments) (before : Memory Word) : Except InputError Unit :=
  match args with
  | .nonTensors => .error .tensors
  | .nonStrided => .error .layout
  | .nonMatrix => .error .rank
  | .negativeStrides => .error .negativeStride
  | .tensors r aPointer bPointer => checkInputs r aPointer bPointer before

def Admitted (args : Arguments) (before : Memory Word) : Prop :=
  match args with
  | .tensors r aPointer bPointer => InputContract r aPointer bPointer before
  | _ => False

end VeriTile.Bench.Repairs.Int8MatmulKernel
