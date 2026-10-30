import bench.tritonbench_g.rotary_transform.Proofs.HistoricalWhole
import bench.tritonbench_g.rotary_emb.Contract

namespace VeriTile.Bench.Repairs.RotaryTransform.Checked

abbrev Tensor := VeriTile.Bench.Repairs.RotaryEmb.Checked.Tensor

/-- Observed host shapes and original tensor descriptors. Metadata values are
decoded from caller bytes by configuration, never accepted as a Config oracle. -/
structure Call where
  host : Host.Call
  views : Physical.InputViews
  input : Tensor
  cosine : Tensor
  sine : Tensor
  cumulative : Tensor
  offsets : Tensor

def configuration (c : Call) (initial : Physical.Bytes) : Config := Physical.callerConfiguration c.host.shape c.host.abi c.views initial

def shapeCheck (s : Shape) : Bool :=
  decide (0<s.heads ∧ 0<s.width ∧ True ∧ 2*s.half≤s.width)

def metadataCheck (c : Config) : Bool :=
  decide (c.metadata.cumulative 0=0) &&
  decide (c.shape.total=if c.shape.varlen then c.metadata.cumulative c.shape.batches
    else c.shape.batches*c.shape.maximum) &&
  decide (∀ i : Fin c.shape.batches,
    c.metadata.cumulative i.val≤c.metadata.cumulative (i.val+1) ∧ c.length i.val≤c.shape.maximum)

def indexCheck (c : Config) : Bool :=
  let s := c.shape
  decide (Host.size s+8*s.heads*s.width+s.blockK<2^31 ∧ s.maximum+8<2^31 ∧
    s.batches+1<2^31 ∧ s.coefficientRows*s.half<2^31) &&
  decide (∀ b : Fin s.batches,c.shift b.val+s.maximum+8<2^31 ∧
    (c.shift b.val+s.maximum+8)*s.half+s.blockK<2^31)

def ordinary (t : Tensor) (device : Nat) (dtype : String) (unit : Nat) : Bool :=
  t.cuda && t.strided && !t.negative && !t.conjugate &&
  decide (t.device=device ∧ t.dtype=dtype ∧ t.elementBytes=unit ∧ t.storageOffset=0)

def span (t : Tensor) : Nat :=
  (List.zipWith (fun n stride => (n-1)*stride) t.shape t.strides).sum+1

def fits (t : Tensor) : Bool :=
  decide (t.shape.prod=0 ∨ (t.base=t.allocationBase ∧ t.base%t.elementBytes=0 ∧
    t.elementBytes*span t≤t.allocationBytes ∧ t.base+t.allocationBytes≤2^64))

def separate (a b : Tensor) : Prop :=
  a.shape.prod=0 ∨ b.shape.prod=0 ∨
  a.allocationBase+a.allocationBytes≤b.allocationBase ∨
  b.allocationBase+b.allocationBytes≤a.allocationBase
instance (a b : Tensor) : Decidable (separate a b) := inferInstanceAs (Decidable (_∨_∨_∨_))

def tensors (c : Call) : List Tensor := [c.input,c.cosine,c.sine] ++
  (if c.host.shape.varlen then [c.cumulative] else []) ++
  (if c.host.shape.offsetTensor then [c.offsets] else [])

def canonical : List Nat → List Nat
  | [] => []
  | _::tail => (tail.map (max 1)).prod :: canonical tail

def payloadShape (s : Shape) : List Nat :=
  if s.varlen then [s.total,s.heads,s.width] else [s.batches,s.maximum,s.heads,s.width]

def tensorCheck (c : Call) : Bool :=
  let s := c.host.shape
  let device := c.host.device
  ordinary c.input device "float32" 4 && ordinary c.cosine device "float32" 4 &&
  ordinary c.sine device "float32" 4 && c.input.contiguous &&
  decide (c.input.shape=payloadShape s ∧ c.cosine.shape=[s.coefficientRows,s.half] ∧
    c.sine.shape=c.cosine.shape ∧ c.input.object=c.host.inputIdentity ∧
    (c.input.shape.prod=0 ∨ c.input.strides=canonical c.input.shape)) &&
  decide (c.cosine.strides=[c.views.cosineRow,c.views.cosineColumn] ∧
    c.sine.strides=[c.views.sineRow,c.views.sineColumn] ∧
    (c.host.shape.offsetTensor=true → c.offsets.strides=[c.views.offsetsStride])) &&
  decide (0<c.views.cosineRow ∧ 0<c.views.cosineColumn ∧ 0<c.views.sineRow ∧
    0<c.views.sineColumn ∧ 0<c.views.offsetsStride) &&
  decide (c.host.abi.cumulativeStride=1) &&
  (if s.varlen then ordinary c.cumulative device
    (if c.host.abi.cumulative64 then "int64" else "int32") (if c.host.abi.cumulative64 then 8 else 4) &&
    decide (c.cumulative.shape=[s.batches+1] ∧ c.cumulative.strides=[1]) else true) &&
  (if s.offsetTensor then ordinary c.offsets device
    (if c.host.abi.offsets64 then "int64" else "int32") (if c.host.abi.offsets64 then 8 else 4) &&
    decide (c.offsets.shape=[s.batches]) else true) &&
  (tensors c).all fits && decide ((tensors c).Pairwise separate)

def check (c : Call) (initial : Physical.Bytes) : Bool :=
  shapeCheck c.host.shape && tensorCheck c && metadataCheck (configuration c initial) &&
  indexCheck (configuration c initial)

end VeriTile.Bench.Repairs.RotaryTransform.Checked
