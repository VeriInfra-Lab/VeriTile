import bench.tritonbench_g.bmm_chunk_fwd.Kernels
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.BmmChunkFwd
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section

inductive Role where | a | b | seq deriving DecidableEq, Repr
structure Metadata where
  tensor : Role → VeriTile.Torch.TensorMetadata
  strides : Role → Host.Strides
  hasSeq : Bool
  chunkIsInt : Bool
  chunk : Int
  causalIsBool : Bool
  causal : Bool
  outputDType : Option VeriTile.Torch.DType

def Used (p : Metadata) (r : Role) : Prop := r≠.seq ∨ p.hasSeq=Bool.true
def bytes : Role → Nat | .a | .b => 4 | .seq => 8
def base (p : Metadata) (r : Role) := (p.tensor r).dataPtr.toNat
def shape (p : Metadata) : Host.Shape :=
  let grouped := decide ((p.tensor .a).shape.length=4)
  ⟨(p.tensor .a).shape.getD 0 0,(p.tensor .a).shape.getD 1 0,
    if grouped then (p.tensor .a).shape.getD 2 0 else 1,
    (p.tensor .a).shape.getD (if grouped then 3 else 2) 0,p.chunk.toNat,grouped⟩
def call (p : Metadata) : Host.Call where
  shape := shape p
  aStrides := p.strides .a
  bStrides := p.strides .b
  A := "A"
  B := "B"
  Out := "Out"
  Seq := ⟨"Seq"⟩
  seqBatch := (p.strides .seq).batch
  seqRow := (p.strides .seq).row
  causal := p.causal
  hasSeq := p.hasSeq
  requestedOutput := if p.outputDType.isSome then some .fp32 else none

/-- Literal source stride observations. Empty inputs have no loaded cells;
last stride one keeps the original conditional contiguous operations inactive. -/
def Strides (p : Metadata) : Prop :=
  ((p.strides .a).column=1 ∧ (0<(p.tensor .a).numel → p.strides .a=Host.contiguous (shape p))) ∧
  ((p.strides .b).column=1 ∧ (0<(p.tensor .b).numel → p.strides .b=Host.contiguous (shape p))) ∧
  (p.hasSeq=Bool.true → 0<(p.tensor .seq).numel →
    (p.strides .seq).batch=(shape p).seqlen ∧ (p.strides .seq).row=1)

def Types (p : Metadata) : Prop := (p.tensor .a).dtype=.floating .fp32 ∧
  (p.tensor .b).dtype=.floating .fp32 ∧ (p.hasSeq=Bool.true → (p.tensor .seq).dtype=.other "int64")
def OutputType (p : Metadata) : Prop := p.outputDType=none ∨ p.outputDType=some (.floating .fp32)
def Shapes (p : Metadata) : Prop :=
  ((p.tensor .a).shape.length=3 ∨ (p.tensor .a).shape.length=4) ∧
  (p.tensor .b).shape=(p.tensor .a).shape
def SeqShape (p : Metadata) : Prop := p.hasSeq=Bool.true → (p.tensor .seq).shape=(p.tensor .a).shape.take 2
def Extents (p : Metadata) : Prop :=
  (∀ r,Used p r → bytes r*(p.tensor r).numel<2^63) ∧ 4*(shape p).outputSize<2^63
def Grid (p : Metadata) : Prop := (shape p).outputSize≠0 →
  (shape p).batch<2^16 ∧ (shape p).chunks*(shape p).groups<2^16 ∧
  ∀ c∈Host.configurations, Host.cdiv (shape p).chunk c.M*Host.cdiv (shape p).chunk c.N<2^31
def Separate (p : Metadata) : Prop := ∀ r s, Used p r → Used p s → r≠s →
  base p r+bytes r*(p.tensor r).numel≤base p s ∨ base p s+bytes s*(p.tensor s).numel≤base p r

structure InputContract (p : Metadata) : Prop where
  layout : ∀ r,Used p r → (p.tensor r).layout=.strided
  dtype : Types p
  output : OutputType p
  cuda : ∀ r,Used p r → (p.tensor r).device.kind="cuda"
  device : ∀ r,Used p r → (p.tensor r).device=(p.tensor .a).device
  shapes : Shapes p
  sequence : SeqShape p
  contiguous : ∀ r,Used p r → (p.tensor r).isContiguous=Bool.true
  strides : Strides p
  noNeg : ∀ r,Used p r → (p.tensor r).isNeg=Bool.false
  chunk : p.chunkIsInt=Bool.true ∧ 0<p.chunk
  causal : p.causalIsBool=Bool.true
  extents : Extents p
  grid : Grid p
  alignment : ∀ r,Used p r → 0<(p.tensor r).numel → base p r%bytes r=0
  separate : Separate p

inductive InputError where
  | tensors | layout | dtype | output | cuda | device | shapes | sequence | contiguous
  | strides | negative | chunk | causal | extent | grid | alignment | overlap
  deriving DecidableEq, Repr
open Classical in
def checkInputs (p : Metadata) : Except InputError Unit :=
  if ¬(∀ r,Used p r → (p.tensor r).layout=.strided) then .error .layout
  else if ¬Types p then .error .dtype
  else if ¬OutputType p then .error .output
  else if ¬(∀ r,Used p r → (p.tensor r).device.kind="cuda") then .error .cuda
  else if ¬(∀ r,Used p r → (p.tensor r).device=(p.tensor .a).device) then .error .device
  else if ¬Shapes p then .error .shapes
  else if ¬SeqShape p then .error .sequence
  else if ¬(∀ r,Used p r → (p.tensor r).isContiguous=Bool.true) then .error .contiguous
  else if ¬Strides p then .error .strides
  else if ¬(∀ r,Used p r → (p.tensor r).isNeg=Bool.false) then .error .negative
  else if ¬(p.chunkIsInt=Bool.true ∧ 0<p.chunk) then .error .chunk
  else if ¬p.causalIsBool=Bool.true then .error .causal
  else if ¬Extents p then .error .extent
  else if ¬Grid p then .error .grid
  else if ¬(∀ r,Used p r → 0<(p.tensor r).numel → base p r%bytes r=0) then .error .alignment
  else if ¬Separate p then .error .overlap
  else .ok ()
def checkArguments (p : Option Metadata) : Except InputError Unit :=
  match p with | none => .error .tensors | some p => checkInputs p
def Admitted (p : Option Metadata) : Prop :=
  match p with | none => False | some p => InputContract p
end
end VeriTile.Bench.Repairs.BmmChunkFwd
