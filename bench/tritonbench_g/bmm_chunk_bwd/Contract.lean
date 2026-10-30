import bench.tritonbench_g.bmm_chunk_bwd.Kernels
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.BmmChunkBwd
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
noncomputable section

inductive Role where | a | dout | residual | output deriving DecidableEq, Repr
structure Metadata where
  tensor : Role → VeriTile.Torch.TensorMetadata
  tensorStrides : Role → TypedSource.TensorStrides
  doutStrides : TypedSource.DoutStrides
  hasResidual : Bool
  hasOutput : Bool

def Used (p : Metadata) (r : Role) : Prop := r=.a ∨ r=.dout ∨
  (r=.residual ∧ p.hasResidual=Bool.true) ∨ (r=.output ∧ p.hasOutput=Bool.true)
def base (p : Metadata) (r : Role) := (p.tensor r).dataPtr.toNat

def grouped (p : Metadata) := decide ((p.tensor .a).shape.length=4)
def batch (p : Metadata) := (p.tensor .a).shape.getD 0 0
def seqlen (p : Metadata) := (p.tensor .a).shape.getD 1 0
def groups (p : Metadata) := if grouped p then (p.tensor .a).shape.getD 2 0 else 1
def width (p : Metadata) := (p.tensor .a).shape.getD (if grouped p then 3 else 2) 0
def chunks (p : Metadata) := (p.tensor .dout).shape.getD 1 0
def chunk (p : Metadata) := (p.tensor .dout).shape.getLast?.getD 0
def canonical (p : Metadata) : TypedSource.TensorStrides :=
  ⟨seqlen p*groups p*width p,groups p*width p,if grouped p then width p else 0,1⟩
def canonicalDout (p : Metadata) : TypedSource.DoutStrides :=
  ⟨chunks p*groups p*chunk p*chunk p,groups p*chunk p*chunk p,
    if grouped p then chunk p*chunk p else 0,chunk p,1⟩
def call (p : Metadata) : HostCall where
  A := "A"
  Dout := "Dout"
  Out := "Out"
  Residual := "Residual"
  batch := batch p
  seqlen := seqlen p
  K := width p
  nchunks := chunks p
  chunk := chunk p
  groups := groups p
  hasGroups := grouped p
  hasResidual := p.hasResidual
  aStrides := p.tensorStrides .a
  doutStrides := p.doutStrides
  outStrides := if p.hasOutput then p.tensorStrides .output else canonical p
  residualStrides := p.tensorStrides .residual

def Ranks (p : Metadata) : Prop := ((p.tensor .a).shape.length=3 ∨ (p.tensor .a).shape.length=4) ∧
  (p.tensor .dout).shape.length=(p.tensor .a).shape.length+1
def Shapes (p : Metadata) : Prop := (p.tensor .dout).shape=(call p).doutShape ∧
  chunks p=(seqlen p+chunk p-1)/chunk p
def ResidualShape (p : Metadata) : Prop := p.hasResidual=Bool.true → (p.tensor .residual).shape=(p.tensor .a).shape
def OutputShape (p : Metadata) : Prop := p.hasOutput=Bool.true → (p.tensor .output).shape=(p.tensor .a).shape

def TensorStrideEq (a b : TypedSource.TensorStrides) : Prop :=
  a.batch=b.batch ∧ a.sequence=b.sequence ∧ a.head=b.head ∧ a.feature=b.feature
def DoutStrideEq (a b : TypedSource.DoutStrides) : Prop :=
  a.batch=b.batch ∧ a.chunk=b.chunk ∧ a.head=b.head ∧ a.row=b.row ∧ a.column=b.column
def Strides (p : Metadata) : Prop :=
  (∀ r,Used p r → r≠.dout → (p.tensorStrides r).feature=1 ∧
    (0<(p.tensor r).numel → TensorStrideEq (p.tensorStrides r) (canonical p))) ∧
  p.doutStrides.column=1 ∧ (0<(p.tensor .dout).numel → DoutStrideEq p.doutStrides (canonicalDout p))
def Extents (p : Metadata) : Prop := ∀ r,Used p r → 4*(p.tensor r).numel<2^63
def Grid (p : Metadata) : Prop := (p.tensor .a).numel≠0 → batch p<2^16 ∧ chunks p*groups p<2^16 ∧
  ∀ c : Fin 9,((chunk p+(tuning c).BM-1)/(tuning c).BM)*((width p+(tuning c).BN-1)/(tuning c).BN)<2^31
def Separate (p : Metadata) : Prop := ∀ r s,Used p r → Used p s → r≠s →
  base p r+4*(p.tensor r).numel≤base p s ∨ base p s+4*(p.tensor s).numel≤base p r

structure InputContract (p : Metadata) : Prop where
  layout : ∀ r,Used p r → (p.tensor r).layout=.strided
  dtype : ∀ r,Used p r → (p.tensor r).dtype=.floating .fp32
  cuda : ∀ r,Used p r → (p.tensor r).device.kind="cuda"
  device : ∀ r,Used p r → (p.tensor r).device=(p.tensor .a).device
  ranks : Ranks p
  chunk : 0<chunk p
  shapes : Shapes p
  residual : ResidualShape p
  output : OutputShape p
  contiguous : ∀ r,Used p r → (p.tensor r).isContiguous=Bool.true
  strides : Strides p
  noNeg : ∀ r,Used p r → (p.tensor r).isNeg=Bool.false
  extents : Extents p
  grid : Grid p
  alignment : ∀ r,Used p r → 0<(p.tensor r).numel → base p r%4=0
  separate : Separate p

inductive InputError where
  | tensors | layout | dtype | cuda | device | ranks | chunk | shapes | residual | output
  | contiguous | strides | negative | extent | grid | alignment | overlap
  deriving DecidableEq, Repr
open Classical in
def checkInputs (p : Metadata) : Except InputError Unit :=
  if ¬(∀ r,Used p r → (p.tensor r).layout=.strided) then .error .layout
  else if ¬(∀ r,Used p r → (p.tensor r).dtype=.floating .fp32) then .error .dtype
  else if ¬(∀ r,Used p r → (p.tensor r).device.kind="cuda") then .error .cuda
  else if ¬(∀ r,Used p r → (p.tensor r).device=(p.tensor .a).device) then .error .device
  else if ¬Ranks p then .error .ranks
  else if ¬0<chunk p then .error .chunk
  else if ¬Shapes p then .error .shapes
  else if ¬ResidualShape p then .error .residual
  else if ¬OutputShape p then .error .output
  else if ¬(∀ r,Used p r → (p.tensor r).isContiguous=Bool.true) then .error .contiguous
  else if ¬Strides p then .error .strides
  else if ¬(∀ r,Used p r → (p.tensor r).isNeg=Bool.false) then .error .negative
  else if ¬Extents p then .error .extent
  else if ¬Grid p then .error .grid
  else if ¬(∀ r,Used p r → 0<(p.tensor r).numel → base p r%4=0) then .error .alignment
  else if ¬Separate p then .error .overlap
  else .ok ()
def checkArguments (p : Option Metadata) : Except InputError Unit :=
  match p with | none => .error .tensors | some p => checkInputs p
def Admitted (p : Option Metadata) : Prop :=
  match p with | none => False | some p => InputContract p
end
end VeriTile.Bench.Repairs.BmmChunkBwd
