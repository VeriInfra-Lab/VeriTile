import VeriTile.Torch.Tensor
import bench.tritonbench_g.matmul_leakyrelu.Sources.Kernels
import bench.tritonbench_g.matmul_leakyrelu.Sources.Mathematical_FiniteDomain

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole

structure Metadata where
  a : VeriTile.Torch.TensorMetadata
  b : VeriTile.Torch.TensorMetadata
  am : Nat
  ak : Nat
  bk : Nat
  bn : Nat

def request (p : Metadata) (activation : String) : Request where
  A := "a"; B := "b"; C := "c"
  rows := p.a.shape[0]?.getD 0
  inner := p.a.shape[1]?.getD 0
  innerB := p.b.shape[0]?.getD 0
  columns := p.b.shape[1]?.getD 0
  strideAM := p.am; strideAK := p.ak; strideBK := p.bk; strideBN := p.bn
  aContiguous := p.a.isContiguous
  activation := activation

def metadataChecks (p : Metadata) : Bool :=
  decide (p.a.layout = .strided ∧ p.b.layout = .strided ∧
    p.a.shape.length = 2 ∧ p.b.shape.length = 2 ∧
    p.a.dtype = .floating .fp32 ∧ p.b.dtype = .floating .fp32 ∧
    p.a.device.kind = "cuda" ∧ p.b.device.kind = "cuda" ∧ p.a.device = p.b.device ∧
    p.a.shape[1]?.getD 0 = p.b.shape[0]?.getD 0 ∧ p.a.isContiguous = Bool.true ∧
    p.a.isNeg = Bool.false ∧ p.b.isNeg = Bool.false ∧
    (p.a.numel=0 ∨ p.a.dataPtr.toNat%4=0) ∧
    (p.b.numel=0 ∨ p.b.dataPtr.toNat%4=0) ∧
    p.a.dataPtr.toNat+4*((request p "").rows*p.am+(paddedInner (request p "")+32)*p.ak)<2^64 ∧
    p.b.dataPtr.toNat+4*((paddedInner (request p "")+32)*p.bk+(request p "").columns*p.bn)<2^64)

/-- Exact original finite source envelope, including padded final advances. -/
def indexChecks (p : Metadata) : Bool :=
  let r := request p ""
  decide (r.rows+32 < signedLimit ∧ r.columns+32 < signedLimit ∧ r.inner+32 < signedLimit ∧
    r.strideAM < signedLimit ∧ r.strideAK < signedLimit ∧ r.strideBK < signedLimit ∧ r.strideBN < signedLimit ∧
    programs r < signedLimit ∧ 4*tileColumns r < signedLimit ∧
    r.rows*r.strideAM+(paddedInner r+32)*r.strideAK < signedLimit ∧
    (paddedInner r+32)*r.strideBK+r.columns*r.strideBN < signedLimit ∧
    (tileRows r*32)*max r.columns 1+tileColumns r*32 < signedLimit)

structure InputContract (p : Metadata) : Prop where
  metadata : metadataChecks p = Bool.true
  indices : IndexDomain (request p "")

inductive InputError where
  | tensors | activation | negativeStride | metadata | indices | outputAddress
  deriving DecidableEq, Repr

def checkInputs (p : Metadata) : Except InputError Unit :=
  if !metadataChecks p then .error .metadata
  else if !indexChecks p then .error .indices
  else .ok ()

/-- Exact post-allocation check before launch, distinct from input admission. -/
def outputCheck (p : Metadata) (baseC : Nat) : Bool :=
  decide (baseC+2*((tileRows (request p "")*32)*max (request p "").columns 1+
    tileColumns (request p "")*32)<2^64)

inductive Arguments where
  | nonTensors
  | nonStringActivation
  | negativeStrides
  | tensors (metadata : Metadata) (activation : String)

def checkArguments (args : Arguments) : Except InputError Unit :=
  match args with
  | .nonTensors => .error .tensors
  | .nonStringActivation => .error .activation
  | .negativeStrides => .error .negativeStride
  | .tensors p _ => checkInputs p

def Admitted (args : Arguments) : Prop :=
  match args with
  | .tensors p _ => InputContract p
  | _ => False

end VeriTile.Bench.Repairs.MatmulLeakyrelu
