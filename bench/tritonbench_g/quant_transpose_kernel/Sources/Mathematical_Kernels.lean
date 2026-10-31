import bench.tritonbench_g.quantize_global.Sources.Mathematical_Kernels

/-! Source definitions shared by the independent mathematical and floating
proofs. Production input precision is unspecified, so the task selects FP32;
the output allocation explicitly uses int8. Duplicate autotune entries and the
source's pid % group_size schedule are preserved. -/
namespace VeriTile.Bench.TritonBenchG.QuantTransposeKernel
open VeriTile.Triton

end VeriTile.Bench.TritonBenchG.QuantTransposeKernel

namespace VeriTile.Bench.TritonBenchG.QuantTransposeKernel
open VeriTile.Triton

end VeriTile.Bench.TritonBenchG.QuantTransposeKernel

namespace QuantizeIntegerStage
open VeriTile.Triton

end QuantizeIntegerStage

/-! Dtype-independent source coordinates and wrapper metadata. The historical
namespace is retained to preserve the accepted public statements. -/
namespace QuantTransposeFinite
def gridM (m : Nat) := (m+127)/128
def gridN (n : Nat) := (n+127)/128

def groupWidth (gn : Nat) := 8*gn
def groupId (gn pid : Nat) := pid/groupWidth gn
def groupSize (gm gn pid : Nat) := min (gm-groupId gn pid*8) 8
def tileRow (gm gn pid : Nat) := groupId gn pid*8+pid%groupSize gm gn pid
def tileCol (gm gn pid : Nat) := (pid%groupWidth gn)/groupSize gm gn pid

/-- Source pid uses pid%group_size, not residual%group_size. The final
partial group may therefore rotate the row order. -/
def inversePid (gm gn r c : Nat) : Nat :=
  let g := r/8
  let s := min (gm-g*8) 8
  let a := g*(8*gn)
  let rotation := a%s
  let u := if rotation≤r%8 then r%8-rotation else s+r%8-rotation
  a+c*s+u

def row (m n pid lane : Nat) := tileRow (gridM m) (gridN n) pid*128+lane/128
def col (m n pid lane : Nat) := tileCol (gridM m) (gridN n) pid*128+lane%128

/-- Two distinct decorator entries with identical actual launch metadata. -/
inductive Config where | first | second
  deriving DecidableEq, Repr

def Config.blockM (_ : Config) : Nat := 128
def Config.blockN (_ : Config) : Nat := 128
def Config.groupM (_ : Config) : Nat := 8
def Config.warps (_ : Config) : Nat := 4
def Config.stages (_ : Config) : Nat := 3

structure Parameters where
  M : Nat
  N : Nat
  sam : Nat
  san : Nat
  x : String
  scale : String
  output : String

def Parameters.n (p : Parameters) := p.M*p.N
def programs (p : Parameters) (_ : Config) : Nat := gridM p.M*gridN p.N
def offset (p : Parameters) (pid lane : Nat) := col p.M p.N pid lane*p.M+row p.M p.N pid lane
def inputOffset (p : Parameters) (i : Nat) := (i%p.M)*p.sam+(i/p.M)*p.san
def laneInput (p : Parameters) (pid lane : Nat) := row p.M p.N pid lane*p.sam+col p.M p.N pid lane*p.san
def active (p : Parameters) (pid lane : Nat) : Prop := row p.M p.N pid lane<p.M ∧ col p.M p.N pid lane<p.N
instance (p : Parameters) (pid lane : Nat) : Decidable (active p pid lane) := inferInstanceAs (Decidable (_ ∧ _))

def laneIndices (p : Parameters) (pid : Nat) (lanes : List Nat) :=
  lanes.filterMap (fun lane => if active p pid lane then some (offset p pid lane) else none)

def footprint (p : Parameters) (pids : List Nat) : List Nat :=
  pids.flatMap (fun pid => laneIndices p pid (List.range (128*128)))

structure Input where
  shape : List Nat
  strides : List Nat
  /-- Relative to the actual x.data_ptr(), including view stride effects. -/
  logicalOffsets : List Nat
  isCuda : Bool
  deviceIndex : Nat
  storageBase : Nat
  storageOffset : Nat
  storageLength : Nat

def Input.numel (a : Input) := a.shape.prod
def Input.M (a : Input) := a.shape.headD 0
def Input.N (a : Input) := a.shape[1]?.getD 0
def Input.sam (a : Input) := a.strides.headD 0
def Input.san (a : Input) := a.strides[1]?.getD 0

/-- Ordinary Torch nonnegative shape/stride metadata and tensor numel fit
signed64. This primitive metadata contract permits every runtime i32/i64
specialization and imposes no arithmetic or active-address safety condition. -/
def MetadataDomain (a : Input) : Prop :=
  a.M<2^63 ∧ a.N<2^63 ∧ a.sam<2^63 ∧ a.san<2^63 ∧ a.numel<2^63

def Input.outputShape (a : Input) := [a.N,a.M]

def viewOffsets : List Nat → List Nat → List Nat
  | [],[] => [0]
  | n::ns,t::ts => (List.range n).flatMap (fun i => (viewOffsets ns ts).map (fun o => i*t+o))
  | _,_ => []

structure Allocations where
  maximum : String
  scale : String
  output : String
  /-- Fresh torch.empty bytes need not be zero. -/
  initialOutput : Nat → BitVec 8
  maximumBase : Nat
  scaleBase : Nat
  outputBase : Nat
  currentDevice : Nat

def Allocations.Fresh (a : Allocations) (x : String) : Prop :=
  x≠a.maximum ∧ x≠a.scale ∧ x≠a.output ∧
  a.maximum≠a.scale ∧ a.maximum≠a.output ∧ a.scale≠a.output

def parameters (a : Input) (alloc : Allocations) (x : String) : Parameters :=
  ⟨a.M,a.N,a.sam,a.san,x,alloc.scale,alloc.output⟩

inductive Stage where
  | abs | maximum | unsqueeze | reciprocal | shapeUnpack | outputAllocation | outputShapeAssertion | inputStrideAssertion | outputStrideAssertion | signature | compilation | driverContext | gridParse | launchEnterHook | pointerPacking | launch | launchExitHook
  deriving DecidableEq, Repr

/-- Primitive exceptions are an explicit trusted runtime outcome, distinct
from source diagnostics. The no-failure profile is stated when proving success. -/
abbrev Environment := Stage → Option String

end QuantTransposeFinite
