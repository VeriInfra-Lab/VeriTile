import VeriTile.Triton

/-! Shared source definitions for the original quantize_global program.
The wrapper has no input dtype assertion; the selected task precision is FP32.
The output allocation explicitly uses int8. Mathematical Real and floating
proofs consume these definitions independently; no BF16 variant is introduced.
-/
namespace QuantizationSource
/-- Source parenthesization and conversion sequence, interpreted by each
carrier without changing the Python program. This is syntax, not a result law. -/
def scalar {Arithmetic Integer64 Output : Type}
    (input inverse literal127 : Arithmetic)
    (multiply : Arithmetic → Arithmetic → Arithmetic)
    (round64 : Arithmetic → Integer64) (narrow8 : Integer64 → Output) : Output :=
  narrow8 (round64 (multiply literal127 (multiply input inverse)))
end QuantizationSource

namespace VeriTile.Bench.TritonBenchG.QuantizeGlobal
open VeriTile.Triton

end VeriTile.Bench.TritonBenchG.QuantizeGlobal

namespace VeriTile.Bench.TritonBenchG.QuantizeGlobal
open VeriTile.Triton

end VeriTile.Bench.TritonBenchG.QuantizeGlobal

namespace QuantizeIntegerStage
open VeriTile.Triton

end QuantizeIntegerStage

/-! Dtype-independent original wrapper/grid metadata, shared by Real and FP. -/
namespace QuantizeGlobalFinite
inductive Config where | block1024 | block2048
  deriving DecidableEq, Repr

def Config.block : Config → Nat | .block1024 => 1024 | .block2048 => 2048
/-- Includes constructor defaults, not invented explicit decorator arguments. -/
def Config.warps (_ : Config) : Nat := 4
def Config.stages : Config → Nat | .block1024 => 3 | .block2048 => 1

def programs (n : Nat) (c : Config) : Nat := (n+c.block-1)/c.block
def offset (c : Config) (pid lane : Nat) := pid*c.block+lane

structure Parameters where
  n : Nat
  x : String
  scale : String
  output : String

def footprint (c : Config) (pids : List Nat) : List Nat :=
  pids.flatMap (fun pid => (List.range c.block).map (offset c pid))

structure Input where
  shape : List Nat
  strides : List Nat
  /-- Relative to the actual x.data_ptr(), including view stride effects. -/
  logicalOffsets : List Nat
  isCuda : Bool
  storageOffset : Nat
  storageLength : Nat
  /-- CUDA ordinal owning the actual input view. -/
  deviceIndex : Nat := 0

def Input.numel (a : Input) := a.shape.prod

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
  /-- Device selected by torch.empty(device="cuda") at this invocation. -/
  currentDevice : Nat := 0

def Allocations.Fresh (a : Allocations) (x : String) : Prop :=
  x≠a.maximum ∧ x≠a.scale ∧ x≠a.output ∧
  a.maximum≠a.scale ∧ a.maximum≠a.output ∧ a.scale≠a.output

def parameters (a : Input) (alloc : Allocations) (x : String) : Parameters :=
  ⟨a.numel,x,alloc.scale,alloc.output⟩

inductive Stage where
  | abs | maximum | unsqueeze | reciprocal | outputAllocation | cudaAssertion | signature | compilation | launch
  deriving DecidableEq, Repr

/-- Primitive exceptions are an explicit trusted runtime outcome, distinct
from source diagnostics. The no-failure profile is stated when proving success. -/
abbrev Environment := Stage → Option String

def contiguousStrides : List Nat → List Nat
  | [] => []
  | _::ns => ns.prod::contiguousStrides ns

end QuantizeGlobalFinite
