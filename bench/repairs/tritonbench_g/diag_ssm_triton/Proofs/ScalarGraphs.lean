import bench.repairs.tritonbench_g.diag_ssm_triton.Kernels
import VeriTile.Triton.Float.ScalarOps

/-! Optional scalar graph interpretations retained from the earlier byte/host
proofs. Kernel acceptance uses the canonical typed DSL, not this graph model. -/
set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Kernels
open VeriTile.Triton

inductive Route where
  | real32 | complex64
  deriving DecidableEq, Repr

def Route.components : Route → Nat
  | .real32 => 1
  | .complex64 => 2

inductive Graph where
  | separateSource
  | sm80DefaultFusion
  deriving DecidableEq, Repr

/-- Total primitive operations. Memory/execution/frame laws are derived
separately and are not fields of this ordinary scalar backend interface. -/
structure Operations (α : Type) where
  zero : α
  add : α → α → α
  subtract : α → α → α
  multiply : α → α → α
  fma : α → α → α → α
  flipSign : α → α

abbrev Word := FP.Value .fp32

/-- Executable concrete all-bit-pattern software profile. It includes NaNs,
infinities, signed zero and subnormals. This is not a GPU conformance theorem. -/
def software (cfg : FP.Config) : Operations Word where
  zero := FP.Value.zero .fp32
  add := FP.add cfg
  subtract := FP.sub cfg
  multiply := FP.mul cfg
  fma := FP.fma cfg
  flipSign := FP.Value.neg

structure Pair (α : Type) where
  re : α
  im : α
  deriving DecidableEq

def zeroPair {α : Type} (ops : Operations α) : Pair α := ⟨ops.zero, ops.zero⟩

def realForward {α : Type} (graph : Graph) (ops : Operations α) (s transition x : α) : α :=
  match graph with
  | .separateSource => ops.add (ops.multiply s transition) x
  | .sm80DefaultFusion => ops.fma s transition x

/-- The product's second multiplication is separately rounded in both fused
complex components; the first multiplication participates in the FMA. -/
def complexProduct {α : Type} (graph : Graph) (ops : Operations α) (a b : Pair α) : Pair α :=
  match graph with
  | .separateSource =>
    ⟨ops.subtract (ops.multiply a.re b.re) (ops.multiply a.im b.im),
     ops.add (ops.multiply a.re b.im) (ops.multiply a.im b.re)⟩
  | .sm80DefaultFusion =>
    ⟨ops.fma a.re b.re (ops.flipSign (ops.multiply a.im b.im)),
     ops.fma a.re b.im (ops.multiply a.im b.re)⟩

def complexForward {α : Type} (graph : Graph) (ops : Operations α)
    (s transition x : Pair α) : Pair α :=
  let product := complexProduct graph ops s transition
  ⟨ops.add product.re x.re, ops.add product.im x.im⟩

def forward {α : Type} (route : Route) (graph : Graph) (ops : Operations α)
    (s transition x : Pair α) : Pair α :=
  match route with
  | .real32 => ⟨realForward graph ops s.re transition.re x.re, ops.zero⟩
  | .complex64 => complexForward graph ops s transition x

structure BackwardRegisters (α : Type) where
  carry : Pair α
  transition : Pair α

structure BackwardStep (α : Type) where
  registers : BackwardRegisters α
  inputGradient : Pair α

def initialBackward {α : Type} (ops : Operations α) : BackwardRegisters α :=
  ⟨zeroPair ops, zeroPair ops⟩

def accumulatedGradient {α : Type} (graph : Graph) (ops : Operations α)
    (loaded carried : α) : α :=
  match graph with
  | .separateSource => ops.add loaded carried
  | .sm80DefaultFusion => ops.add carried loaded

def realBackward {α : Type} (graph : Graph) (ops : Operations α)
    (previous transition gy : α) (registers : BackwardRegisters α) : BackwardStep α :=
  let a := accumulatedGradient graph ops gy registers.carry.re
  let gl := match graph with
    | .separateSource => ops.add registers.transition.re (ops.multiply a previous)
    | .sm80DefaultFusion => ops.fma a previous registers.transition.re
  ⟨⟨⟨ops.multiply a transition, ops.zero⟩, ⟨gl, ops.zero⟩⟩, ⟨a, ops.zero⟩⟩

/-- Source unary minus is lowered to arithmetic subtraction from positive
zero at gradient loads/stores. It is not the sign-bit operation used inside
the fused complex multiplication graph. This distinction includes signed zero. -/
def gradientConjugate {α : Type} (ops : Operations α) (g : Pair α) : Pair α :=
  ⟨g.re, ops.subtract ops.zero g.im⟩

def complexBackward {α : Type} (graph : Graph) (ops : Operations α)
    (previous transition gy : Pair α) (registers : BackwardRegisters α) : BackwardStep α :=
  let loaded := gradientConjugate ops gy
  let a := Pair.mk (accumulatedGradient graph ops loaded.re registers.carry.re)
    (accumulatedGradient graph ops loaded.im registers.carry.im)
  let product := complexProduct graph ops a previous
  let gl := Pair.mk (ops.add registers.transition.re product.re)
    (ops.add registers.transition.im product.im)
  let carry := complexProduct graph ops a transition
  ⟨⟨carry, gl⟩, gradientConjugate ops a⟩

def backward {α : Type} (route : Route) (graph : Graph) (ops : Operations α)
    (previous transition gy : Pair α) (registers : BackwardRegisters α) : BackwardStep α :=
  match route with
  | .real32 => realBackward graph ops previous.re transition.re gy.re registers
  | .complex64 => complexBackward graph ops previous transition gy registers

def terminalGradient {α : Type} (route : Route) (ops : Operations α) (g : Pair α) : Pair α :=
  match route with
  | .real32 => g
  | .complex64 => gradientConjugate ops g

end VeriTile.Bench.Repairs.DiagSsmTriton.Kernels
