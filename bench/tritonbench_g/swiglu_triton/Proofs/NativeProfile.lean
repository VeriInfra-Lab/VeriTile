import bench.tritonbench_g.swiglu_triton.Proofs.Native
import bench.tritonbench_g.swiglu_triton.Proofs.Source

namespace VeriTile.Bench.Repairs.SwigluTriton.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term graphAlgebra)
open Memory Physical
noncomputable section

abbrev Word := BitVec 32

def positiveZero : Word := BitVec.ofNat 32 0x00000000
def one : Word := BitVec.ofNat 32 0x3f800000
def log2e : Word := BitVec.ofNat 32 0x3fb8aa3b

/-- Precision/configuration key of the actual compiler-resolved instruction
profile. The 14 evidence samples instantiate CUDA/sm90/Triton3.7.1 and the
unchanged source options; other profiles are not inferred from those samples. -/
structure Profile where
  backend : String
  architecture : Nat
  compiler : String
  fpFusion : Bool
  flushToZero : Bool
  columns : Nat
  block : Nat
  warps : Nat
  stride : Nat

/-- Independently specified scalar instruction semantics. Their exact
native-to-target correspondence is the only numerical boundary:
round-to-nearest FP32 add/sub/mul/FMA, PTX div.full.f32, ex2.approx.f32.
All ports take raw32 words, retaining nonfinite values, NaNs and signed zeros. -/
structure InstructionSemantics where
  addRN : Word → Word → Word
  subRN : Word → Word → Word
  mulRN : Word → Word → Word
  fmaRN : Word → Word → Word → Word
  divFull : Word → Word → Word
  exp2Approx : Word → Word

/-- Exact sigmoid subgraph observed in every pinned PTX profile. Negation is
sub(+0,a), not an assumed real negation, and log2(e) is the encoded constant. -/
def sigmoidGraph (p : InstructionSemantics) (a : Word) : Word :=
  p.divFull one (p.addRN (p.exp2Approx (p.mulRN (p.subRN positiveZero a) log2e)) one)

def compiledPorts (p : InstructionSemantics) : Primitives Word where
  literal := fun x => if x=1 then one else positiveZero
  cast := id
  sigmoid := sigmoidGraph p
  sub := p.subRN
  mul := p.mulRN
  fma := p.fmaRN
  unused := positiveZero

/-- This independent FP reference explicitly fixes the compiled fused
policy and all operand orders; it is not source split arithmetic. -/
def compiledReference (p : InstructionSemantics) (ch : Channel) (a b dc : Word) : Word :=
  let s := p.divFull one (p.addRN (p.exp2Approx (p.mulRN (p.subRN positiveZero a) log2e)) one)
  let silu := p.mulRN s a
  match ch with
  | .c => p.mulRN silu b
  | .da => p.mulRN (p.mulRN (p.fmaRN silu (p.subRN one s) s) dc) b
  | .db => p.mulRN silu dc

theorem compiled_lowering (p : InstructionSemantics) (ch : Channel) (a b dc : Word) :
    sourceScalar (compiledPorts p) ch a b dc=compiledReference p ch a b dc := by
  rw [graph_lowering]
  cases ch <;> simp [reference,compiledReference,compiledPorts,sigmoidGraph]

/-- Exact local obligations for one resolved profile. No whole sigmoid,
SwiGLU scalar, memory operation, launch or host result is an assumption. -/
structure PrimitiveObligations (profile : Profile) (native target : InstructionSemantics) : Prop where
  addRN : ∀ a b,native.addRN a b=target.addRN a b
  subRN : ∀ a b,native.subRN a b=target.subRN a b
  mulRN : ∀ a b,native.mulRN a b=target.mulRN a b
  fmaRN : ∀ a b c,native.fmaRN a b c=target.fmaRN a b c
  divFull : ∀ a b,native.divFull a b=target.divFull a b
  exp2Approx : ∀ a,native.exp2Approx a=target.exp2Approx a

theorem primitive_bridge (profile : Profile) (native target : InstructionSemantics)
    (h : PrimitiveObligations profile native target) (ch : Channel) (a b dc : Word) :
    sourceScalar (compiledPorts native) ch a b dc=compiledReference target ch a b dc := by
  rw [compiled_lowering]
  cases ch <;> simp only [compiledReference,h.addRN,h.subRN,h.mulRN,h.fmaRN,h.divFull,h.exp2Approx]

/-- Same actual caller/private word reads become atoms in the original
full typed source execution. Only node interpretation supplies instructions. -/
def graphState {M β : Type} (ops : Ops M Word β) (backward : Bool)
    (p : Layout) (first : Nat) (s : M) : State (Term Word) where
  mem := fun reg offset => .mk .real (.input (
    if reg="A" then ops.read s ((p.allocation .a).key (first*p.block+offset))
    else if reg="B" then ops.read s ((p.allocation .b).key (first*p.block+offset))
    else if reg=Source.region backward .third then ops.read s ((p.allocation .third).key (first*p.block+offset))
    else positiveZero))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun dtype _ _ => defaultValue dtype

theorem graphState_read {M β : Type} (ops : Ops M Word β) (backward : Bool)
    (p : Layout) (first : Nat) (s : M) (role : Role) (offset : Nat) :
    (graphState ops backward p first s).mem (Source.region backward role) offset=
      Cell.mk .real (.input (ops.read s ((p.allocation role).key (first*p.block+offset)))) := by
  cases backward <;> cases role <;> simp [graphState,Source.region]

#specNonCircular compiledReference avoiding [interpret,sourceScalar,Tile.scalar]
#axiomsClean primitive_bridge
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.Native
