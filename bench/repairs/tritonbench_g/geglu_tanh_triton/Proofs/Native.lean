import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.Scalar
import bench.tritonbench_g.swiglu_backward.Proofs.Observed

namespace VeriTile.Bench.Repairs.GegluTanhTriton.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term graphAlgebra)
noncomputable section
set_option maxRecDepth 40000
set_option maxHeartbeats 8000000

/-- Local source-operation ports. The tanh port is expanded into the exact
libdevice instruction branch graph below; FMA remains genuinely ternary. -/
structure Primitives (α : Type) where
  literal : ℝ → α
  cast : α → α
  tanh : α → α
  neg : α → α
  add : α → α → α
  mul : α → α → α
  fma : α → α → α → α
  unused : α

/-- Compiler-selected graph rules, not arithmetic identities. They retain
the actual operand orders and four external FMA sites observed in backward. -/
def interpret (m : Primitives α) : Term α → α
  | .input x => x
  | .literal (some .fp32) .real x => m.literal x
  | .literal none .real x => m.literal x
  | .cast (some .fp32) .real .real x => m.cast (interpret m x)
  | .unary (some .fp32) .libdeviceTanh x => m.tanh (interpret m x)
  | .binary (some .fp32) .real .add a
      (.binary (some .fp32) .real .mul (.literal p d k) b) =>
      m.fma (interpret m b) (interpret m (.literal p d k)) (interpret m a)
  | .binary (some .fp32) .real .add (.literal p d k)
      (.binary (some .fp32) .real .mul a b) =>
      m.fma (interpret m a) (interpret m b) (interpret m (.literal p d k))
  | .binary (some .fp32) .real .add
      (.binary (some .fp32) .real .mul (.literal p d k) a) b =>
      m.fma (interpret m a) (interpret m (.literal p d k)) (interpret m b)
  | .binary (some .fp32) .real .sub (.literal p d k)
      (.binary (some .fp32) .real .mul a b) =>
      m.fma (m.neg (interpret m a)) (interpret m b) (interpret m (.literal p d k))
  | .binary (some .fp32) .real .mul a
      (.binary (some .fp32) .real .mul (.literal p d k) b) =>
      m.mul (m.mul (interpret m b) (interpret m (.literal p d k))) (interpret m a)
  | .binary (some .fp32) .real .mul a
      (.binary (some .fp32) .real .mul
        (.binary (some .fp32) .real .mul (.literal p d k) b)
        (.binary (some .fp32) .real .add c e)) =>
      m.mul (m.mul (m.mul (interpret m b) (interpret m (.literal p d k)))
        (interpret m (.binary (some .fp32) .real .add c e))) (interpret m a)
  | .binary (some .fp32) .real .mul (.literal p d k) a =>
      m.mul (interpret m a) (interpret m (.literal p d k))
  | .binary (some .fp32) .real .add (.literal p d k) a =>
      m.add (interpret m a) (interpret m (.literal p d k))
  | .binary (some .fp32) .real .mul a b => m.mul (interpret m a) (interpret m b)
  | .binary (some .fp32) .real .add a b => m.add (interpret m a) (interpret m b)
  | _ => m.unused

/-- Independent resolved scalar graph, with original FP32 precision and
source constants. No split multiply/add equality or reassociation is used. -/
def reference (m : Primitives α) (ch : Channel) (a b dc : α) : α :=
  let a := m.cast a
  let cube := m.mul (m.mul a a) a
  let z := m.mul (m.fma cube (m.literal 0.044715) a) (m.literal 0.7978845608028654)
  let t := m.tanh z
  let halfA := m.mul a (m.literal 0.5)
  let plus := m.add t (m.literal 1)
  let g := m.mul halfA plus
  match ch with
  | .c => m.mul g b
  | .db => m.mul g dc
  | .da =>
      let derivative := m.mul
        (m.mul (m.fma (m.mul a (m.literal 0.134145)) a (m.literal 1))
          (m.literal 0.7978845608028654))
        (m.mul halfA (m.fma (m.neg t) t (m.literal 1)))
      m.mul (m.mul dc b) (m.fma plus (m.literal 0.5) derivative)

def sourceScalar (m : Primitives α) (ch : Channel) (a b dc : α) : α :=
  interpret m (Tile.scalar graphAlgebra ch (.input a) (.input b) (.input dc))

theorem graph_lowering (m : Primitives α) (ch : Channel) (a b dc : α) :
    sourceScalar m ch a b dc=reference m ch a b dc := by
  cases ch <;> rfl

abbrev Word := BitVec 32
def word (n : Nat) : Word := BitVec.ofNat 32 n

/-- Exact scalar instructions including the libdevice FTZ sites. Raw32 inputs
retain NaN payloads, infinity, signed zero, subnormals and overflow outcomes. -/
structure Instructions where
  addRN : Word → Word → Word
  mulRN : Word → Word → Word
  fmaRN : Word → Word → Word → Word
  fmaRNFTZ : Word → Word → Word → Word
  absFTZ : Word → Word
  neg : Word → Word
  exp2ApproxFTZ : Word → Word
  reciprocalApproxFTZ : Word → Word
  ltUnordered : Word → Word → Bool
  geOrdered : Word → Word → Bool

/-- Actual __nv_tanhf branch graph in all pinned CUDA/sm90 offline profiles.
The unordered comparison deliberately preserves the NaN branch behavior. -/
def tanhGraph (p : Instructions) (x : Word) : Word :=
  let absolute := p.absFTZ x
  if p.ltUnordered absolute (word 0x3f19999a) then
    let square := p.mulRN x x
    let c1 := p.fmaRNFTZ (word 0x3c80f082) square (word 0xbd563cae)
    let c2 := p.fmaRNFTZ c1 square (word 0x3e085941)
    let c3 := p.fmaRNFTZ c2 square (word 0xbeaaa9ed)
    let c4 := p.fmaRNFTZ c3 square (word 0x00000000)
    p.fmaRNFTZ c4 x x
  else
    let exponential := p.exp2ApproxFTZ (p.mulRN absolute (word 0x4038aa3b))
    let reciprocal := p.reciprocalApproxFTZ (p.addRN exponential (word 0x3f800000))
    let approximation := p.fmaRNFTZ reciprocal (word 0xc0000000) (word 0x3f800000)
    let magnitude := if p.geOrdered absolute (word 0x41102cb4) then word 0x3f800000 else approximation
    (x &&& word 0x80000000) ||| magnitude

def compiledPorts (p : Instructions) : Primitives Word where
  literal := fun x =>
    if x=1 then word 0x3f800000
    else if x=0.5 then word 0x3f000000
    else if x=0.7978845608028654 then word 0x3f4c422a
    else if x=0.044715 then word 0x3d372713
    else if x=0.134145 then word 0x3e095d4f
    else word 0
  cast := id
  tanh := tanhGraph p
  neg := p.neg
  add := p.addRN
  mul := p.mulRN
  fma := p.fmaRN
  unused := word 0

def compiledReference (p : Instructions) (ch : Channel) (a b dc : Word) : Word :=
  let cube := p.mulRN (p.mulRN a a) a
  let z := p.mulRN (p.fmaRN cube (word 0x3d372713) a) (word 0x3f4c422a)
  let t := tanhGraph p z
  let halfA := p.mulRN a (word 0x3f000000)
  let plus := p.addRN t (word 0x3f800000)
  let g := p.mulRN halfA plus
  match ch with
  | .c => p.mulRN g b
  | .db => p.mulRN g dc
  | .da =>
      let factor := p.mulRN
        (p.mulRN (p.fmaRN (p.mulRN a (word 0x3e095d4f)) a (word 0x3f800000))
          (word 0x3f4c422a))
        (p.mulRN halfA (p.fmaRN (p.neg t) t (word 0x3f800000)))
      p.mulRN (p.mulRN dc b) (p.fmaRN plus (word 0x3f000000) factor)

theorem compiled_lowering (p : Instructions) (ch : Channel) (a b dc : Word) :
    sourceScalar (compiledPorts p) ch a b dc=compiledReference p ch a b dc := by
  rw [graph_lowering]
  cases ch <;> norm_num [reference,compiledPorts,compiledReference]

/-- Optional model-conformance interface for individual instruction ports.
These unchanged primitives are model boundaries, not numerical experiment
obligations for integer index commutation. Public kernel results use one
instruction model and require no such premise. -/
structure PrimitiveObligations (native target : Instructions) : Prop where
  addRN : ∀ a b,native.addRN a b=target.addRN a b
  mulRN : ∀ a b,native.mulRN a b=target.mulRN a b
  fmaRN : ∀ a b c,native.fmaRN a b c=target.fmaRN a b c
  fmaRNFTZ : ∀ a b c,native.fmaRNFTZ a b c=target.fmaRNFTZ a b c
  absFTZ : ∀ a,native.absFTZ a=target.absFTZ a
  neg : ∀ a,native.neg a=target.neg a
  exp2ApproxFTZ : ∀ a,native.exp2ApproxFTZ a=target.exp2ApproxFTZ a
  reciprocalApproxFTZ : ∀ a,native.reciprocalApproxFTZ a=target.reciprocalApproxFTZ a
  ltUnordered : ∀ a b,native.ltUnordered a b=target.ltUnordered a b
  geOrdered : ∀ a b,native.geOrdered a b=target.geOrdered a b

theorem tanh_bridge (native target : Instructions) (h : PrimitiveObligations native target)
    (x : Word) : tanhGraph native x=tanhGraph target x := by
  simp only [tanhGraph,h.addRN,h.mulRN,h.fmaRNFTZ,h.absFTZ,h.exp2ApproxFTZ,
    h.reciprocalApproxFTZ,h.ltUnordered,h.geOrdered]

theorem primitive_bridge (native target : Instructions) (h : PrimitiveObligations native target)
    (ch : Channel) (a b dc : Word) :
    sourceScalar (compiledPorts native) ch a b dc=compiledReference target ch a b dc := by
  rw [compiled_lowering]
  cases ch <;> simp only [compiledReference,h.mulRN,h.addRN,h.fmaRN,h.neg,tanh_bridge native target h]

#specNonCircular compiledReference avoiding [interpret,sourceScalar,Tile.scalar]
#axiomsClean graph_lowering
#axiomsClean compiled_lowering
#axiomsClean primitive_bridge
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.GegluTanhTriton.Native
