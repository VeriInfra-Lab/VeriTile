import bench.repairs.tritonbench_g.swiglu_triton.Proofs.NativeProfile
import VeriTile.Triton.Float.ScalarOps

namespace VeriTile.Bench.Repairs.SwigluTriton.Words
open VeriTile.Triton
open Memory Native
noncomputable section

abbrev Byte := BitVec 8

def finite (word : Word) : Prop := ((⟨word⟩ : FP.Value .fp32).toRat?).isSome

def add (a b : Word) : Word := (FP.add FP.Config.ieeeValue (⟨a⟩ : FP.Value .fp32) ⟨b⟩).bits
def sub (a b : Word) : Word := (FP.sub FP.Config.ieeeValue (⟨a⟩ : FP.Value .fp32) ⟨b⟩).bits
def mul (a b : Word) : Word := (FP.mul FP.Config.ieeeValue (⟨a⟩ : FP.Value .fp32) ⟨b⟩).bits
def fma (a b c : Word) : Word := (FP.fma FP.Config.ieeeValue (⟨a⟩ : FP.Value .fp32) ⟨b⟩ ⟨c⟩).bits

/-- Original approximate/full instruction ports, pinned to the actual resolved
compiler configuration. These are not real exp2 or correctly-rounded division. -/
structure OriginalPorts (profile : Native.Profile) where
  exp2Approx : Word → Word
  divFull : Word → Word → Word

def referenceInstructions (ports : OriginalPorts profile) : InstructionSemantics where
  addRN := add
  subRN := sub
  mulRN := mul
  fmaRN := fma
  divFull := ports.divFull
  exp2Approx := ports.exp2Approx

def reference (ports : OriginalPorts profile) (ch : Channel) (a b dc : Word) : Word :=
  compiledReference (referenceInstructions ports) ch a b dc

/-- Exact scalar numerical deferrals, all basic operations nearest-even with
no FTZ. Nonfinite basic operands retain the unconditional native-word result;
this optional reference bridge does not assert a false all-NaN payload law. -/
structure LocalExact (profile : Native.Profile) (native : InstructionSemantics)
    (ports : OriginalPorts profile) : Prop where
  add : ∀ a b,finite a → finite b → native.addRN a b=add a b
  sub : ∀ a b,finite a → finite b → native.subRN a b=sub a b
  mul : ∀ a b,finite a → finite b → native.mulRN a b=mul a b
  fma : ∀ a b c,finite a → finite b → finite c → native.fmaRN a b c=fma a b c
  exp2 : ∀ a,native.exp2Approx a=ports.exp2Approx a
  division : ∀ a b,native.divFull a b=ports.divFull a b

/-- Every actual basic-operation operand, including intermediates after exp2
and multiplication. Finite leaves alone do not imply this numerical domain. -/
def NumericDomain (p : InstructionSemantics) (a b dc : Word) : Prop :=
  let negative := p.subRN positiveZero a
  let exponent := p.exp2Approx (p.mulRN negative log2e)
  let s := sigmoidGraph p a
  let silu := p.mulRN s a
  let difference := p.subRN one s
  let factor := p.fmaRN silu difference s
  finite a ∧ finite b ∧ finite dc ∧ finite negative ∧ finite exponent ∧ finite s ∧ finite silu ∧
    finite difference ∧ finite factor ∧ finite (p.mulRN factor dc)

theorem reference_exact (profile : Native.Profile) (p : InstructionSemantics) (ports : OriginalPorts profile)
    (laws : LocalExact profile p ports) (a b dc : Word) (domain : NumericDomain p a b dc) (ch : Channel) :
    sourceScalar (compiledPorts p) ch a b dc=reference ports ch a b dc := by
  obtain ⟨ha,hb,hd,hn,he,hs,hsi,hdiff,hfactor,hfirst⟩ := domain
  have hz : finite positiveZero := by unfold finite; decide
  have ho : finite one := by unfold finite; decide
  have hl : finite log2e := by unfold finite; decide
  have negative := laws.sub positiveZero a hz ha
  have scale := laws.mul (p.subRN positiveZero a) log2e hn hl
  have plus := laws.add (p.exp2Approx (p.mulRN (p.subRN positiveZero a) log2e)) one he ho
  have sig : sigmoidGraph p a=sigmoidGraph (referenceInstructions ports) a := by
    unfold sigmoidGraph
    rw [laws.division,plus,laws.exp2,scale,negative]
    rfl
  have silu := laws.mul (sigmoidGraph p a) a hs ha
  have difference := laws.sub one (sigmoidGraph p a) ho hs
  have factor := laws.fma (p.mulRN (sigmoidGraph p a) a) (p.subRN one (sigmoidGraph p a)) (sigmoidGraph p a) hsi hdiff hs
  have first := laws.mul (p.fmaRN (p.mulRN (sigmoidGraph p a) a) (p.subRN one (sigmoidGraph p a)) (sigmoidGraph p a)) dc hfactor hd
  rw [compiled_lowering]
  cases ch
  · change p.mulRN (p.mulRN (sigmoidGraph p a) a) b=_
    rw [laws.mul _ b hsi hb,silu,sig]
    rfl
  · change p.mulRN (p.mulRN (p.fmaRN (p.mulRN (sigmoidGraph p a) a)
      (p.subRN one (sigmoidGraph p a)) (sigmoidGraph p a)) dc) b=_
    rw [laws.mul _ b hfirst hb,first,factor,silu,difference,sig]
    rfl
  · change p.mulRN (p.mulRN (sigmoidGraph p a) a) dc=_
    rw [laws.mul _ dc hsi hd,silu,sig]
    rfl

def signNegation : Negation Word where
  apply := fun word => word ^^^ BitVec.ofNat 32 0x80000000
  involutive := by intro word; simp [BitVec.xor_assoc]

/-- Ordinary little-endian raw word storage. All addresses are actual byte
starts, including unaligned caller spans; Real is never encoded by this layer. -/
structure Codec {M : Type} (ops : Ops M Word Byte) : Prop where
  byte : ∀ s key n,n<4 → ops.observe s key.owner (key.address+n)=
    (ops.read s key).extractLsb' (8*n) 8

theorem logical_bytes {M : Type} (ops : Ops M Word Byte) (codec : Codec ops)
    (s : M) (view : View) (i : Nat) (value : Word)
    (readback : logical ops signNegation s view i=value) (n : Nat) (small : n<4) :
    ops.observe s (view.key i).owner ((view.key i).address+n)=
      (signNegation.encode view.negative value).extractLsb' (8*n) 8 := by
  rw [codec.byte s (view.key i) n small]
  have physical := congrArg (signNegation.encode view.negative) readback
  simp only [logical,Negation.encode_twice] at physical
  rw [physical]

#specNonCircular reference avoiding [Tile.scalar,Tile.sourceRun,Native.interpret,Native.sourceScalar]
#axiomsClean reference_exact
#axiomsClean logical_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.Words
