import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.Native
import VeriTile.Triton.Float.BitValue

namespace VeriTile.Bench.Repairs.RbeTritonTransform.Numerics
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
abbrev Word := FP.Value .fp32

/-- The observed default-fusion profile contracts the left multiply in
subtraction and the right multiply in addition, with trig-first operands. Every field is a scalar primitive, never a pair,
row, kernel, memory or returned-output oracle. Native bindings stay explicit. -/
structure Primitives where
  literal : ℝ → Word
  natural : Nat → Word
  integer : Int → Word
  add : Word → Word → Word
  sub : Word → Word → Word
  mul : Word → Word → Word
  div : Word → Word → Word
  pow : Word → Word → Word
  neg : Word → Word
  sin : Word → Word
  cos : Word → Word
  fma : Word → Word → Word → Word

/-- Product provenance permits local contraction at the following source
add/sub. Only `word` is written to physical memory. This is a source-expression
lowering profile, not a change to Python arithmetic order or fusion options. -/
structure Tagged where
  word : Word
  product : Option (Word × Word)
instance : Inhabited Tagged := ⟨⟨FP.Value.zero .fp32,none⟩⟩
def leaf (w : Word) : Tagged := ⟨w,none⟩

def operations (p : Primitives) : Operations Tagged where
  literal := fun r => leaf (p.literal r)
  natural := fun n => leaf (p.natural n)
  integer := fun n => leaf (p.integer n)
  mul := fun x y => ⟨p.mul y.word x.word,some (y.word,x.word)⟩
  add := fun x y => leaf (match y.product with
    | some (a,b) => p.fma a b x.word | none => p.add x.word y.word)
  sub := fun x y => leaf (match x.product with
    | some (a,b) => p.fma a b (p.neg y.word) | none => p.sub x.word y.word)
  div := fun x y => leaf (p.div x.word y.word)
  pow := fun x y => leaf (p.pow x.word y.word)
  sin := fun x => leaf (p.sin x.word)
  cos := fun x => leaf (p.cos x.word)

def exponent (p : Primitives) (n : Nat) := p.div (p.natural (n%128)) (p.literal 128)
def power (p : Primitives) (n : Nat) := p.pow (p.literal 10000) (exponent p n)
def phase (p : Primitives) (r : Request) (m n : Nat) := p.div (p.integer (r.position+(m : Int))) (power p n)
def partner (p : Primitives) (r : Request) (x : Nat → Word) (b m n : Nat) :=
  if n+1<r.columns then x (r.offset b m (n+1)) else p.literal 0
def roundedInput (p : Primitives) (r : Request) (x : Nat → Word) (b m n : Nat) (im : Bool) :=
  if im then x (r.offset b m n) else partner p r x b m n
def fusedInput (p : Primitives) (r : Request) (x : Nat → Word) (b m n : Nat) (im : Bool) :=
  if im then partner p r x b m n else x (r.offset b m n)
def roundedProduct (p : Primitives) (r : Request) (x : Nat → Word) (b m n : Nat) (im : Bool) :=
  p.mul (p.sin (phase p r m n)) (roundedInput p r x b m n im)
def addend (p : Primitives) (r : Request) (x : Nat → Word) (b m n : Nat) (im : Bool) :=
  if im then roundedProduct p r x b m n im else p.neg (roundedProduct p r x b m n im)
def pair (p : Primitives) (r : Request) (x : Nat → Word) (b m n : Nat) (im : Bool) :=
  p.fma (p.cos (phase p r m n)) (fusedInput p r x b m n im) (addend p r x b m n im)
def result (p : Primitives) (r : Request) (x : Nat → Word) (b m c : Nat) :=
  pair p r x b m (c/2*2) (decide (c%2=1))

theorem source_profile (p : Primitives) (r : Request) (x : Nat → Tagged) (b m c : Nat) :
    (scalarResult (operations p) r x b m c).word = result p r (fun o => (x o).word) b m c := by
  simp only [scalarResult,result,scalarPair,pair,operations,scalarAngle,leaf,phase,power,exponent,
    partner,roundedInput,fusedInput,roundedProduct,addend]
  split <;> split <;> rfl

def decode (w : Word) : ℝ := (w.toRat?.getD 0 : ℝ)
def Finite (w : Word) : Prop := w.toRat?.isSome = Bool.true
def ExactValue (w : Word) (v : ℝ) : Prop := Finite w ∧ decode w=v
def UnaryExact (op : Word → Word) (realOp : ℝ → ℝ) (x : Word) : Prop :=
  Finite x ∧ ExactValue (op x) (realOp (decode x))
def BinaryExact (op : Word → Word → Word) (realOp : ℝ → ℝ → ℝ) (x y : Word) : Prop :=
  Finite x ∧ Finite y ∧ ExactValue (op x y) (realOp (decode x) (decode y))
def FMAExact (op : Word → Word → Word → Word) (x y z : Word) : Prop :=
  Finite x ∧ Finite y ∧ Finite z ∧ ExactValue (op x y z) (decode x*decode y+decode z)

/-- Thirteen local groups on actual primitive operands. No completed phase,
rotation, source execution or physical output equality is a premise. -/
structure LocalExact (p : Primitives) (r : Request) (x : Nat → Word) (b m n : Nat) (im : Bool) : Prop where
  zero : ExactValue (p.literal 0) 0
  dimension : ExactValue (p.literal 128) 128
  theta : ExactValue (p.literal 10000) 10000
  natural : ExactValue (p.natural (n%128)) (n%128 : Nat)
  integer : ExactValue (p.integer (r.position+(m : Int))) ((r.position : ℝ)+(m : ℝ))
  exponentDivision : BinaryExact p.div (·/·) (p.natural (n%128)) (p.literal 128)
  frequencyPower : BinaryExact p.pow Real.rpow (p.literal 10000) (exponent p n)
  phaseDivision : BinaryExact p.div (·/·) (p.integer (r.position+(m : Int))) (power p n)
  cosine : UnaryExact p.cos Real.cos (phase p r m n)
  sine : UnaryExact p.sin Real.sin (phase p r m n)
  roundedMultiply : BinaryExact p.mul (·*·) (p.sin (phase p r m n)) (roundedInput p r x b m n im)
  negateRounded : im=Bool.false → UnaryExact p.neg (fun z => -z) (roundedProduct p r x b m n im)
  fusedMultiplyAdd : FMAExact p.fma (p.cos (phase p r m n))
    (fusedInput p r x b m n im) (addend p r x b m n im)

theorem phase_decode (p : Primitives) (r : Request) (x : Nat → Word) (b m n : Nat) (im : Bool)
    (h : LocalExact p r x b m n im) : decode (phase p r m n)=Spec.angle r m n := by
  unfold phase
  rw [h.phaseDivision.2.2.2]
  unfold power
  rw [h.frequencyPower.2.2.2]
  unfold exponent
  rw [h.exponentDivision.2.2.2,h.natural.2,h.dimension.2,h.theta.2,h.integer.2]
  rfl

theorem pair_decode (p : Primitives) (r : Request) (x : Nat → Word) (b m n : Nat) (im : Bool)
    (h : LocalExact p r x b m n im) :
    decode (pair p r x b m n im)=Spec.pair r (fun o => decode (x o)) b m n im := by
  have ha := phase_decode p r x b m n im h
  have hs : decode (p.sin (phase p r m n))=Real.sin (Spec.angle r m n) := by rw [h.sine.2.2,ha]
  have hc : decode (p.cos (phase p r m n))=Real.cos (Spec.angle r m n) := by rw [h.cosine.2.2,ha]
  have hv : decode (partner p r x b m n)=
      if n+1<r.columns then decode (x (r.offset b m (n+1))) else 0 := by
    unfold partner
    split <;> simp_all only [h.zero.2]
  have hround : decode (roundedProduct p r x b m n im)=
      decode (p.sin (phase p r m n))*decode (roundedInput p r x b m n im) :=
    h.roundedMultiply.2.2.2
  change decode (p.fma _ _ _)=_
  rw [h.fusedMultiplyAdd.2.2.2.2]
  cases him : im
  · have hneg := h.negateRounded him
    simp only [addend,him,Bool.false_eq_true,↓reduceIte] at *
    rw [hneg.2.2,hround]
    simp [fusedInput,roundedInput,him,hs,hc,hv,Spec.pair] at *
    ring
  · simp only [addend,him,↓reduceIte]
    rw [him] at hround
    rw [hround]
    simp [fusedInput,roundedInput,him,hs,hc,hv,Spec.pair]
    ring

def TaskExact (p : Primitives) (r : Request) (x : Nat → Word) : Prop :=
  ∀ b m c, b<r.batches → m<r.rows → c<r.columns →
    LocalExact p r x b m (c/2*2) (decide (c%2=1))

theorem result_decode (p : Primitives) (r : Request) (x : Nat → Word)
    (h : TaskExact p r x) (b m c : Nat) (hb : b<r.batches) (hm : m<r.rows) (hc : c<r.columns) :
    decode (result p r x b m c)=Spec.result r (fun o => decode (x o)) b m c :=
  pair_decode p r x b m (c/2*2) (decide (c%2=1)) (h b m c hb hm hc)

def wordByte (w : Word) (k : Fin 4) : BitVec 8 := BitVec.ofNat 8 (w.bits.toNat / 2^(8*k.val))

#specNonCircular LocalExact avoiding [Source.source,Native.entry,Memory.run,FP.Structural.exec]
#axiomsClean source_profile
#axiomsClean pair_decode
theorem original_input_bits (api : Memory.API Word Store (BitVec 8)) (p : Primitives)
    (r : Request) (input : RegionName) (s : State Tagged) (call : Native.Preparation api leaf r input s)
    (b m c : Nat) (hb : b<r.batches) (hm : m<r.rows) (hc : c<r.columns) :
    result p r (fun o => ((s.mem input o).read .fp32).word) b m c=
      result p r (fun o => api.read call.before (call.input.base+4*o)) b m c := by
  have hn : c/2*2<r.columns := lt_of_le_of_lt (Nat.div_mul_le_self c 2) hc
  have hu := congrArg Tagged.word (call.inputBinding b m (c/2*2) hb hm hn)
  change ((s.mem input (r.offset b m (c/2*2))).read .fp32).word=_ at hu
  by_cases hp : c/2*2+1<r.columns
  · have hv := congrArg Tagged.word (call.inputBinding b m (c/2*2+1) hb hm hp)
    simp [result,pair,addend,roundedProduct,roundedInput,fusedInput,partner,hp,hu,hv,leaf]
  · simp [result,pair,addend,roundedProduct,roundedInput,fusedInput,partner,hp,hu,leaf]

#axiomsClean result_decode
end
end VeriTile.Bench.Repairs.RbeTritonTransform.Numerics
