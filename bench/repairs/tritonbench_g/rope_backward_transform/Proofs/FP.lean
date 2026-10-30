import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.Grid
import VeriTile.Triton.Float.BitValue
namespace VeriTile.Bench.Repairs.RopeBackwardTransform.Numerics
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
abbrev Word := FP.Value .fp32

/-- Local primitive implementation parameter for the observed SM80 profile.
Binding these functions to native IEEE instructions is explicitly deferred. -/
structure Primitives where
  zero : Word
  (add sub mul : Word→Word→Word)
  neg : Word→Word
  fma : Word→Word→Word→Word
structure Tagged where
  word : Word
  product : Option (Word×Word)
instance : Inhabited Tagged := ⟨⟨FP.Value.zero .fp32,none⟩⟩
def leaf (w : Word) : Tagged := ⟨w,none⟩

/-- Both branches contract the left product, with coefficient-first operands.
The high result negates the rounded right product before FMA. -/
def operations (p : Primitives) : Operations Tagged where
  zero := leaf p.zero
  cast := fun x => leaf x.word
  mul := fun x y => ⟨p.mul y.word x.word,some (y.word,x.word)⟩
  add := fun x y => leaf (match x.product with | some (a,b) => p.fma a b y.word | none => p.add x.word y.word)
  sub := fun x y => leaf (match x.product with | some (a,b) => p.fma a b (p.neg y.word) | none => p.sub x.word y.word)
def pair (p : Primitives) (second : Bool) (a b c s : Word) :=
  if second then p.fma c b (p.neg (p.mul s a)) else p.fma c a (p.mul s b)

theorem source_pair (p : Primitives) (second : Bool) (a b c s : Tagged) :
    (scalarPair (operations p) second a b c s).word=pair p second a.word b.word c.word s.word := by
  cases second <;> rfl

def result (p : Primitives) (r : Request) (q : Bool→Nat→Word) (c s : Nat→Word)
    (key : Bool) (a h col : Nat) :=
  if col<2*r.half then
    let j := if col<r.half then col else col-r.half
    pair p (decide (r.half≤col)) (q key (r.offset key a h j)) (q key (r.offset key a h (j+r.half)))
      (c (a%r.sequence*r.cosRowStride+j*r.cosColStride))
      (s (a%r.sequence*r.sinRowStride+j*r.sinColStride))
  else q key (r.offset key a h col)

theorem source_profile (p : Primitives) (r : Request) (m : Source.Memory Tagged)
    (key : Bool) (a h col : Nat) :
    (scalarResult (operations p) r m key a h col).word=result p r
      (fun key o => ((m (region key) o).read .fp32).word)
      (fun o => ((m "C" o).read .fp32).word) (fun o => ((m "S" o).read .fp32).word) key a h col := by
  unfold scalarResult result
  split
  · exact source_pair p _ _ _ _ _
  · rfl

def decode (w : Word) : ℝ := (w.toRat?.getD 0 : ℝ)
def Finite (w : Word) : Prop := w.toRat?.isSome=Bool.true
def ExactValue (w : Word) (v : ℝ) := Finite w ∧ decode w=v
def UnaryExact (op : Word→Word) (realOp : ℝ→ℝ) (x : Word) := Finite x ∧ ExactValue (op x) (realOp (decode x))
def BinaryExact (op : Word→Word→Word) (realOp : ℝ→ℝ→ℝ) (x y : Word) :=
  Finite x ∧ Finite y ∧ ExactValue (op x y) (realOp (decode x) (decode y))
def FMAExact (op : Word→Word→Word→Word) (x y z : Word) :=
  Finite x ∧ Finite y ∧ Finite z ∧ ExactValue (op x y z) (decode x*decode y+decode z)

/-- Five reached local operation families, never a completed pair/output premise. -/
structure LocalExact (p : Primitives) (second : Bool) (a b c s : Word) : Prop where
  firstMultiply : second=Bool.false → BinaryExact p.mul (·*·) s b
  firstFMA : second=Bool.false → FMAExact p.fma c a (p.mul s b)
  secondMultiply : second=Bool.true → BinaryExact p.mul (·*·) s a
  negateSecond : second=Bool.true → UnaryExact p.neg (-·) (p.mul s a)
  secondFMA : second=Bool.true → FMAExact p.fma c b (p.neg (p.mul s a))

theorem pair_decode (p : Primitives) (second : Bool) (a b c s : Word)
    (h : LocalExact p second a b c s) :
    decode (pair p second a b c s)=if second then (Spec.pair (decode a) (decode b) (decode c) (decode s)).2
      else (Spec.pair (decode a) (decode b) (decode c) (decode s)).1 := by
  cases hi : second
  · simp only [pair,hi,Bool.false_eq_true,↓reduceIte]
    rw [(h.firstFMA hi).2.2.2.2,(h.firstMultiply hi).2.2.2]
    simp [Spec.pair]; ring
  · simp only [pair,hi,↓reduceIte]
    rw [(h.secondFMA hi).2.2.2.2,(h.negateSecond hi).2.2,(h.secondMultiply hi).2.2.2]
    simp [Spec.pair]; ring

def TaskExact (p : Primitives) (r : Request) (q : Bool→Nat→Word) (c s : Nat→Word) :=
  ∀ key a h col,a<r.rows → h<r.heads key → col<2*r.half →
    let j := if col<r.half then col else col-r.half
    LocalExact p (decide (r.half≤col)) (q key (r.offset key a h j)) (q key (r.offset key a h (j+r.half)))
      (c (a%r.sequence*r.cosRowStride+j*r.cosColStride))
      (s (a%r.sequence*r.sinRowStride+j*r.sinColStride))

def wordByte (w : Word) (k : Fin 4) : BitVec 8 := BitVec.ofNat 8 (w.bits.toNat/2^(8*k.val))

#specNonCircular LocalExact avoiding [Source.kernel,scalarPair,scalarResult,pair,result,FP.Structural.exec]
#axiomsClean source_profile
#axiomsClean pair_decode
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.Numerics
