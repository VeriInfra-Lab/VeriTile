import bench.repairs.tritonbench_g.rope_embedding.Proofs.Native
import bench.repairs.tritonbench_g.rope_embedding.Proofs.NativeGraph
namespace VeriTile.Bench.Repairs.RopeEmbedding.Numerics
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
def decode (w : Word) : ℝ := (w.toRat?.getD 0 : ℝ)
def Finite (w : Word) : Prop := w.toRat?.isSome=Bool.true
def ExactValue (w : Word) (v : ℝ) : Prop := Finite w ∧ decode w=v
def UnaryExact (op : Word→Word) (realOp : ℝ→ℝ) (x : Word) : Prop := Finite x ∧ ExactValue (op x) (realOp (decode x))
def BinaryExact (op : Word→Word→Word) (realOp : ℝ→ℝ→ℝ) (x y : Word) : Prop :=
  Finite x ∧ Finite y ∧ ExactValue (op x y) (realOp (decode x) (decode y))
def FMAExact (op : Word→Word→Word→Word) (x y z : Word) : Prop :=
  Finite x ∧ Finite y ∧ Finite z ∧ ExactValue (op x y z) (decode x*decode y+decode z)

/-- Seven local, reached-node groups. The source -sin subtraction and final
product sign-bit negation are deliberately separate. No completed pair,
reduction, memory or returned result is an assumption. -/
structure LocalExact (p : Primitives) (back second : Bool) (a b c s : Word) : Prop where
  zero : back=Bool.true → ExactValue p.zero 0
  sineSub : back=Bool.true → BinaryExact p.sub (·-·) p.zero s
  firstMultiply : second=Bool.false → BinaryExact p.mul (·*·) (signed p back s) b
  negateFirst : second=Bool.false → UnaryExact p.neg (-·) (p.mul (signed p back s) b)
  firstFMA : second=Bool.false → FMAExact p.fma c a (p.neg (p.mul (signed p back s) b))
  secondMultiply : second=Bool.true → BinaryExact p.mul (·*·) c b
  secondFMA : second=Bool.true → FMAExact p.fma (signed p back s) a (p.mul c b)

theorem signed_decode (p : Primitives) (back second : Bool) (a b c s : Word)
    (h : LocalExact p back second a b c s) :
    decode (signed p back s)=if back then -decode s else decode s := by
  cases hb : back
  · rfl
  · simp only [signed,hb,↓reduceIte]
    rw [(h.sineSub hb).2.2.2,(h.zero hb).2]
    ring

theorem pair_decode (p : Primitives) (back second : Bool) (a b c s : Word)
    (h : LocalExact p back second a b c s) :
    decode (pair p back second a b c s)=
      if second then (Spec.pair back (decode a) (decode b) (decode c) (decode s)).2
      else (Spec.pair back (decode a) (decode b) (decode c) (decode s)).1 := by
  have hs := signed_decode p back second a b c s h
  cases hi : second
  · simp only [pair,hi,Bool.false_eq_true,↓reduceIte]
    rw [(h.firstFMA hi).2.2.2.2,(h.negateFirst hi).2.2,(h.firstMultiply hi).2.2.2,hs]
    cases back <;> simp [Spec.pair] <;> ring
  · simp only [pair,hi,↓reduceIte]
    rw [(h.secondFMA hi).2.2.2.2,(h.secondMultiply hi).2.2.2,hs]
    cases back <;> simp [Spec.pair] <;> ring

def TaskExact (p : Primitives) (r : Request) (back : Bool) (q c s : Nat→Word) : Prop :=
  ∀ a h col,a<r.rows → h<r.heads → col<2*r.half →
    let j := if col<r.half then col else col-r.half
    LocalExact p back (decide (r.half≤col)) (q (r.offset a h j)) (q (r.offset a h (j+r.half)))
      (c (a%r.sequence*r.half+j)) (s (a%r.sequence*r.half+j))

theorem result_decode (p : Primitives) (r : Request) (back : Bool) (q c s : Nat→Word)
    (nodes : TaskExact p r back q c s) (a h col : Nat) (ha : a<r.rows) (hh : h<r.heads) :
    decode (result p r back q c s a h col)=
      Spec.result r back (fun o => decode (q o)) (fun o => decode (c o)) (fun o => decode (s o)) a h col := by
  unfold result Spec.result
  split
  · rw [pair_decode p back _ _ _ _ _ (nodes a h col ha hh ‹_›)]
    split_ifs <;> simp_all <;> omega
  · rfl

def wordByte (w : Word) (k : Fin 4) : BitVec 8 := BitVec.ofNat 8 (w.bits.toNat/2^(8*k.val))

#specNonCircular LocalExact avoiding [Source.source,Native.entry,Memory.run,FP.Structural.exec]
#axiomsClean source_profile
#axiomsClean pair_decode
#axiomsClean result_decode
end
end VeriTile.Bench.Repairs.RopeEmbedding.Numerics
