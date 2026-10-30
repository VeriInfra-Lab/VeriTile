import bench.repairs.tritonbench_g.rope_embedding.Proofs.Spec
import VeriTile.Triton.Float.BitValue
namespace VeriTile.Bench.Repairs.RopeEmbedding.Numerics
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
abbrev Word := FP.Value .fp32

/-- Scalar primitives for the observed default-fusion profile. IEEE instruction semantics and this source-specific compiler lowering are
the explicit implementation model boundary, not task rewrite assumptions. -/
structure Primitives where
  zero : Word
  add : Word→Word→Word
  sub : Word→Word→Word
  mul : Word→Word→Word
  neg : Word→Word
  fma : Word→Word→Word→Word
structure Tagged where
  word : Word
  product : Option (Word×Word)
instance : Inhabited Tagged := ⟨⟨FP.Value.zero .fp32,none⟩⟩
def leaf (w : Word) : Tagged := ⟨w,none⟩

/-- Same-FP32 loads/casts retain the word. Compiler contraction preserves
coefficient-first operands, contracts the left product in subtraction and the
right product in addition. Source unary -sin lowered to sub(+0,sin). -/
def operations (p : Primitives) : Operations Tagged where
  zero := leaf p.zero
  load := fun x => leaf x.word
  cast := fun x => leaf x.word
  neg := fun x => leaf (p.sub p.zero x.word)
  mul := fun x y => ⟨p.mul y.word x.word,some (y.word,x.word)⟩
  add := fun x y => leaf (match y.product with | some (a,b) => p.fma a b x.word | none => p.add x.word y.word)
  sub := fun x y => leaf (match x.product with | some (a,b) => p.fma a b (p.neg y.word) | none => p.sub x.word y.word)

def signed (p : Primitives) (back : Bool) (s : Word) := if back then p.sub p.zero s else s
def pair (p : Primitives) (back second : Bool) (a b c s : Word) :=
  if second then p.fma (signed p back s) a (p.mul c b)
  else p.fma c a (p.neg (p.mul (signed p back s) b))
def result (p : Primitives) (r : Request) (back : Bool) (q c s : Nat→Word) (a h col : Nat) :=
  if col<2*r.half then
    let j := if col<r.half then col else col-r.half
    pair p back (decide (r.half≤col)) (q (r.offset a h j)) (q (r.offset a h (j+r.half)))
      (c (a%r.sequence*r.half+j)) (s (a%r.sequence*r.half+j))
  else q (r.offset a h col)

theorem source_pair (p : Primitives) (back second : Bool) (a b c s : Tagged) :
    (scalarPair (operations p) back a b c s second).word=pair p back second a.word b.word c.word s.word := by
  cases back <;> cases second <;> rfl

theorem source_profile (p : Primitives) (r : Request) (back : Bool) (q c s : Nat→Tagged) (a h col : Nat) :
    (scalarResult (operations p) r back q c s a h col).word=
      result p r back (fun o => (q o).word) (fun o => (c o).word) (fun o => (s o).word) a h col := by
  unfold scalarResult result
  split
  · exact source_pair p back _ _ _ _ _
  · rfl

/-- Concrete FP32 primitive interpretation. Rounding, FTZ and NaN policy are
chosen once by cfg; FMA rounds once. No real-arithmetic exactness is assumed. -/
def softwarePrimitives (cfg : FP.Config) : Primitives where
  zero := FP.Value.zero .fp32
  add := FP.add cfg
  sub := FP.sub cfg
  mul := FP.mul cfg
  neg := FP.Value.neg
  fma := FP.fma cfg

#axiomsClean source_pair
#axiomsClean source_profile
end
end VeriTile.Bench.Repairs.RopeEmbedding.Numerics
