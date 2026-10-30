import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.Scalar
import VeriTile.Triton.Float.BitValue

namespace VeriTile.Bench.Repairs.RbeTritonTransform.Ordered
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

#guard_msgs (drop info) in
#specNonCircular result avoiding [Source.source,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransform.Ordered
