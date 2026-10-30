import bench.tritonbench_g.chunk_gla_simple.Sources.NativeGraph
import bench.tritonbench_g.chunk_gla_fwd.Proofs.NativeComposition

namespace VeriTile.Bench.Repairs.ChunkGlaSimple.ScalarGraph
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Host
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions
open NativeGraph (Expr)
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 8000000

def reduce (op : Binary) (seed : Expr) (p : Option ComputeDType) {shape : TileShape}
    (ax : Fin shape.length) (keep : Bool) (xs : Values Expr .real shape)
    (i : TileIndex (TileShape.reduceShape shape ax keep)) : Expr := by
  cases keep with
  | «false» => exact (List.finRange (TileShape.axisDim shape ax)).foldl (fun acc j =>
      Expr.binary p .real op acc (xs (TileShape.insertAxisIndex shape ax i j))) seed
  | «true» => exact (List.finRange (TileShape.axisDim shape ax)).foldl (fun acc j =>
      Expr.binary p .real op acc (xs (TileShape.replaceAxisIndex shape ax i j))) seed

def rest : Algebra Expr where
  literal:=Expr.literal
  negInf:=.leaf 0xFF800000
  binary:=Expr.binary
  unary:=Expr.unary
  cast:=Expr.cast
  fromNat:=fun p n=>.literal p .real n
  fromInt:=fun p n=>.literal p .real n
  fp32Bits:=fun b=>.leaf b.bits
  fp32Load:=Expr.load
  reduceSum:=fun p=>reduce .add (.literal p .real 0) p
  reduceMax:=fun p=>reduce .max (.leaf 0xFF800000) p

def algebra : Algebra Expr := NativeGraph.algebra rest
def dots (c : Shape) : Typed.Dots algebra c := NativeGraph.dots rest c

def scalarDot (seed : Graph) : (batch : TileShape) → {m k n : Nat} →
    (TileIndex (batch++[m,k]) → Graph) → (TileIndex (batch++[k,n]) → Graph) →
    TileIndex (batch++[m,n]) → Graph
  | [],_,k,_,a,b,i => ieeeDot k (fun j=>a (i.1,j,())) (fun j=>b (j,i.2.1,())) seed
  | _::ds,_,_,_,a,b,i => scalarDot seed ds (fun j=>a (i.1,j)) (fun j=>b (i.1,j)) i.2

/-- The scale argument position is tagged independently of literal zero. -/
def scaleTag : ℝ := -17

def lower (scale : BitVec 32) : Expr → Graph
  | .leaf word => .bits word
  | .literal p d x => if x=scaleTag then .bits scale else .literal p d x
  | .load x => .move .register (lower scale x)
  | .cast p a b x => .cast p a b (lower scale x)
  | .unary p .exp x => .unary p .exp2 (.binary p .real .mul (lower scale x) (.bits 0x3FB8AA3B))
  | .unary p op x => .unary p op (lower scale x)
  | .binary p d op a b => .binary p d op (lower scale a) (lower scale b)
  | .pendingDot _ _ batch _ _ _ a b i =>
      scalarDot zero batch (fun j=>lower scale (a j)) (fun j=>lower scale (b j)) i
  | .dotAcc _ _ batch _ _ _ a b carry i =>
      scalarDot (lower scale carry) batch (fun j=>lower scale (a j)) (fun j=>lower scale (b j)) i

theorem scale_word (word : BitVec 32) : lower word (.literal (some .fp32) .real scaleTag)=.bits word := by
  simp [lower]
theorem padding_zero (word : BitVec 32) : lower word (.literal (some .fp32) .real 0)=zero := by
  norm_num [lower,scaleTag,zero]

/-- The actual accumulator node becomes a complete ordered scalar FMA chain;
there is no variable-sized dot numerical interface. -/
theorem dot_chain (scale : BitVec 32) (p : Option ComputeDType) (o : DotOptions)
    (m k n : Nat) (a : TileIndex [m,k]→Expr) (b : TileIndex [k,n]→Expr)
    (carry : Expr) (i : TileIndex [m,n]) :
    lower scale (.dotAcc p o [] m k n a b carry i)=
      ieeeDot k (fun j=>lower scale (a (i.1,j,())))
        (fun j=>lower scale (b (j,i.2.1,()))) (lower scale carry) := rfl

def value (c : Shape) (s : State Expr) (scale : BitVec 32)
    (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) (bt : 0<c.BT) : Graph :=
  lower scale (FPValue.outputValue algebra c (dots c)
    (FPGrid.launch c s (owner c bt h t v)) scaleTag (lane c bt t v))

theorem complete (c : Shape) (bt : 0<c.BT) (s : State Expr) (scale : BitVec 32) :
    ∃ final,FPGrid.run algebra c scaleTag (FPGrid.programs c) s=some final ∧
      FPGrid.Result algebra c scaleTag (dots c) s final ∧
      (∀ h t v,lower scale ((final.mem "o" (h.val*(c.T*c.V)+t.val*c.V+v.val)).read .fp32)=
        value c s scale h t v bt) := by
  obtain ⟨final,source,values⟩:=FPGrid.whole_grid algebra c scaleTag (dots c) s
  refine ⟨final,source,values,?_⟩
  intro h t v
  rw [FPGrid.returned algebra c bt scaleTag (dots c) s final values h t v]
  simp only [Cell.read_same,value]

/-- Generic representation transport plus five fixed scalar arithmetic
families. Concrete native words use `WordPrimitives`, whose bit leaves are
structurally the identity. No dot, reduction or output numerical port exists. -/
structure Primitives (α : Type) where
  bits : BitVec 32 → α
  literal : Option ComputeDType → FloatDType → ℝ → α
  binary : Option ComputeDType → FloatDType → Binary → α → α → α
  unary : Option ComputeDType → Unary → α → α
  cast : Option ComputeDType → FloatDType → FloatDType → α → α
  fma : α → α → α → α

def Primitives.isa {α : Type} (p : Primitives α) : ISA α where
  bits:=p.bits
  literal:=p.literal
  binary:=p.binary
  unary:=p.unary
  cast:=p.cast
  fma:=p.fma
  mmaTF32:=fun _ _ _ _ carry=>carry

/-- Concrete FP32 numerical boundary. Raw loads and bit leaves preserve
their exact words; only these five arithmetic families are numerical ports. -/
structure WordPrimitives where
  literal : Option ComputeDType → FloatDType → ℝ → BitVec 32
  binary : Option ComputeDType → FloatDType → Binary → BitVec 32 → BitVec 32 → BitVec 32
  unary : Option ComputeDType → Unary → BitVec 32 → BitVec 32
  cast : Option ComputeDType → FloatDType → FloatDType → BitVec 32 → BitVec 32
  fma : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 32

def WordPrimitives.scalar (p : WordPrimitives) : Primitives (BitVec 32) where
  bits:=id
  literal:=p.literal
  binary:=p.binary
  unary:=p.unary
  cast:=p.cast
  fma:=p.fma

def WordPrimitives.isa (p : WordPrimitives) : ISA (BitVec 32) := p.scalar.isa

theorem word_leaf (p : WordPrimitives) (word : BitVec 32) : (Graph.bits word).eval p.isa=word := rfl

structure ScalarTransport {α β : Type} (f : α→β) (A : Primitives α) (B : Primitives β) : Prop where
  bits : ∀ x,f (A.bits x)=B.bits x
  literal : ∀ p d x,f (A.literal p d x)=B.literal p d x
  binary : ∀ p d op a b,f (A.binary p d op a b)=B.binary p d op (f a) (f b)
  unary : ∀ p op a,f (A.unary p op a)=B.unary p op (f a)
  cast : ∀ p s t a,f (A.cast p s t a)=B.cast p s t (f a)
  fma : ∀ a b c,f (A.fma a b c)=B.fma (f a) (f b) (f c)

/-- Only fixed local operations transport numbers through the derived graph. -/
theorem transport {α β : Type} (f : α→β) (A : Primitives α) (B : Primitives β)
    (ports : ScalarTransport f A B) (c : Shape) (bt : 0<c.BT) (s : State Expr)
    (scale : BitVec 32) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    f ((value c s scale h t v bt).eval A.isa)=(value c s scale h t v bt).eval B.isa := by
  apply VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions.transport
  exact ⟨ports.bits,ports.literal,ports.binary,ports.unary,ports.cast,ports.fma,fun _ _ _ _ _=>rfl⟩

#specNonCircular value avoiding [FPGrid.run,FP.Structural.exec]
#axiomsClean complete
#axiomsClean transport
end
end VeriTile.Bench.Repairs.ChunkGlaSimple.ScalarGraph
