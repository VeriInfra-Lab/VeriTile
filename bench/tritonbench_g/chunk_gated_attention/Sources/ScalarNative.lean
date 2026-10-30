import bench.tritonbench_g.chunk_gated_attention.Sources.NativeGrid

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.ScalarNative
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions
noncomputable section

/-- No runtime scalar tagging is needed by these two kernels. Every source
literal, cast, live dot carry and exp operation is lowered structurally. -/
def lower : NativeGraph.Expr → Graph
  | .leaf word=>.bits word
  | .literal p d x=>.literal p d x
  | .load x=>.move .register (lower x)
  | .cast p a b x=>.cast p a b (lower x)
  | .unary p .exp x=>.unary p .exp2 (.binary p .real .mul (lower x) (.bits 0x3FB8AA3B))
  | .unary p op x=>.unary p op (lower x)
  | .binary p d op a b=>.binary p d op (lower a) (lower b)
  | .pendingDot _ _ batch _ _ _ a b i=>
      ScalarGraph.scalarDot zero batch (fun j=>lower (a j)) (fun j=>lower (b j)) i
  | .dotAcc _ _ batch _ _ _ a b carry i=>
      ScalarGraph.scalarDot (lower carry) batch (fun j=>lower (a j)) (fun j=>lower (b j)) i

theorem dot_chain (p : Option ComputeDType) (o : DotOptions) (m k n : Nat)
    (a : TileIndex [m,k]→NativeGraph.Expr) (b : TileIndex [k,n]→NativeGraph.Expr)
    (carry : NativeGraph.Expr) (i : TileIndex [m,n]) :
    lower (.dotAcc p o [] m k n a b carry i)=
      ieeeDot k (fun j=>lower (a (i.1,j,()))) (fun j=>lower (b (j,i.2.1,()))) (lower carry) := rfl

/-- Five arithmetic families; raw FP32 words are fixed to identity. -/
abbrev Primitives := ScalarGraph.WordPrimitives
def eval (I : Primitives) (value : NativeGraph.Expr) : BitVec 32 := (lower value).eval I.isa
theorem load_word (I : Primitives) (word : BitVec 32) : eval I (.leaf word)=word := rfl

theorem transport {α β : Type} (f : α→β) (A : ScalarGraph.Primitives α) (B : ScalarGraph.Primitives β)
    (ports : ScalarGraph.ScalarTransport f A B) (value : NativeGraph.Expr) :
    f ((lower value).eval A.isa)=(lower value).eval B.isa := by
  apply VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions.transport
  exact ⟨ports.bits,ports.literal,ports.binary,ports.unary,ports.cast,ports.fma,fun _ _ _ _ _=>rfl⟩

#axiomsClean dot_chain
#axiomsClean transport
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.ScalarNative
