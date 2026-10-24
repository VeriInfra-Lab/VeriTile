import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.FPBackend

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.NativeGraph
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Word := BitVec 32
inductive Tree (n : Nat) where
  | input (i : Fin n)
  | padding (word : BitVec 32)
  | node (a b : Tree n)

def Tree.indices : Tree n → List (Fin n)
  | .input i => [i] | .padding _ => [] | .node a b => a.indices++b.indices

structure OrderedTree (n : Nat) where
  tree : Tree n
  complete : tree.indices.Perm (List.finRange n)

def nativeDType : FloatDType → FloatDType
  | .real => .fp32 | d => d

def dotOptions (left right : FloatDType) (source : DotOptions) : DotOptions :=
  if nativeDType left = .fp32 ∧ nativeDType right = .fp32 ∧ source.inputPrecision = none then
    { source with inputPrecision := some "tf32" }
  else source

structure Operations where
  literal : Option ComputeDType → FloatDType → ℝ → Word
  fromNat : Option ComputeDType → Nat → Word
  fromInt : Option ComputeDType → Int → Word
  cast : Option ComputeDType → FloatDType → FloatDType → Word → Word
  unary : Option ComputeDType → Unary → Word → Word
  binary : Option ComputeDType → FloatDType → Binary → Word → Word → Word
  fma : Option ComputeDType → FloatDType → Word → Word → Word → Word
  dotAcc : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    (batch : TileShape) → (m k n : Nat) →
    (TileIndex (batch ++ [m,k]) → Word) → (TileIndex (batch ++ [k,n]) → Word) →
    Word → TileIndex (batch ++ [m,n]) → Word


end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.NativeGraph
