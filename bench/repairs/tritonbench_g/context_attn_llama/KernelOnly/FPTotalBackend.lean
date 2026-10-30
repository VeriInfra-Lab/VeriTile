import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.FPSuccess

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.FPSuccess
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

/-- Ordinary implementations of the two reached optional primitives. Values
are arbitrary. This total interface contains no successful-kernel, expected
output, mathematical exactness, finiteness or memory hypothesis. -/
structure TotalDots (α : Type) where
  dot : Option ComputeDType → {batch : TileShape} → {m k n : Nat} →
    Values α .real (batch ++ [m,k]) → Values α .real (batch ++ [k,n]) →
    Values α .real (batch ++ [m,n])
  dotAcc : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    Values α left.toTileDType (batch ++ [m,k]) →
    Values α right.toTileDType (batch ++ [k,n]) →
    Values α out.toTileDType (batch ++ [m,n]) →
    Values α out.toTileDType (batch ++ [m,n])

def TotalDots.install (D : TotalDots α) (M : Algebra α) : Algebra α :=
  { M with
    dot := fun precision {_} {_} {_} {_} q k => some (D.dot precision q k)
    dotAcc := fun precision left right out options {_} {_} {_} {_} p v acc =>
      some (D.dotAcc precision left right out options p v acc)}

def TotalDots.support (D : TotalDots α) (M : Algebra α) (p : Layout) :
    DotSupport p (D.install M) where
  qk := D.dot (some .fp32) (batch := []) (m := p.BM) (k := p.D) (n := p.BN)
  pv := D.dotAcc (some .fp32) .real .real .real {} (batch := []) (m := p.BM) (k := p.BN) (n := p.D)
  qk_bound := fun _ _ => rfl
  pv_bound := fun _ _ _ => rfl

end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.FPSuccess
