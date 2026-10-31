import bench.tritonbench_g.mixed_sparse_attention.Proofs.Kernels
import bench.tritonbench_g.parallel_attention.Sources.Mathematical_FPRange
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
structure Config where
  storage : TypedSource.Storage
  q : RegionName
  k : RegionName
  v : RegionName
  out : RegionName
  seq : Region .nat
  blocks : Region .nat
  offsets : Region .nat
  counts : Region .nat
  columns : Region .nat
  sqz : Nat
  sqh : Nat
  sqm : Nat
  sqk : Nat
  skz : Nat
  skh : Nat
  skn : Nat
  skk : Nat
  svz : Nat
  svh : Nat
  svn : Nat
  svk : Nat
  soz : Nat
  soh : Nat
  som : Nat
  sok : Nat
  Z : Nat
  H : Nat
  T : Nat
  NR : Nat
  NS : Nat
  NV : Nat
  BM : Nat
  BN : Nat
  BD : Nat
  scale : ℝ

/-- Typed dot availability only; its numerical result remains the exact
primitive call on independently specified fp16 operands. -/
structure Dot (α : Type) where
  equal : α → α → Bool
  apply : Option ComputeDType → DotOptions → {batch : TileShape} → {m k n : Nat} →
    (TileIndex (batch ++ [m,k]) → α) → (TileIndex (batch ++ [k,n]) → α) →
    (TileIndex (batch ++ [m,n]) → α)

def backend {α : Type} (M : Algebra α) (D : Dot α) : Algebra α :=
  { M with
    dotTyped := fun p l r o opts {_} {_} {_} {_} a b =>
      match l,r,o with
      | .fp16,.fp16,.fp32 => some (D.apply p opts a b)
      | .bf16,.bf16,.fp32 => some (D.apply p opts a b)
      | l,r,o => M.dotTyped p l r o opts a b
    compareEq := fun _ _ => some D.equal }

/-- The actual parsed floating maximum, retaining dtype and precision.
No comparison/select replacement or finiteness premise is needed. -/
def maximumAssignment (BM BN : Nat) : ComputeStmt :=
  .assign .real [BM] "m_i_new" (.compute (.alg .fp32
    (.maximumFloat .real (.consSame .nil) (.ref .real [BM] "m_i")
      (.reduceMax (shape := [BM,BN]) ⟨1,by simp⟩ Bool.false (.ref .real [BM,BN] "qk")))))

end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Source
