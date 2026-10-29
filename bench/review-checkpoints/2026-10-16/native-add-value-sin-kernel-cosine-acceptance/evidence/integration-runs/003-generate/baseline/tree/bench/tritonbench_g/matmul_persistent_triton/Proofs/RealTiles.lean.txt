import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealMath
import VeriTile.Triton.Semantics.TileOps

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealTiles
open VeriTile.Triton
open scoped BigOperators
noncomputable section

/-- The source clamps inactive row/column input lanes to zero before loading.
Only the K mask supplies zero padding; no stronger source load mask is assumed. -/
def clamp (bound index : Nat) : Nat := if index < bound then index else 0

theorem clamp_bound (bound index : Nat) (h : 0 < bound) : clamp bound index < bound := by
  unfold clamp
  split <;> omega

theorem clamp_active (bound index : Nat) (h : index < bound) : clamp bound index = index := by
  simp [clamp, h]

def aValues (p : Parameters) (xs : RealMath.Inputs) (row : Fin p.BM → Nat)
    (ki : Nat) : Tile .real [p.BM, p.BK] :=
  Tile.mat fun i k => some (if k.val < p.K - ki * p.BK then
    xs.a (clamp p.M (row i)) (ki * p.BK + k.val) else 0)

def bValues (p : Parameters) (xs : RealMath.Inputs) (column : Fin p.BN → Nat)
    (ki : Nat) : Tile .real [p.BK, p.BN] :=
  Tile.mat fun k j => some (if k.val < p.K - ki * p.BK then
    xs.b (ki * p.BK + k.val) (clamp p.N (column j)) else 0)

theorem finite_dot (M K N : Nat) (a : Tile .real [M,K]) (b : Tile .real [K,N])
    (i : Fin M) (j : Fin N) (fa fb : Fin K → ℝ)
    (ha : ∀ k, a.data (i,k,PUnit.unit) = some (fa k))
    (hb : ∀ k, b.data (k,j,PUnit.unit) = some (fb k)) :
    (Tile.dot [] a b).data (i,j,PUnit.unit) = some (∑ k, fa k * fb k) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ
      (fun k => Option.map₂ (· * ·) (a.data (i,k,PUnit.unit)) (b.data (k,j,PUnit.unit)))) =
      @Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ (fun k => some (fa k * fb k))
    from Finset.sum_congr rfl (fun k _ => by rw [ha k, hb k]; rfl)]
  show (∑ k, ((fa k * fb k : ℝ) : WithBot ℝ)) = _
  rw [← WithBot.coe_sum]
  rfl

/-- Exact mathematical dot of the actual source's K-masked, clamped input
lanes. A partial final K tile contributes exactly its in-bounds products. -/
theorem masked_dot (p : Parameters) (xs : RealMath.Inputs)
    (row : Fin p.BM → Nat) (column : Fin p.BN → Nat) (ki : Nat) :
    Tile.dot [] (aValues p xs row ki) (bValues p xs column ki) =
      Tile.mat fun i j => some (RealMath.tileSum p.K p.BK ki xs
        (clamp p.M (row i)) (clamp p.N (column j))) := by
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  rw [finite_dot p.BM p.BK p.BN _ _ i j _ _ (by intro k; rfl) (by intro k; rfl)]
  apply congrArg some
  unfold RealMath.tileSum
  apply Finset.sum_congr rfl
  intro k _
  by_cases hk : k.val < p.K - ki * p.BK <;> simp [hk]

/-- For an active output lane the input clamp is inactive, so the completed
accumulator equals the independent original-row/original-column GEMM sum. -/
theorem completed_active (p : Parameters) (xs : RealMath.Inputs)
    (row column : Nat) (hBK : 0 < p.BK) (hr : row < p.M) (hc : column < p.N) :
    RealMath.partialSum p.K p.BK (RealMath.blocks p.K p.BK) xs
      (clamp p.M row) (clamp p.N column) = RealMath.formula p.K xs row column := by
  rw [RealMath.partial_complete _ _ hBK, clamp_active _ _ hr, clamp_active _ _ hc]

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealTiles
