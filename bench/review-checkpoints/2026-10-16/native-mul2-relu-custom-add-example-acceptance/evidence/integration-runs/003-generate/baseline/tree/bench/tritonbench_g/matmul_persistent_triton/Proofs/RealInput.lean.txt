import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealTiles
import VeriTile.Triton.Semantics.State

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealInput
open VeriTile.Triton
noncomputable section

/-- Independent input arrays inhabit their original finite matrix domains.
The relation constrains typed source cells, not prepared dot intermediates. -/
structure Represents (p : Parameters) (xs : RealMath.Inputs) (s : BlockState) : Prop where
  a : ∀ row k, row < p.M → k < p.K →
    s.mem p.A (row*p.sam+k*p.sak) = MemCell.real (xs.a row k)
  b : ∀ k column, k < p.K → column < p.N →
    s.mem p.B (k*p.sbk+column*p.sbn) = MemCell.real (xs.b k column)

def arrays (p : Parameters) (s : BlockState) : RealMath.Inputs where
  a row k := s.readMem p.A (row*p.sam+k*p.sak)
  b k column := s.readMem p.B (k*p.sbk+column*p.sbn)

theorem a_read (p : Parameters) (xs : RealMath.Inputs) (s : BlockState)
    (h : Represents p xs s) (row k : Nat) (hr : row < p.M) (hk : k < p.K) :
    (arrays p s).a row k = xs.a row k := by
  simp [arrays, BlockState.readMem, h.a row k hr hk]

theorem b_read (p : Parameters) (xs : RealMath.Inputs) (s : BlockState)
    (h : Represents p xs s) (k column : Nat) (hk : k < p.K) (hc : column < p.N) :
    (arrays p s).b k column = xs.b k column := by
  simp [arrays, BlockState.readMem, h.b k column hk hc]

/-- K-masked A loads need only the in-bounds independent input domain;
clamped inactive output rows still read original row zero. -/
theorem a_values (p : Parameters) (xs : RealMath.Inputs) (s : BlockState)
    (h : Represents p xs s) (hM : 0 < p.M) (row : Fin p.BM → Nat) (ki : Nat) :
    RealTiles.aValues p (arrays p s) row ki = RealTiles.aValues p xs row ki := by
  apply Tile.ext
  rintro ⟨i,k,⟨⟩⟩
  unfold RealTiles.aValues Tile.mat
  dsimp only
  by_cases hk : k.val < p.K-ki*p.BK
  · simp only [hk, if_true]
    rw [a_read p xs s h _ _ (RealTiles.clamp_bound _ _ hM) (by omega)]
  · simp only [hk, if_false]

theorem b_values (p : Parameters) (xs : RealMath.Inputs) (s : BlockState)
    (h : Represents p xs s) (hN : 0 < p.N) (column : Fin p.BN → Nat) (ki : Nat) :
    RealTiles.bValues p (arrays p s) column ki = RealTiles.bValues p xs column ki := by
  apply Tile.ext
  rintro ⟨k,j,⟨⟩⟩
  unfold RealTiles.bValues Tile.mat
  dsimp only
  by_cases hk : k.val < p.K-ki*p.BK
  · simp only [hk, if_true]
    rw [b_read p xs s h _ _ (by omega) (RealTiles.clamp_bound _ _ hN)]
  · simp only [hk, if_false]

/-- Any source state with the same two input regions preserves the original
input relation; all output stores will instantiate this complete raw-cell fact. -/
theorem preserved (p : Parameters) (xs : RealMath.Inputs) (s t : BlockState)
    (h : Represents p xs s)
    (ha : ∀ o, t.mem p.A o = s.mem p.A o)
    (hb : ∀ o, t.mem p.B o = s.mem p.B o) : Represents p xs t := by
  exact ⟨fun row k hr hk => (ha _).trans (h.a row k hr hk),
    fun k column hk hc => (hb _).trans (h.b k column hk hc)⟩

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealInput
