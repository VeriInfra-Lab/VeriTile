import bench.tritonbench_g.softmax_optimize.IO
import bench.tritonbench_g.softmax_optimize.Proofs.IOSafety
import bench.tritonbench_g.softmax_optimize.Proofs.SourceIO

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.IOCorrect
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open SoftmaxIO
noncomputable section
set_option maxHeartbeats 1000000

/-- Compose the actual four-loop trace proof and complete existing row result
with the public N-element IO, without restricting the source program ID. -/
theorem correct (Y X : RegionName) (M N : Nat) (separate : Y ≠ X)
    (positive : 0 < N) :
    (io Y X M N).RegionalImplements (target Y X M N) := by
  unfold UKernelIO.RegionalImplements
  apply UKernelIO.RegionalImplements3D.intro
  · intro bounds s vals _hi hib hob _hsb
    apply IOSafety.source_safe bounds Y X M N (Config.tile N) positive
      (Config.tile_positive N) s
    · intro i hi
      exact hib (0 : Fin 1) ⟨i,hi⟩ trivial
    · intro i hi
      exact hob (0 : Fin 1) ⟨i,hi⟩ trivial
  · intro s vals hpins
    obtain ⟨final, executed, outputs, frame, _⟩ :=
      SourceIO.selected_tile_io Y X M N separate positive s
    refine ⟨final, executed, ?_, ?_⟩
    · intro o j _
      change final.readMem Y (s.pid*N+j.val) = rowSoftmax (vals (0 : Fin 1)) j
      rw [outputs j,softmaxOptimizeFullSpec_eq_div]
      unfold rowSoftmax
      rw [← Fin.sum_univ_eq_sum_range]
      have pinned (k : Fin N) : s.readMem X (s.pid*N+k.val) = vals (0 : Fin 1) k :=
        hpins (0 : Fin 1) k trivial
      simp only [pinned]
      congr 1
    · intro r address outside _
      apply frame r address
      by_cases hr : r = Y
      · subst r
        right
        by_cases before : address < s.pid*N
        · exact Or.inl before
        · right
          by_contra after
          have hj : address-s.pid*N < N := by omega
          have excluded := outside (0 : Fin 1) ⟨address-s.pid*N,hj⟩ trivial
          change Y ≠ Y ∨ address ≠ s.pid*N+(address-s.pid*N) at excluded
          rcases excluded with absurd | unequal
          · exact absurd rfl
          · exact unequal (by omega)
      · exact Or.inl hr

#axiomsClean correct
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.IOCorrect
