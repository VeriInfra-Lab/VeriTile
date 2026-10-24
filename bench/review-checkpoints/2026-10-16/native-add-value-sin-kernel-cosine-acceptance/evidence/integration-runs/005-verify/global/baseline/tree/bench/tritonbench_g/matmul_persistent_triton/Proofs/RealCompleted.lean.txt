import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealLoop
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealOutputAddresses

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealCompleted
open VeriTile.Triton RealLoopState RealLoop
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 14000

/-- Every previously completed source tile contains its independent input-only
matrix product, read from the actual tagged output memory. -/
def Holds (p : Parameters) (xs : RealMath.Inputs) (pid completed : Nat) (s : BlockState) : Prop :=
  ∀ q, q<completed → ∀ j, RealStore.active p (tileRow p pid q) (tileColumn p pid q) j →
    s.mem p.C (RealStore.offset p (tileRow p pid q) (tileColumn p pid q) j) =
      MemCell.real (RealMath.formula p.K xs (RealStore.rows p (tileRow p pid q) j.1)
        (RealStore.columns p (tileColumn p pid q) j.2.1))

theorem earlier_frame (p : Parameters) (s : BlockState) (xs : RealMath.Inputs) (pid i q : Nat)
    (hk : 0<count p) (hBM : 0<p.BM) (hBN : 0<p.BN) (hGM : 0<p.GM)
    (hs : 0<p.SMS) (hp : pid<p.SMS) (hi : i<stop p pid) (hq : q < i/count p)
    (hinj : RealOutputAddresses.Injective p)
    (j : TileIndex [p.BM,p.BN]) (hj : RealStore.active p (tileRow p pid q) (tileColumn p pid q) j) :
    (RealIteration.result p s xs pid i).mem p.C
        (RealStore.offset p (tileRow p pid q) (tileColumn p pid q) j) =
      s.mem p.C (RealStore.offset p (tileRow p pid q) (tileColumn p pid q) j) := by
  have hc := (Schedule.iteration_bounds (count p) (work p pid) i hk hi).1
  have hqt : Schedule.tileId p.SMS pid q < RealGroup.rows p*RealGroup.columns p :=
    (Schedule.ordinal_bound_iff _ _ _ _ hs hp).mp (lt_trans hq hc)
  have hne : tile p pid i ≠ Schedule.tileId p.SMS pid q := by
    intro he
    have hh := (Schedule.tile_unique p.SMS pid (i/count p) pid q hs hp hp he).2
    omega
  apply RealIteration.frame p s xs pid i
  intro k hactive he
  exact RealOutputAddresses.tiles_disjoint p hinj hBM hBN hGM
    (tile p pid i) (Schedule.tileId p.SMS pid q)
    (tile_bound p pid i hk hs hp hi) hqt hne k j hactive hj he.2

/-- The last-K branch adds exactly one completed output tile; all earlier
outputs survive both branches because grouped tiles have disjoint addresses. -/
theorem preserves (p : Parameters) (s : BlockState) (xs : RealMath.Inputs) (pid i : Nat)
    (hk : 0<count p) (hBK : 0<p.BK) (hBM : 0<p.BM) (hBN : 0<p.BN) (hGM : 0<p.GM)
    (hs : 0<p.SMS) (hp : pid<p.SMS) (hi : i<stop p pid)
    (hinj : RealOutputAddresses.Injective p) (h : Holds p xs pid (i/count p) s) :
    Holds p xs pid ((i+1)/count p) (RealIteration.result p s xs pid i) := by
  intro q hq j hj
  by_cases hearlier : q < i/count p
  · rw [earlier_frame p s xs pid i q hk hBM hBN hGM hs hp hi hearlier hinj j hj]
    exact h q hearlier j hj
  · have hlast : phase p i+1=count p := by
      by_contra hn
      rw [FlatSchedule.quotient_continue (count p) i hk hn] at hq
      exact hearlier hq
    have hqeq : q=i/count p := by
      rw [FlatSchedule.quotient_last (count p) i hk hlast] at hq
      omega
    subst q
    exact RealStep.complete_readback p (RealIteration.scheduled p s pid i) xs
      (row p pid i) (column p pid i) (phase p i) hBK hlast j hj
      (RealOutputAddresses.lanes_injective p hinj _ _)

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealCompleted
