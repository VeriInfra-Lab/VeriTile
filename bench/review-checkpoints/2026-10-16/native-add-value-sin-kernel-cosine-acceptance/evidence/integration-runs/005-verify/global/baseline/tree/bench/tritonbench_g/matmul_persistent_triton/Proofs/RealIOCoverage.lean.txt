import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealIO
import bench.tritonbench_g.matmul_persistent_triton.Proofs.ScheduleCoverage

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealIOCoverage
open VeriTile.Triton RealIO
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 14000

/-- Every logical matrix element belongs to an active output lane of a PID
launched by the actual wrapper, including partial M/N tiles and final groups. -/
theorem covered (p : Parameters) (hBM : 0 < p.BM) (hBN : 0 < p.BN)
    (hGM : 0 < p.GM) (hs : 0 < p.SMS) (r c : Nat) (hr : r < p.M) (hc : c < p.N) :
    ∃ (pid q : Nat) (j : TileIndex [p.BM,p.BN]), pid < grid p ∧ outputMask p pid q j ∧
      outputWindow p pid q j = p.scm*r+p.scn*c ∧
      ∀ xs, expected p xs pid q j = RealMath.formula p.K xs r c := by
  have hm : r/p.BM < RealGroup.rows p := by
    apply (Nat.div_lt_iff_lt_mul hBM).mpr
    exact lt_of_lt_of_le hr (RealMath.blocks_cover p.M p.BM hBM)
  have hn : c/p.BN < RealGroup.columns p := by
    apply (Nat.div_lt_iff_lt_mul hBN).mpr
    exact lt_of_lt_of_le hc (RealMath.blocks_cover p.N p.BN hBN)
  obtain ⟨tile,ht,hrow,hcol⟩ := Schedule.grouped_covered _ _ _ _ _ hGM hm hn
  obtain ⟨hpid,hq,htile⟩ := Schedule.tile_covered _ p.SMS tile hs ht
  let i : Fin p.BM := ⟨r%p.BM,Nat.mod_lt r hBM⟩
  let j : Fin p.BN := ⟨c%p.BN,Nat.mod_lt c hBN⟩
  have hi : RealStore.rows p (RealLoop.tileRow p (tile%p.SMS) (tile/p.SMS)) i = r := by
    simp only [RealStore.rows,RealLoop.tileRow,htile,hrow,i]
    simpa only [Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div r p.BM
  have hj : RealStore.columns p (RealLoop.tileColumn p (tile%p.SMS) (tile/p.SMS)) j = c := by
    simp only [RealStore.columns,RealLoop.tileColumn,htile,hcol,j]
    simpa only [Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div c p.BN
  refine ⟨tile%p.SMS,tile/p.SMS,(i,j,PUnit.unit),hpid,⟨hq,?_,?_⟩,?_,?_⟩
  · simpa only [hi] using hr
  · simpa only [hj] using hc
  · simp only [outputWindow,RealStore.offset,hi,hj]
  · intro xs
    simp only [expected,hi,hj]

/-- Active output windows of distinct legal host-grid PIDs are disjoint under
logical output-stride injectivity. This is a source-schedule fact, not a
concurrency or GPU execution claim. -/
theorem pids_disjoint (p : Parameters) (h : Admissible p)
    (pid pid' q q' : Nat) (hp : pid < grid p) (hp' : pid' < grid p) (hne : pid ≠ pid')
    (i j : TileIndex [p.BM,p.BN]) (hi : outputMask p pid q i) (hj : outputMask p pid' q' j) :
    outputWindow p pid q i ≠ outputWindow p pid' q' j := by
  have hb : pid < p.SMS := lt_of_lt_of_le hp (Nat.min_le_left _ _)
  have hb' : pid' < p.SMS := lt_of_lt_of_le hp' (Nat.min_le_left _ _)
  have ht := (Schedule.ordinal_bound_iff _ _ _ _ h.sms hb).mp hi.1
  have ht' := (Schedule.ordinal_bound_iff _ _ _ _ h.sms hb').mp hj.1
  apply RealOutputAddresses.tiles_disjoint p h.output_injective h.bm h.bn h.gm
    (Schedule.tileId p.SMS pid q) (Schedule.tileId p.SMS pid' q') ht ht' _ i j hi.2 hj.2
  intro he
  exact hne (Schedule.tile_unique p.SMS pid q pid' q' h.sms hb hb' he).1

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealIOCoverage
