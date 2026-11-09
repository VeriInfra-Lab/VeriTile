import bench.tritonbench_g.layer_norm_welfold.Proofs.ReadableAccumulate
import bench.tritonbench_g.layer_norm_welfold.Proofs.Statistics
namespace VeriTile.Bench.Repairs.LayerNormWelfold.Readable
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem mean_run (M : Algebra α) (s t : State α)
    (Mean : RegionName) (S D B : Nat) (rt : ReductionTree B)
    (mean total : α) (h : Summed Bool.false s t B mean total) :
    ∃ u,run M (meanTail Mean S D B rt) t=some u ∧
      MeanStored s u Mean B (M.binary (some .fp32) .real .div total (M.fromNat (some .fp32) D)) :=
  LayerNormWelfold.mean_run M s t Mean D B mean total h

theorem inverse_run (M : Algebra α) (s t : State α)
    (Rstd : RegionName) (S D B : Nat) (rt : ReductionTree B)
    (mean total : α) (h : Summed Bool.true s t B mean total) :
    ∃ u,run M (inverseTail Rstd S D B rt) t=some u ∧
      InverseStored s u Rstd B mean (inverseFromTotal M D total) :=
  LayerNormWelfold.inverse_run M s t Rstd D B mean total h

theorem stats_run (M : Algebra α) (s : State α)
    (X Mean Rstd : RegionName) (S D B : Nat) (rt : ReductionTree B) (hX : X≠Mean) :
    ∃ t,run M (stats X Mean Rstd S D B rt) s=some t ∧
      StatisticsReady s t Mean Rstd B (meanValue M s X D B rt) (inverseValue M s X D B rt) := by
  obtain ⟨a,ha,hia⟩ := init_run M s S B
  obtain ⟨b,hb,hib⟩ := phase_loop M Bool.false s a X S D B 0 (blockCount D B) rt _ _ hia (by omega)
  obtain ⟨c,hc,hic⟩ := sum_run M s b _ _ hib rt
  obtain ⟨d,hd,hid⟩ := mean_run M s c Mean S D B rt _ _ hic
  obtain ⟨e,he,hie⟩ := reset_mean_run M s d Mean B _ hid
  obtain ⟨f,hf,hif⟩ := phase_loop M Bool.true _ e X S D B 0 (blockCount D B) rt _ _ hie (by omega)
  obtain ⟨g,hg,hig⟩ := sum_run M _ f _ _ hif rt
  obtain ⟨t,ht,hit⟩ := inverse_run M _ g Rstd S D B rt _ _ hig
  have hxmem : (s.write Mean (s.pids 0) (.mk .real (meanValue M s X D B rt))).mem X=s.mem X := by
    funext off
    exact State.write_other _ _ _ _ _ _ (.inl hX)
  have hv := recurrence_congr M Bool.true
    (s.write Mean (s.pids 0) (.mk .real (meanValue M s X D B rt))) s X D B
    0 (blockCount D B) (meanValue M s X D B rt) (fun _ => zero M) hxmem rfl
  simp only [Nat.zero_mul] at hb hf
  have hbr : run M [.forRange "roffset" 0 D B (phaseBody Bool.false X S D B rt)] a=some b := by
    simp [run,step,hb]
  have hfr : run M [.forRange "roffset" 0 D B (phaseBody Bool.true X S D B rt)] e=some f := by
    simp [run,step,hf]
  refine ⟨t,?_,?_⟩
  · rw [stats_split]
    simp only [run_append,ha,hbr,hc,hd,he,hfr,hg,ht,Option.bind]
  · change InverseStored (s.write Mean (s.pids 0) (.mk .real (meanValue M s X D B rt))) t Rstd B
      (meanValue M s X D B rt)
      (inverseFromTotal M D (treeValue M
        (recurrence M Bool.true (s.write Mean (s.pids 0) (.mk .real (meanValue M s X D B rt)))
          X D B (meanValue M s X D B rt) 0 (blockCount D B) (fun _ => zero M)) rt)) at hit
    rw [hv] at hit
    exact ⟨hit.mem,hit.pids,hit.undef,hit.row,hit.rbase,hit.mean,hit.inverse⟩
#axiomsClean stats_run
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Readable
