import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.FPEpilogue

namespace VeriTile.Bench.Repairs.BmmChunkFwd.FPSuccess
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Successful complete original-order typed program under arbitrary local
floating callbacks. Causal skipped tiles preserve their entire memory. -/
theorem complete {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config) (s : State α) :
    ∃ t, FP.Structural.exec (engine o.algebra) (Kernels.source h c) s=some t ∧ (∀ rg off, ¬ProgramWrites h c s rg off → t.mem rg off=s.mem rg off) ∧
      (active h c s=Bool.false → t.mem=s.mem) := by
  obtain ⟨initial,hpre,hbase,hmem,hactive,hcontrols⟩ := prologue o h c s
  cases ha : active h c s
  · refine ⟨initial,?_,?_,fun _ => hmem⟩
    · change run _ (Kernels.source h c).surfaceBody s=some initial
      rw [source_split,run_append,hpre]
      simp only [Option.bind_some,run,step,hactive,ha,Bool.false_eq_true,↓reduceIte]
      rfl
    · intro rg off _
      rw [hmem]
  · obtain ⟨prepared,hsetup,hready,hm0,hf0⟩ := setup o h c initial hbase
    obtain ⟨mid,hloop,hmready,hm1,hf1⟩ := loop_success o h c prepared hready
    obtain ⟨final,hpost,hframe⟩ := epilogue o h c s mid hmready ((hcontrols.fixed hf0).fixed hf1)
    refine ⟨final,?_,?_,by simp [ha]⟩
    · change run _ (Kernels.source h c).surfaceBody s=some final
      rw [source_split,run_append,hpre]
      simp only [Option.bind_some,run,step,hactive,ha,↓reduceIte]
      rw [branch_split,run_append,run_append,hsetup]
      simp only [Option.bind_some,run,hloop]
      simp [Option.bind,hpost]
    · intro rg off hrg
      rw [hframe rg off (fun hw => hrg ⟨ha,hw⟩),hm1,hm0,hmem]

#axiomsClean complete
end
end VeriTile.Bench.Repairs.BmmChunkFwd.FPSuccess

#guard_msgs (drop info) in
#auditModuleAxioms
