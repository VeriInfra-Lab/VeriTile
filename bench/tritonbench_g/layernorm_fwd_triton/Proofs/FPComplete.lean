import bench.tritonbench_g.layernorm_fwd_triton.Proofs.FPOutput
namespace VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 22000

theorem output_range_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (start stop : Nat) (mem : RegionName → Nat → Cell α)
    (hx : p.x ≠ p.y) (hw : p.weight ≠ p.y) (h : OutputReady M p s t mem) :
    ∃ u, FP.Structural.range M "off" start stop p.block (loopBody p 12) t = some u ∧
      OutputReady M p s u (foldRange (outputStep M p s) start stop p.block mem) := by
  apply range_values M "off" (loopBody p 12) (outputStep M p s)
    (fun _ mem t => OutputReady M p s t mem) (fun mem t => OutputReady M p s t mem)
    start stop p.block mem t
  · intro n a u hu _
    exact output_body_run M p s u n a hx hw hu
  · intro _ _ _ hu
    exact hu
  · exact h

theorem source_split (p : Parameters) : (kernel p).surfaceBody = prelude p ++
    [ComputeStmt.forRange "off" 0 p.columns p.block (loopBody p 6)] ++ meanPost p ++
    [ComputeStmt.forRange "off" 0 p.columns p.block (loopBody p 9)] ++ varPost p ++
    [ComputeStmt.forRange "off" 0 p.columns p.block (loopBody p 12)] := by rfl

theorem complete_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (hx : p.x ≠ p.y) (hw : p.weight ≠ p.y) :
    ∃ t, FP.Structural.exec M (kernel p) s = some t ∧ t.mem = referenceMem M p s := by
  obtain ⟨a,ha,hr⟩ := prelude_run M p s
  obtain ⟨b,hb,hs⟩ := mean_range_run M p s a 0 p.columns (initial M p) hr
  obtain ⟨c,hc,ht⟩ := mean_post_run M p s b hs
  obtain ⟨d,hd,hu⟩ := var_range_run M p s c 0 p.columns (initial M p) ht
  obtain ⟨e,he,hv⟩ := var_post_run M p s d hu
  obtain ⟨f,hf,hw'⟩ := output_range_run M p s e 0 p.columns s.mem hx hw hv
  refine ⟨f,?_,hw'.memory⟩
  change run M (kernel p).surfaceBody s = some f
  rw [source_split]
  simp only [run_append, ha, run, List.foldlM_cons, List.foldlM_nil, bind, pure, Option.bind,
    step, hb, hc, hd, he, hf]

#axiomsClean output_range_run
#axiomsClean source_split
#axiomsClean complete_run
end
end VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPStages
