import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.FPSymbolic

namespace VeriTile.Bench.Repairs.ContextAttnNopad.FPSymbolic
open VeriTile.Triton FP.Structural
open FP.ProfiledRewrite (engine)
open VeriTile.Bench.TritonBenchG.ContextAttnNopad
open Formula
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Construct all iterations of the signed source range. The induction is on
its actual decreasing distance; no execution/final-state condition is assumed. -/
theorem range_complete {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    (p : Layout) (r : Program.Regions) (scale : ℝ) (hBN : 0 < p.BN)
    (cur stop : Int) (s : State α) (ready : Ready p s) :
    ∃ t, rangeInt (engine o.algebra) "start_n" cur stop p.BN (parts p r scale).body s = some t ∧
      Ready p t ∧ t.mem = s.mem ∧ Controls p s t := by
  by_cases hc : cur < stop
  · obtain ⟨mid,he,hr,hm,hc1⟩ := body o p r scale s hBN ready cur
    obtain ⟨t,ht,hrt,hmt,hc2⟩ := range_complete o p r scale hBN (cur+p.BN) stop mid hr
    refine ⟨t,?_,hrt,hmt.trans hm,hc1.trans hc2⟩
    rw [rangeInt,if_neg (Nat.ne_of_gt hBN),if_pos hc,he]
    exact ht
  · refine ⟨s,?_,ready,rfl,Controls.refl p s⟩
    rw [rangeInt,if_neg (Nat.ne_of_gt hBN),if_neg hc]
termination_by (stop-cur).toNat
decreasing_by omega

theorem loop_complete {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    (p : Layout) (r : Program.Regions) (scale : ℝ) (hBN : 0 < p.BN)
    (s : State α) (ready : Ready p s) :
    ∃ t, step (engine o.algebra)
      (.forRangeInt "start_n" (parts p r scale).start (parts p r scale).stop
        (parts p r scale).stride (parts p r scale).body) s = some t ∧
      Ready p t ∧ t.mem = s.mem ∧ Controls p s t := by
  have hb := ready.2.1
  have hmask := ready.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨block,hb⟩ := Option.isSome_iff_exists.mp hb
  obtain ⟨mask,hmask⟩ := Option.isSome_iff_exists.mp hmask
  have he := range_complete o p r scale hBN 0
    ((mask PUnit.unit * (block PUnit.unit+1) * p.BM : Nat) : Int) s ready
  simpa (config := {maxSteps := 300000}) [parts,Program.parts,Program.mathematical,Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnNopad.original,Source.intRange?,ComputeKernel.surfaceBody,
    step,evalOp_unfold,Option.bind,numeric,bop,State.setReg,hb,hmask] using he

/-- The actual source store is reached successfully after the opaque loop. -/
theorem epilogue {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    (p : Layout) (r : Program.Regions) (scale : ℝ) (s : State α) (ready : Ready p s) :
    ∃ t, run (engine o.algebra) (parts p r scale).after s = some t ∧
      ∀ rg off, rg ≠ r.out → t.mem rg off = s.mem rg off := by
  rcases ready with ⟨hh,hb,hs,hl,hr,hk,hd,hq,hkp,hvp,hm,hn,ha,hmask⟩
  obtain ⟨head,hh⟩ := Option.isSome_iff_exists.mp hh
  obtain ⟨start,hs⟩ := Option.isSome_iff_exists.mp hs
  obtain ⟨length,hl⟩ := Option.isSome_iff_exists.mp hl
  obtain ⟨rows,hr⟩ := Option.isSome_iff_exists.mp hr
  obtain ⟨dims,hd⟩ := Option.isSome_iff_exists.mp hd
  obtain ⟨accumulator,ha⟩ := Option.isSome_iff_exists.mp ha
  simp (config := {maxSteps := 300000}) [parts,program,Program.parts,Program.mathematical,Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnNopad.original,Source.intRange?,ComputeKernel.surfaceBody,
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,store,
    engine,Algebra.withDefaultPrecision,resolvePrecision,FPBackend.Ops.algebra,
    ComputeDType.eraseDType,numeric,natLt,natLe,numericLt,numericLe,numericEq,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,integralFloorDiv,integralMod,
    hh,hs,hl,hr,hd,ha]
  intro rg off hrg
  apply State.masked_scatter_frame
  exact Or.inl hrg

/-- Full successful typed FP execution, constructed from the actual prelude,
signed loop and epilogue, for arbitrary opaque FP values and dot callbacks. -/
theorem complete {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    (p : Layout) (r : Program.Regions) (scale : ℝ) (hBN : 0 < p.BN) (s : State α) :
    ∃ t, FP.Structural.exec (engine o.algebra) (program p r scale) s = some t ∧
      ∀ rg off, rg ≠ r.out → t.mem rg off = s.mem rg off := by
  obtain ⟨pre,hpre,ready,hpm,hgeo⟩ := prelude o p r scale s
  obtain ⟨post,hloop,ready',hlm,hcontrols⟩ := loop_complete o p r scale hBN pre ready
  obtain ⟨final,hfinal,hfm⟩ := epilogue o p r scale post ready'
  refine ⟨final,?_,?_⟩
  · change run _ (program p r scale).surfaceBody s = some final
    rw [source_split,List.append_assoc,run_append,hpre]
    simpa only [Option.bind_some,List.singleton_append,run,hloop] using hfinal
  · intro rg off hrg
    rw [hfm rg off hrg,hlm,hpm]

end
end VeriTile.Bench.Repairs.ContextAttnNopad.FPSymbolic
