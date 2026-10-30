import bench.tritonbench_g.context_attn_mistral.Proofs.FPSymbolic

namespace VeriTile.Bench.Repairs.ContextAttnMistral.FPSymbolic
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open Formula
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- The signed source loop terminates for every reached bound and preserves
its register invariant. Successful loop bodies are constructed by `body`. -/
theorem loop {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    (p : Layout) (r : Program.Regions) (storage : Storage) (scale : ℝ)
    (hBN : 0 < p.BN) (cur stop : Int) (s : State α) (ready : Ready p storage s) :
    ∃ t, rangeInt (engine o.algebra) "start_n" cur stop p.BN
        (parts p r storage scale).body s = some t ∧ Ready p storage t ∧ t.mem = s.mem := by
  have hn : p.BN ≠ 0 := by omega
  by_cases hlt : cur < stop
  · obtain ⟨mid,hm,hr,hmem⟩ := body o p r storage scale s hBN ready cur
    obtain ⟨t,ht,htr,htm⟩ := loop o p r storage scale hBN (cur+p.BN) stop mid hr
    refine ⟨t,?_,htr,htm.trans hmem⟩
    rw [rangeInt,if_neg hn,if_pos hlt,hm]
    exact ht
  · refine ⟨s,?_,ready,rfl⟩
    rw [rangeInt,if_neg hn,if_neg hlt]
termination_by (stop-cur).toNat
decreasing_by omega

/-- The actual source bounds evaluate before the loop; no loop success premise. -/
theorem bounds {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    (p : Layout) (r : Program.Regions) (storage : Storage) (scale : ℝ)
    (s : State α) (ready : Ready p storage s) :
    ∃ stop : Int,
      FP.Structural.evalOp (engine o.algebra) none (parts p r storage scale).start s = some (fun _ => (0 : Int)) ∧
      FP.Structural.evalOp (engine o.algebra) none (parts p r storage scale).stop s = some (fun _ => stop) ∧
      FP.Structural.evalOp (engine o.algebra) none (parts p r storage scale).stride s = some (fun _ => p.BN) := by
  rcases ready with ⟨_,hb,_,_,_,_,_,_,_,_,_,_,_,hm⟩
  obtain ⟨block,hb⟩ := Option.isSome_iff_exists.mp hb
  obtain ⟨mask,hm⟩ := Option.isSome_iff_exists.mp hm
  cases storage
  all_goals
    simp (config := {maxSteps := 100000}) [parts,program,
      VeriTile.Bench.TritonBenchG.ContextAttnMistral.Program.kernel,
      VeriTile.Bench.TritonBenchG.ContextAttnMistral.original,Source.intRange?,ComputeKernel.surfaceBody,
      evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,
      FPBackend.Ops.algebra,numeric,bop,hb,hm]
  all_goals
    refine ⟨(mask PUnit.unit : Int) * ((block PUnit.unit : Int) + 1) * (p.BM : Int),?_⟩
    funext i
    cases i
    rfl

def RegionFrame {α : Type} (out : RegionName) (initial final : State α) : Prop :=
  ∀ rg off, rg ≠ out → final.mem rg off = initial.mem rg off

theorem fold_frame {α ι : Type} (l : List ι) (f : State α → ι → State α)
    (s : State α) (rg : RegionName) (off : Nat)
    (hf : ∀ t i, (f t i).mem rg off = t.mem rg off) :
    (l.foldl f s).mem rg off = s.mem rg off := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [List.foldl_cons,ih,hf]

/-- The complete typed epilogue casts and stores the actual accumulator.
Its finite masked scatter preserves every cell outside the private output. -/
theorem epilogue {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    (p : Layout) (r : Program.Regions) (storage : Storage) (scale : ℝ)
    (s : State α) (ready : Ready p storage s) :
    ∃ t, run (engine o.algebra) (parts p r storage scale).after s = some t ∧
      RegionFrame r.out s t := by
  rcases ready with ⟨hh,_,hs,hl,hr,_,hd,_,_,_,_,_,ha,_⟩
  obtain ⟨head,hh⟩ := Option.isSome_iff_exists.mp hh
  obtain ⟨start,hs⟩ := Option.isSome_iff_exists.mp hs
  obtain ⟨length,hl⟩ := Option.isSome_iff_exists.mp hl
  obtain ⟨rows,hr⟩ := Option.isSome_iff_exists.mp hr
  obtain ⟨dims,hd⟩ := Option.isSome_iff_exists.mp hd
  obtain ⟨accumulator,ha⟩ := Option.isSome_iff_exists.mp ha
  cases storage
  all_goals
    simp (config := {maxSteps := 100000}) [parts,program,
      VeriTile.Bench.TritonBenchG.ContextAttnMistral.Program.kernel,
      VeriTile.Bench.TritonBenchG.ContextAttnMistral.original,Source.intRange?,ComputeKernel.surfaceBody,
      run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,FP.Structural.store,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,FPBackend.Ops.algebra,
      ComputeDType.eraseDType,numeric,natLt,natLe,numericLt,numericLe,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,integralFloorDiv,integralMod,
      hh,hs,hl,hr,hd,ha,RegionFrame]
  all_goals
    intro rg off hrg
    apply fold_frame
    intro t i
    split <;> simp_all [State.write]

/-- Full source successful execution for arbitrary shapes/strides, signed
metadata and window, storage branch and opaque numerical primitive values. -/
theorem complete {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    (p : Layout) (r : Program.Regions) (storage : Storage) (scale : ℝ)
    (s : State α) (hBN : 0 < p.BN) :
    ∃ t, FP.Structural.exec (engine o.algebra) (program p r storage scale) s = some t ∧
      RegionFrame r.out s t := by
  obtain ⟨initial,hpre,hready,hm⟩ := prelude o p r storage scale s
  obtain ⟨stop,hs,he,hd⟩ := bounds o p r storage scale initial hready
  obtain ⟨mid,hl,hr,hm'⟩ := loop o p r storage scale hBN 0 stop initial hready
  obtain ⟨final,hpost,hframe⟩ := epilogue o p r storage scale mid hr
  refine ⟨final,?_,?_⟩
  · change run (engine o.algebra) (program p r storage scale).surfaceBody s = some final
    rw [source_split p r storage scale,run_append,run_append,hpre]
    simp only [Option.bind_some,run]
    have hstep : step (engine o.algebra)
        (.forRangeInt "start_n" (parts p r storage scale).start (parts p r storage scale).stop
          (parts p r storage scale).stride (parts p r storage scale).body) initial = some mid := by
      simp only [step,hs,he,hd,Option.bind_some]
      exact hl
    rw [hstep]
    exact hpost
  · intro rg off hrg
    rw [hframe rg off hrg,hm',hm]
end
end VeriTile.Bench.Repairs.ContextAttnMistral.FPSymbolic
