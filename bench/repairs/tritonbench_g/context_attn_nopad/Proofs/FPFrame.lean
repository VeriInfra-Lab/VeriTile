import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.FPComplete

namespace VeriTile.Bench.Repairs.ContextAttnNopad.FPSymbolic
open VeriTile.Triton FP.Structural
open FP.ProfiledRewrite (engine)
open VeriTile.Bench.TritonBenchG.ContextAttnNopad
open Formula
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Exact masked-store frame for arbitrary opaque payloads, including every
inactive row/channel and allocation hole inside the output region itself. -/
theorem epilogue_precise {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    (p : Layout) (r : Program.Regions) (scale : ℝ) (initial s : State α)
    (ready : Ready p s) (geometry : Geometry p r initial s) :
    ∃ t, run (engine o.algebra) (parts p r scale).after s = some t ∧
      ∀ rg off, ¬ Writes p r initial rg off → t.mem rg off = s.mem rg off := by
  rcases geometry with ⟨hh,hs,hl,hr,hd⟩
  rcases ready with ⟨_,_,_,_,_,_,_,_,_,_,_,_,ha,_⟩
  obtain ⟨accumulator,ha⟩ := Option.isSome_iff_exists.mp ha
  simp (config := {maxSteps := 300000}) [parts,program,Program.parts,Program.mathematical,Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnNopad.original,Source.intRange?,ComputeKernel.surfaceBody,
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,store,
    engine,Algebra.withDefaultPrecision,resolvePrecision,FPBackend.Ops.algebra,
    ComputeDType.eraseDType,numeric,natLt,natLe,numericLt,numericLe,numericEq,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,integralFloorDiv,integralMod,
    hh,hs,hl,hr,hd,ha]
  intro rg off hnot
  apply State.masked_scatter_frame
  by_cases he : rg = r.out
  · right
    intro i _ hi hoff
    apply hnot
    refine ⟨i,?_,he.symm,?_⟩
    · simpa only [RealIO.active,Formula.sequence] using hi
    · simpa only [RealIO.write,AddressBounds.outputAddress,launchOf,
        Int.natCast_mul] using hoff.symm
  · exact Or.inl he

/-- Whole typed execution with the exact source active-write footprint. -/
theorem complete_precise {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    (p : Layout) (r : Program.Regions) (scale : ℝ) (hBN : 0 < p.BN) (s : State α) :
    ∃ t, FP.Structural.exec (engine o.algebra) (program p r scale) s = some t ∧
      ∀ rg off, ¬ Writes p r s rg off → t.mem rg off = s.mem rg off := by
  obtain ⟨pre,hpre,ready,hpm,hgeo⟩ := prelude o p r scale s
  obtain ⟨post,hloop,ready',hlm,hcontrols⟩ := loop_complete o p r scale hBN pre ready
  obtain ⟨final,hfinal,hfm⟩ := epilogue_precise o p r scale s post ready' (hgeo.preserve hcontrols)
  refine ⟨final,?_,?_⟩
  · change run _ (program p r scale).surfaceBody s = some final
    rw [source_split,List.append_assoc,run_append,hpre]
    simpa only [Option.bind_some,List.singleton_append,run,hloop] using hfinal
  · intro rg off hrg
    rw [hfm rg off hrg,hlm,hpm]

end
end VeriTile.Bench.Repairs.ContextAttnNopad.FPSymbolic
