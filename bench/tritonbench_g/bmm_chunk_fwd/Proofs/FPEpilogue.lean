import bench.tritonbench_g.bmm_chunk_fwd.Proofs.FPRange

namespace VeriTile.Bench.Repairs.BmmChunkFwd.FPSuccess
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem epilogue {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config)
    (initial s : State α) (ready : Ready h c s) (controls : Controls h c initial s) :
    ∃ t, run (engine o.algebra) (parts h c).after s=some t ∧ (∀ rg off, ¬Writes h c initial rg off → t.mem rg off=s.mem rg off) := by
  rcases ready with ⟨⟨h0,h1,h2,h3,h4⟩,h5,h6,h7,h8,h9,h10,h11,hseq⟩
  rcases controls with ⟨h0,h1,h2,h3,h4⟩
  obtain ⟨v11,h11⟩ := Option.isSome_iff_exists.mp h11
  cases seq : h.hasSeq
  · cases ha : h.aStorage <;> cases hb : h.bStorage
    all_goals
      simp (config := {maxSteps := 400000}) [parts,branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,
        ha,hb,seq,Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,
        run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
        Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
        numeric,natLt,natLe,numericLt,numericLe,numericEq,bop,ofFloat,toFloat,TileShape.axisDim,
        Region.cast,State.setReg,integralFloorDiv,integralMod,h0,h1,h2,h3,h4,h11,
        FP.Structural.store]
    all_goals
      intro rg off hmiss
      apply State.masked_scatter_frame
      by_cases hr : rg=h.Out
      · right
        intro i _ hi ho
        apply hmiss
        refine ⟨i,?_,hr,?_⟩
        · simpa [laneMask,row,column] using hi
        · simpa [offset,row,column,mul_ite] using ho
      · exact Or.inl hr
  · obtain ⟨vs,hs⟩ := Option.isSome_iff_exists.mp (hseq seq)
    cases ha : h.aStorage <;> cases hb : h.bStorage
    all_goals
      simp (config := {maxSteps := 400000}) [parts,branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,
        ha,hb,seq,Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,
        run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
        Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
        numeric,natLt,natLe,numericLt,numericLe,numericEq,bop,ofFloat,toFloat,TileShape.axisDim,
        Region.cast,State.setReg,integralFloorDiv,integralMod,h0,h1,h2,h3,h4,h11,hs,
        FP.Structural.store]
    all_goals
      intro rg off hmiss
      apply State.masked_scatter_frame
      by_cases hr : rg=h.Out
      · right
        intro i _ hi ho
        apply hmiss
        refine ⟨i,?_,hr,?_⟩
        · simpa [laneMask,row,column] using hi
        · simpa [offset,row,column,mul_ite] using ho
      · exact Or.inl hr

end
end VeriTile.Bench.Repairs.BmmChunkFwd.FPSuccess

#guard_msgs (drop info) in
#auditModuleAxioms
