import bench.tritonbench_g.bmm_chunk_fwd.Proofs.FPSuccess
namespace VeriTile.Bench.Repairs.BmmChunkFwd.FPSuccess
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem body {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config)
    (s : State α) (ready : Ready h c s) (index : Nat) :
    ∃ t, run (engine o.algebra) (parts h c).body (s.setReg "k" .nat [] (fun _ => index))=some t ∧
      Ready h c t ∧ t.mem=s.mem ∧ Fixed s t := by
  rcases ready with ⟨⟨h0,h1,h2,h3,h4⟩,h5,h6,h7,h8,h9,h10,h11,hseq⟩
  obtain ⟨v0,h0⟩ := Option.isSome_iff_exists.mp h0
  obtain ⟨v1,h1⟩ := Option.isSome_iff_exists.mp h1
  obtain ⟨v2,h2⟩ := Option.isSome_iff_exists.mp h2
  obtain ⟨v3,h3⟩ := Option.isSome_iff_exists.mp h3
  obtain ⟨v4,h4⟩ := Option.isSome_iff_exists.mp h4
  obtain ⟨v5,h5⟩ := Option.isSome_iff_exists.mp h5
  obtain ⟨v6,h6⟩ := Option.isSome_iff_exists.mp h6
  obtain ⟨v7,h7⟩ := Option.isSome_iff_exists.mp h7
  obtain ⟨v8,h8⟩ := Option.isSome_iff_exists.mp h8
  obtain ⟨v9,h9⟩ := Option.isSome_iff_exists.mp h9
  obtain ⟨v10,h10⟩ := Option.isSome_iff_exists.mp h10
  obtain ⟨v11,h11⟩ := Option.isSome_iff_exists.mp h11
  cases ha : h.aStorage <;> cases hb : h.bStorage
  all_goals
    simp (config := {maxSteps := 400000}) [parts,branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,
      ha,hb,Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
      numeric,natLt,natLe,numericLt,numericLe,bop,ofFloat,toFloat,TileShape.axisDim,
      Region.cast,State.setReg,integralFloorDiv,integralMod,Ready,Base,Has,Fixed,
      h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,hseq]

  all_goals exact hseq

end
end VeriTile.Bench.Repairs.BmmChunkFwd.FPSuccess

#guard_msgs (drop info) in
#auditModuleAxioms
