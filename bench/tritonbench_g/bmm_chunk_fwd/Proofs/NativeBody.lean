import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativeGraph

namespace VeriTile.Bench.Repairs.BmmChunkFwd.NativeGraph
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open FPSuccess
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Every actual default-FP32 K iteration stages one TF32 dotAcc with the
reached carry and the actual source-loaded A/B tile registers. The theorem
executes the entire body, including load masks, casts and pointer advances. -/
theorem body_fp32 (rest : Algebra Expr) (h : Host.Call) (c : Host.Config)
    (s : State Expr) (ready : Ready h c s) (index : Nat)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, ∃ a : Values Expr .real [c.M,c.K], ∃ b : Values Expr .real [c.K,c.N],
      ∃ carry : Values Expr .real [c.M,c.N],
      run (engine (backend rest c).algebra) (parts h c).body
        (s.setReg "k" .nat [] (fun _ => index))=some t ∧
      s.regs .real [c.M,c.N] "acc"=some carry ∧
      t.regs .real [c.M,c.K] "a"=some a ∧ t.regs .real [c.K,c.N] "b"=some b ∧
      t.regs .real [c.M,c.N] "acc"=some (fun i =>
        Expr.dotAcc c (some .fp32) .fp32 .fp32 .fp32 options [] c.M c.K c.N a b (carry i) i) := by
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
  simp (config := {maxSteps := 400000}) [parts,branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,
    ha,hb,Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
    Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
    numeric,natLt,natLe,numericLt,numericLe,bop,ofFloat,toFloat,TileShape.axisDim,
    Region.cast,State.setReg,integralFloorDiv,integralMod,
    h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,backend,algebra,consume]
  rfl

#axiomsClean body_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkFwd.NativeGraph

#guard_msgs (drop info) in
#auditModuleAxioms
