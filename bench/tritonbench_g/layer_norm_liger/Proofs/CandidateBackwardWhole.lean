import bench.tritonbench_g.layer_norm_liger.Proofs.SignedRows
import bench.tritonbench_g.layer_norm_liger.Proofs.CandidateBackwardRun
namespace VeriTile.Bench.Repairs.LayerNormLiger.Backward.Candidate
open VeriTile.Triton FP.Structural
open LayerNormLiger.Backward
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_mem fold_writes_regs)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem range_run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s t : State α) (row finishRow : Nat) (a : Acc α p)
    (h : Ready M p s t row a) :
    ∃ u, FP.Structural.rangeInt (withSum M R) "_" (row : Int) (finishRow : Int) 1 (body p) t = some u ∧
      ∃ k, Ready M p s u k (foldRange (update M R p) row finishRow 1 a) := by
  apply SignedRows.range_values (withSum M R) "_" (body p) (update M R p)
    (fun n a t => Ready M p s t n a) (fun a t => ∃ k, Ready M p s t k a) finishRow
  · intro n a u hu _
    exact body_run M R p s u n a hu
  · intro n a u hu
    exact ⟨n,hu⟩
  · exact h

theorem postlude_run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s t : State α) (row : Nat) (a : Acc α p)
    (h : Ready M p s t row a) :
    ∃ u, run (withSum M R) (postlude p) t = some u ∧
      u.mem = (finish M p (s.pids 0) a).mem := by
  simp [postlude, candidateKernel, candidateSourcePastedKernel, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, evalOffsets, store, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, FP.Structural.bop, numericLt, natLt, Region.cast, withSum, VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.withSum,
    h.mem, h.pid, h.cols, h.sdw, h.sdb, h.mask, h.dw, h.db]
  try simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
  simp only [finish, fold_writes_mem, State.setReg_mem]
  simp [stored, h.mem]

theorem complete_run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec (withSum M R) (candidateKernel p) s = some t ∧
      t.mem = (reference M R p s).mem := by
  obtain ⟨u,hu,hr⟩ := prelude_run M R p s
  obtain ⟨v,hv,k,hk⟩ := range_run M R p s u (start p s) (stop p s) (initial M p s) hr
  obtain ⟨w,hw,hm⟩ := postlude_run M R p s v k _ hk
  refine ⟨w,?_,hm⟩
  change run (withSum M R) (candidateKernel p).surfaceBody s = some w
  rw [source_split, run_append, run_append, hu]
  simp only [Option.bind_some, run, step, evalOp_unfold, hr.first, hr.last,
    Option.bind_some, hv]
  simpa [Option.bind, hv] using hw

#axiomsClean range_run
#axiomsClean postlude_run
#axiomsClean complete_run
end
end VeriTile.Bench.Repairs.LayerNormLiger.Backward.Candidate
