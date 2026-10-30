import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardRun
namespace VeriTile.Bench.Repairs.LayerNormOps.Backward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_mem fold_writes_regs)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem range_run {α : Type} [Inhabited α] (M : Algebra α)
    (p : BackwardParameters) (s t : State α) (row finishRow : Nat) (a : Acc α p)
    (h : Ready p s t row a) :
    ∃ u, FP.Structural.range M "row" row finishRow 1 (body p) t = some u ∧
      ∃ k, Ready p s u k (foldRange (update M p) row finishRow 1 a) := by
  apply range_values M "row" (body p) (update M p)
    (fun n a t => Ready p s t n a) (fun a t => ∃ k, Ready p s t k a)
    row finishRow 1 a t
  · intro n a u hu _
    exact body_run M p s u n a hu
  · intro n a u hu
    exact ⟨n,hu⟩
  · exact h

theorem postlude_run {α : Type} [Inhabited α] (M : Algebra α)
    (p : BackwardParameters) (s t : State α) (row : Nat) (a : Acc α p)
    (h : Ready p s t row a) :
    ∃ u, run M (postlude p) t = some u ∧
      u.mem = (finish M p (s.pids 0) a).mem := by
  cases hr : p.rms <;> cases hd : p.hasDres <;> cases hs : p.storeDres <;>
    cases hy : p.recompute <;> cases hb : p.hasBias
  all_goals
    simp [postlude, isRowLoop, hr,hd,hs,hy, backwardKernel, backwardSource, ComputeKernel.surfaceBody, run, step, evalExpr,
      evalComputeOp, evalOp_unfold, evalOffsets, store, Option.bind, ComputeDType.eraseDType,
      FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
      numeric, FP.Structural.bop, numericLt, natLt, Region.cast, hb,
      h.mem, h.pid, h.cols, h.mask, h.dw, h.db]
    simp only [finish, scatter, fold_writes_mem, State.setReg_mem]
    simp [Forward.stored, h.mem, hb, fold_writes_mem]

theorem complete_run {α : Type} [Inhabited α] (M : Algebra α)
    (p : BackwardParameters) (s : State α) :
    ∃ t, FP.Structural.exec M (backwardKernel p) s = some t ∧
      t.mem = (reference M p s).mem := by
  obtain ⟨u,hu,hr⟩ := prelude_run M p s
  obtain ⟨v,hv,k,hk⟩ := range_run M p s u (start p s) (stop p s) (initial M p s) hr
  obtain ⟨w,hw,hm⟩ := postlude_run M p s v k _ hk
  refine ⟨w,?_,hm⟩
  change run M (backwardKernel p).surfaceBody s = some w
  rw [source_split, run_append, run_append, hu]
  simp only [Option.bind_some, run, step, evalOp_unfold, hr.first, hr.last,
    Option.bind_some, hv]
  simpa [Option.bind, hv] using hw

#axiomsClean range_run
#axiomsClean postlude_run
#axiomsClean complete_run
end
end VeriTile.Bench.Repairs.LayerNormOps.Backward
