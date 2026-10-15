import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardRealRun

/-! Draft full frame theorem over the actual erased source execution witness.
The proof inspects only the final masked scatter; floating expressions do not
need to be simplified or replaced by a formula to establish this result.
-/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

theorem fp32_source_frame (Y X M : RegionName)
    (ym yn xm xn sm K D : Nat) (hD : 0 < D)
    (log : Bool) (mode : MaskMode) (causal fp16 : Bool) (s t : BlockState)
    (hExec : exec (fp32Source Y X M ym yn xm xn sm K D log mode causal fp16).eraseDType.toAlgKernel
      s = some t)
    (r : RegionName) (offset : Nat)
    (hmiss : ∀ i : Fin D, i.val < K →
      ¬(Y = r ∧ s.pids 0 * ym + s.pids 1 * yn + i.val = offset)) :
    t.mem r offset = s.mem r offset := by
  have ht := Option.some.inj
    ((fp32_source_runState Y X M ym yn xm xn sm K D hD log mode causal fp16 s).property.symm.trans
      hExec)
  subst t
  cases mode <;> cases log <;> cases causal <;> cases fp16
  all_goals
    dsimp only [fp32_source_runState]
    refine Eq.trans (foldl_store_preserve_cell _ _ _ r offset _ _ ?_) rfl
    intro i _ hi hc
    exact hmiss i.1 (by simpa using hi) hc

#axiomsClean fp32_source_frame
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward
