import bench.tritonbench_g.context_attn_bloom.Proofs.IOSourceSafety

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.IOSourceSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program
open FPSuccess FPReference
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000
variable {α : Type} [Inhabited α]

theorem range_safe (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : Dot α) (hBN : 0<p.BN) (initialState s : State α) (old : Acc p α)
    (live : Matches p M (FPReference.inputs p r initialState) old s)
    (memory : s.mem=initialState.mem) (cur : Int) (nonnegative : 0≤cur)
    (aligned : ∃ n : Nat,cur=(n*p.BN : Nat))
    (bounds : RegionName→Nat) (hb : Bounds p r (FPReference.inputs p r initialState) bounds) :
    Safety.rangeIntSafe ((withDot M dots).withDefaultPrecision .fp32) bounds "start_n" cur
      ((FPReference.blockMask p (FPReference.inputs p r initialState) : Int)*
        FPReference.keyEnd p (FPReference.inputs p r initialState)) p.BN
      (parts p r scale).body s := by
  rw [Safety.rangeIntSafe_unfold]
  simp only [Nat.ne_of_gt hBN,if_false]
  by_cases active : cur<(FPReference.blockMask p (FPReference.inputs p r initialState) : Int)*
      FPReference.keyEnd p (FPReference.inputs p r initialState)
  · simp only [active,if_true]
    refine ⟨body_safe p r scale M dots hBN initialState s old live memory cur nonnegative aligned active bounds hb,?_⟩
    obtain ⟨t,ran,next,stable⟩ := FPReference.body p r scale M dots hBN initialState s old live memory cur
    rw [ran]
    apply range_safe p r scale M dots hBN initialState t _ next stable (cur+p.BN) (by omega) ?_ bounds hb
    rcases aligned with ⟨n,hn⟩
    exact ⟨n+1,by simp [hn,Nat.add_mul]⟩
  · simp [active]
termination_by ((FPReference.blockMask p (FPReference.inputs p r initialState) : Int)*
  FPReference.keyEnd p (FPReference.inputs p r initialState)-cur).toNat
decreasing_by omega

theorem loop_safe (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : Dot α) (hBN : 0<p.BN) (initialState s : State α)
    (live : Matches p M (FPReference.inputs p r initialState) (FPReference.initial p M) s)
    (memory : s.mem=initialState.mem)
    (bounds : RegionName→Nat) (hb : Bounds p r (FPReference.inputs p r initialState) bounds) :
    Safety.stepSafe ((withDot M dots).withDefaultPrecision .fp32) bounds
      (.forRangeInt "start_n" (parts p r scale).start (parts p r scale).stop
        (parts p r scale).stride (parts p r scale).body) s := by
  have he := live.2.2.1
  have hm := live.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  change Safety.stepSafe ((withDot M dots).withDefaultPrecision .fp32) bounds
    (.forRangeInt "start_n" (.constInt 0)
      (.mul .int .nil (.castNatToInt (.ref .nat [] "block_mask")) (.ref .int [] "block_end_loc"))
      (.constNat p.BN) (parts p r scale).body) s
  simp only [Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,hm,he,Option.bind,numeric,true_and]
  exact range_safe p r scale M dots hBN initialState s _ live memory 0 (by omega)
    ⟨0,by simp⟩ bounds hb

theorem kernel_safe (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : Dot α) (hBN : 0<p.BN) (s : State α)
    (bounds : RegionName→Nat) (hb : Bounds p r (FPReference.inputs p r s) bounds) :
    TraceSafe ((withDot M dots).withDefaultPrecision .fp32) bounds (kernel p r scale) s := by
  obtain ⟨pre,ran,live,stable⟩ := FPReference.prelude p r scale (withDot M dots) s
  obtain ⟨post,loopRan,next,memory⟩ := FPReference.loop p r scale M dots hBN s pre live stable
  unfold TraceSafe
  rw [source_split]
  simp only [List.append_assoc,Safety.traceSafeList_append]
  refine ⟨prelude_safe p r scale (withDot M dots) s bounds hb,?_⟩
  intro t ht
  have eq := Option.some.inj (ht.symm.trans ran)
  subst t
  refine ⟨Safety.traceSafeList_cons_intro (loop_safe p r scale M dots hBN s pre live stable bounds hb)
    (fun _ _=>Safety.traceSafeList_nil _ _ _),?_⟩
  intro t ht
  have equal : post=t := by
    simp only [run,loopRan] at ht
    exact Option.some.inj ht
  subst t
  exact epilogue_safe p r scale (withDot M dots) s post _ next bounds hb

#axiomsClean kernel_safe
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.IOSourceSafety
