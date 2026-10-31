import bench.tritonbench_g.context_attn_llama.Proofs.NativeLoopSafety

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeSourceSafety
open VeriTile Triton FP.Structural FPSuccess FPReference
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem after (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (initialState s : State α) (acc : Acc p α) (live : Matches p M (inputs p r initialState) acc s)
    (bounds : RegionName → Nat) (access : Bounds p r (inputs p r initialState) bounds) :
    Safety.traceSafeList (M.withDefaultPrecision .fp32) bounds (parts p r scale).after s := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  simp [parts,kernel,originalFP32,VeriTile.Bench.TritonBenchG.ContextAttnLlama.Program.kernel,VeriTile.Bench.TritonBenchG.ContextAttnLlama.original,VeriTile.Bench.Repairs.ContextAttnLlama.SelectedSource.original,VeriTile.Bench.Repairs.ContextAttnLlama.SelectedSource.original,VeriTile.Bench.TritonBenchG.ContextAttnLlama.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnLlama.original,VeriTile.Bench.Repairs.ContextAttnLlama.SelectedSource.original,ComputeKernel.surfaceBody,Source.intRange?]
  simp (config := {maxSteps:=250000}) [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,numeric,numericLt,numericLe,natLt,
    bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,integralFloorDiv,integralMod,
    State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
    Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,
    h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14]
  intro i d on
  simpa only [Nat.cast_mul] using access.output i d on

theorem range (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : DotSupport p M) (hBN : 0<p.BN) (initialState s : State α)
    (old : Acc p α) (live : Matches p M (inputs p r initialState) old s)
    (memory : s.mem=initialState.mem) (cur stop : Int) (nonnegative : 0≤cur)
    (bounds : RegionName → Nat) (access : Bounds p r (inputs p r initialState) bounds) :
    Safety.rangeIntSafe (M.withDefaultPrecision .fp32) bounds "start_n" cur stop p.BN
      (parts p r scale).body s := by
  rw [Safety.rangeIntSafe_unfold,if_neg (Nat.ne_of_gt hBN)]
  by_cases inside:cur<stop
  · rw [if_pos inside]
    refine ⟨body p r scale M dots hBN initialState s old live memory cur nonnegative bounds access,?_⟩
    obtain ⟨mid,ran,next,mem⟩:=FPReference.body p r scale M dots hBN initialState s old live memory cur
    rw [ran]
    exact range p r scale M dots hBN initialState mid _ next mem (cur+p.BN) stop
      (by omega) bounds access
  · rw [if_neg inside]
    trivial
termination_by (stop-cur).toNat
decreasing_by omega

theorem whole (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : DotSupport p M) (hBN : 0<p.BN) (s : State α)
    (bounds : RegionName → Nat) (access : Bounds p r (inputs p r s) bounds) :
    TraceSafe (M.withDefaultPrecision .fp32) bounds (kernel p r .fp32 scale) s := by
  change Safety.traceSafeList _ _ (kernel p r .fp32 scale).surfaceBody s
  rw [source_split,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨prelude p r scale M s bounds access,?_⟩
  intro pre ran
  obtain ⟨pre',hpre,live,memory⟩:=FPReference.prelude p r scale M s
  rw [hpre] at ran
  have same : pre = pre' := (Option.some.inj ran).symm
  subst pre
  simp only [List.singleton_append,Safety.traceSafeList_cons]
  have he:=live.2.2.1
  have hm:=live.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have hstart : evalOp (M.withDefaultPrecision .fp32) none (parts p r scale).start pre'=
      some (fun _=>(0 : Int)) := rfl
  have hstride : evalOp (M.withDefaultPrecision .fp32) none (parts p r scale).stride pre'=
      some (fun _=>p.BN) := rfl
  have hstop : evalOp (M.withDefaultPrecision .fp32) none (parts p r scale).stop pre'=
      some (fun _=>(blockMask p (inputs p r s) : Int)*keyEnd p (inputs p r s)) := by
    change evalOp _ _ (.mul .int .nil (.castNatToInt (.ref .nat [] "block_mask"))
      (.ref .int [] "block_end_loc")) pre'=_
    simp only [evalOp_unfold,hm,he,Option.bind,numeric]
    rfl
  refine ⟨?_,?_⟩
  · simp only [Safety.stepSafe_unfold,hstart,hstride,hstop,Option.bind_some]
    refine ⟨?_,?_,?_,
      range p r scale M dots hBN s pre' _ live memory 0 _ (by omega) bounds access⟩
    · change Safety.safeOp _ _ _ _ (.constInt 0)
      simp [Safety.safeOp_unfold]
    · change Safety.safeOp _ _ _ _ (.mul .int .nil (.castNatToInt (.ref .nat [] "block_mask"))
        (.ref .int [] "block_end_loc"))
      simp [Safety.safeOp_unfold]
    · change Safety.safeOp _ _ _ _ (.constNat p.BN)
      simp [Safety.safeOp_unfold]
  · intro post executed
    obtain ⟨post',hloop,finalLive,mem⟩:=FPReference.loop p r scale M dots hBN s pre' live memory
    rw [hloop] at executed
    have same : post = post' := (Option.some.inj executed).symm
    subst post
    exact after p r scale M s post' _ finalLive bounds access

#axiomsClean after
#axiomsClean range
#axiomsClean whole
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeSourceSafety
