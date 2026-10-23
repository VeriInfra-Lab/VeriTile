import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.FPReference

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.FPReference
open VeriTile.Triton FP.Structural FPSuccess
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem range (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : DotSupport p M) (hBN : 0 < p.BN) (initialState s : State α)
    (old : Acc p α) (live : Matches p M (inputs p r initialState) old s)
    (memory : s.mem=initialState.mem) (cur stop : Int) :
    ∃ t,rangeInt (M.withDefaultPrecision .fp32) "start_n" cur stop p.BN
        (parts p r scale).body s=some t ∧
      Matches p M (inputs p r initialState)
        (iterate p M dots scale (inputs p r initialState) cur stop old) t ∧
      t.mem=initialState.mem := by
  by_cases hc : cur < stop
  · obtain ⟨mid,he,live',memory'⟩ := body p r scale M dots hBN initialState s old live memory cur
    obtain ⟨t,ht,livem,memorym⟩ := range p r scale M dots hBN initialState mid
      (advance p M dots scale (inputs p r initialState) cur old) live' memory' (cur+p.BN) stop
    refine ⟨t,?_,?_,memorym⟩
    · rw [rangeInt,if_neg (Nat.ne_of_gt hBN),if_pos hc,he]
      exact ht
    · rw [iterate,dif_neg (Nat.ne_of_gt hBN),dif_pos hc]
      exact livem
  · refine ⟨s,?_,?_,memory⟩
    · rw [rangeInt,if_neg (Nat.ne_of_gt hBN),if_neg hc]
    · rw [iterate,dif_neg (Nat.ne_of_gt hBN),dif_neg hc]
      exact live
termination_by (stop-cur).toNat
decreasing_by omega

theorem loop (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : DotSupport p M) (hBN : 0 < p.BN) (initialState s : State α)
    (live : Matches p M (inputs p r initialState) (initial p M) s)
    (memory : s.mem=initialState.mem) :
    ∃ t,step (M.withDefaultPrecision .fp32)
      (.forRangeInt "start_n" (parts p r scale).start (parts p r scale).stop
        (parts p r scale).stride (parts p r scale).body) s=some t ∧
      Matches p M (inputs p r initialState)
        (iterate p M dots scale (inputs p r initialState) 0
          ((blockMask p (inputs p r initialState) : Int)*keyEnd p (inputs p r initialState))
          (initial p M)) t ∧ t.mem=initialState.mem := by
  have he := live.2.2.1
  have hm := live.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨t,ht,hr,hmem⟩ := range p r scale M dots hBN initialState s (initial p M) live memory 0
    ((blockMask p (inputs p r initialState) : Int)*keyEnd p (inputs p r initialState))
  refine ⟨t,?_,hr,hmem⟩
  have hstart : evalOp (M.withDefaultPrecision .fp32) none (parts p r scale).start s=
      some (fun _ => (0 : Int)) := rfl
  have hstride : evalOp (M.withDefaultPrecision .fp32) none (parts p r scale).stride s=
      some (fun _ => p.BN) := rfl
  have hstop : evalOp (M.withDefaultPrecision .fp32) none (parts p r scale).stop s=
      some (fun _ => (blockMask p (inputs p r initialState) : Int)*keyEnd p (inputs p r initialState)) := by
    change evalOp (M.withDefaultPrecision .fp32) none
      (.mul .int .nil (.castNatToInt (.ref .nat [] "block_mask"))
        (.ref .int [] "block_end_loc")) s = _
    simp only [evalOp_unfold,hm,he,Option.bind,numeric]
    rfl
  simpa only [step,hstart,hstride,hstop,Option.bind_some] using ht

def finish (p : Layout) (M : Algebra α) (acc : Acc p α) : Values α .real [p.BM,p.D] :=
  fun i => M.cast (some .fp32) .real .real
    (M.binary (some .fp32) .real .div (acc.weighted i) (acc.mass (i.1,())))

theorem epilogue (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (initialState s : State α) (acc : Acc p α) (live : Matches p M (inputs p r initialState) acc s) :
    ∃ t,run (M.withDefaultPrecision .fp32) (parts p r scale).after s=some t ∧
      t.mem=(scatter p r initialState s (finish p M acc)).mem := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  simp [parts,kernel,originalFP32,
    VeriTile.Bench.TritonBenchG.ContextAttnLlama.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnLlama.original,ComputeKernel.surfaceBody,
    Source.intRange?]
  simp (config := {maxSteps := 100000}) [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,Algebra.withDefaultPrecision,
    resolvePrecision,numeric,numericLt,natLt,ofFloat,toFloat,Region.cast,
    h9,h10,h11,h12,h5,h4,h13,store,scatter,active,offset,FPSuccess.row,FPSuccess.start,FPSuccess.length,
    inputs,row,finish,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  apply masked_fold_mem_congr
  rfl

/-- Constructive source-to-independent-recurrence theorem for the entire
prelude, signed key loop and epilogue. No output agreement is a premise. -/
theorem program (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : DotSupport p M) (hBN : 0 < p.BN) (s : State α) :
    ∃ t,exec (M.withDefaultPrecision .fp32) (kernel p r .fp32 scale) s=some t ∧
      t.mem=(scatter p r s s (output p M dots scale (inputs p r s))).mem := by
  obtain ⟨pre,hpre,live,memory⟩ := prelude p r scale M s
  obtain ⟨post,hloop,live',memory'⟩ := loop p r scale M dots hBN s pre live memory
  obtain ⟨final,hfinal,frame⟩ := epilogue p r scale M s post _ live'
  refine ⟨final,?_,?_⟩
  · change run _ (kernel p r .fp32 scale).surfaceBody s=some final
    rw [source_split,run_append,run_append,hpre]
    simp only [Option.bind_some,run,hloop]
    exact hfinal
  · rw [frame]
    exact scatter_mem_congr p r s post s _ memory'

#specNonCircular output avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.step,kernel,parts]
#axiomsClean program
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.FPReference
