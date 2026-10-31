import bench.tritonbench_g.context_attn_mistral.Execution

namespace VeriTile.Bench.Repairs.ContextAttnMistral.KernelReference
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ContextAttnMistral Formula
open VeriTile.Bench.TritonBenchG.ContextAttnMistral.Program (Regions)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem range (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α)
    (hBN : 0<p.BN) (initialState s : State α) (old : Acc p α)
    (live : Matches p r o (inputs p r initialState) old s) (memory : s.mem=initialState.mem)
    (cur endKey : Int) :
    ∃ t,rangeInt (engine o.algebra) "start_n" cur endKey p.BN (parts p r .fp32 scale).body s=some t ∧
      Matches p r o (inputs p r initialState)
        (iterate p o scale (inputs p r initialState) cur endKey old) t ∧ t.mem=initialState.mem := by
  by_cases hc : cur<endKey
  · obtain ⟨mid,he,live',memory'⟩ := body p r scale o hBN initialState s old live memory cur
    obtain ⟨t,ht,livem,memorym⟩ := range p r scale o hBN initialState mid
      (advance p o scale (inputs p r initialState) cur old) live' memory' (cur+p.BN) endKey
    refine ⟨t,?_,?_,memorym⟩
    · rw [rangeInt,if_neg (Nat.ne_of_gt hBN),if_pos hc,he]
      exact ht
    · rw [iterate,dif_neg (Nat.ne_of_gt hBN),dif_pos hc]
      exact livem
  · refine ⟨s,?_,?_,memory⟩
    · rw [rangeInt,if_neg (Nat.ne_of_gt hBN),if_neg hc]
    · rw [iterate,dif_neg (Nat.ne_of_gt hBN),dif_neg hc]
      exact live
termination_by (endKey-cur).toNat
decreasing_by omega

theorem loop (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α)
    (hBN : 0<p.BN) (initialState s : State α)
    (live : Matches p r o (inputs p r initialState) (initial p o) s) (memory : s.mem=initialState.mem) :
    ∃ t,step (engine o.algebra)
        (.forRangeInt "start_n" (parts p r .fp32 scale).start (parts p r .fp32 scale).stop
          (parts p r .fp32 scale).stride (parts p r .fp32 scale).body) s=some t ∧
      Matches p r o (inputs p r initialState)
        (iterate p o scale (inputs p r initialState) 0 (stop p (inputs p r initialState)) (initial p o)) t ∧
      t.mem=initialState.mem := by
  obtain ⟨t,ht,hr,hmem⟩ := range p r scale o hBN initialState s (initial p o) live memory 0 (stop p (inputs p r initialState))
  refine ⟨t,?_,hr,hmem⟩
  have hstart : evalOp (engine o.algebra) none (parts p r .fp32 scale).start s=some (fun _ => (0 : Int)) := rfl
  have hstride : evalOp (engine o.algebra) none (parts p r .fp32 scale).stride s=some (fun _ => p.BN) := rfl
  have hstop : evalOp (engine o.algebra) none (parts p r .fp32 scale).stop s=
      some (fun _ => stop p (inputs p r initialState)) := by
    rcases live with ⟨_,hb,_,_,_,_,_,_,_,_,_,_,_,hm⟩
    simp [parts,program,VeriTile.Bench.TritonBenchG.ContextAttnMistral.Program.kernel,
      VeriTile.Bench.TritonBenchG.ContextAttnMistral.original,VeriTile.Bench.Repairs.ContextAttnMistralKernel.kernel,VeriTile.Bench.Repairs.ContextAttnMistralKernel.storageProfile,Source.intRange?,ComputeKernel.surfaceBody,
      evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,FPBackend.Ops.algebra,
      resolvePrecision,numeric,bop,hb,hm,stop]
    rfl
  simpa only [step,hstart,hstride,hstop,Option.bind_some] using ht

def active (p : Layout) (r : Regions) (s : State α) (i : TileIndex [p.BM,p.D]) : Prop :=
  (row p (inputs p r s) i.1.val : Int)<(inputs p r s).length

def offset (p : Layout) (r : Regions) (s : State α) (i : TileIndex [p.BM,p.D]) : Nat :=
  (((inputs p r s).start+row p (inputs p r s) i.1.val)*p.outRow+
    ((inputs p r s).head*p.outHead : Nat)+(i.2.1.val*p.outDim : Nat)).toNat

def scatter (p : Layout) (r : Regions) (initialState s : State α) (v : Values α .real [p.BM,p.D]) : State α := by
  classical
  exact (TileShape.allIndices [p.BM,p.D]).foldl (fun t i =>
    if active p r initialState i then t.write r.out (offset p r initialState i) (.mk .real (v i)) else t) s

theorem masked_fold_mem_congr {I : Type} (rg : RegionName) (off : I → Nat)
    (v : I → Cell α) (on : I → Prop) [DecidablePred on] (lanes : List I)
    (s t : State α) (h : s.mem=t.mem) :
    (lanes.foldl (fun u i => if on i then u.write rg (off i) (v i) else u) s).mem=
      (lanes.foldl (fun u i => if on i then u.write rg (off i) (v i) else u) t).mem := by
  induction lanes generalizing s t with
  | nil => exact h
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : on i
    · simp only [hi,↓reduceIte]
      apply ih
      simp only [State.write,h]
    · simp only [hi,↓reduceIte]
      exact ih s t h

theorem scatter_mem_congr (p : Layout) (r : Regions) (initialState s t : State α)
    (v : Values α .real [p.BM,p.D]) (h : s.mem=t.mem) :
    (scatter p r initialState s v).mem=(scatter p r initialState t v).mem := by
  classical
  unfold scatter
  apply masked_fold_mem_congr
  exact h

def finish (p : Layout) (o : FPBackend.Ops α) (acc : Acc p α) : Values α .real [p.BM,p.D] :=
  fun i => o.scalar.cast (some .fp32) .real .real (acc.weighted i)

theorem epilogue (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α)
    (initialState s : State α) (acc : Acc p α) (live : Matches p r o (inputs p r initialState) acc s) :
    ∃ t,run (engine o.algebra) (parts p r .fp32 scale).after s=some t ∧
      t.mem=(scatter p r initialState s (finish p o acc)).mem := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13⟩
  simp [parts,program,VeriTile.Bench.TritonBenchG.ContextAttnMistral.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnMistral.original,VeriTile.Bench.Repairs.ContextAttnMistralKernel.kernel,VeriTile.Bench.Repairs.ContextAttnMistralKernel.storageProfile,ComputeKernel.surfaceBody,Source.intRange?]
  simp (config := {maxSteps := 100000}) [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,engine,Algebra.withDefaultPrecision,FPBackend.Ops.algebra,
    resolvePrecision,numeric,numericLt,natLt,ofFloat,toFloat,Region.cast,
    h0,h2,h3,h4,h6,h12,store,scatter,active,offset,inputs,row,finish,bop,
    Broadcast.leftIndex,Broadcast.rightIndex]
  apply masked_fold_mem_congr
  rfl

/-- Complete actual source execution computes the independent recurrence's
masked memory effect; successful execution is a conclusion. -/
theorem exec_complete (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α)
    (hBN : 0<p.BN) (s : State α) :
    ∃ t,exec (engine o.algebra) (program p r .fp32 scale) s=some t ∧
      t.mem=(scatter p r s s (output p o scale (inputs p r s))).mem := by
  obtain ⟨pre,hpre,live,memory⟩ := prelude p r scale o s
  obtain ⟨post,hloop,live',memory'⟩ := loop p r scale o hBN s pre live memory
  obtain ⟨final,hfinal,frame⟩ := epilogue p r scale o s post _ live'
  refine ⟨final,?_,?_⟩
  · change run _ (program p r .fp32 scale).surfaceBody s=some final
    rw [source_split,run_append,run_append,hpre]
    simp only [Option.bind_some,run,hloop]
    exact hfinal
  · rw [frame]
    exact scatter_mem_congr p r s post s _ memory'

#specNonCircular output avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.step,program,parts]
#axiomsClean exec_complete
end
end VeriTile.Bench.Repairs.ContextAttnMistral.KernelReference
