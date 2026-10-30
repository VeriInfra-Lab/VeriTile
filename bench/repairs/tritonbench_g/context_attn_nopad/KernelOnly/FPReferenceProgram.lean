import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.FPReference
import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Scatter

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.FPReference
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Formula Program
open FP.ProfiledRewrite (engine)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem range (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α)
    (hBN : 0 < p.BN) (initialState s : State α) (old : Acc p α)
    (live : Matches p r o.algebra (inputs r initialState) old s)
    (memory : s.mem=initialState.mem) (cur stop : Int) :
    ∃ t,rangeInt (engine o.algebra) "start_n" cur stop p.BN (FPSymbolic.parts p r scale).body s=some t ∧
      Matches p r o.algebra (inputs r initialState)
        (iterate p o scale (inputs r initialState) cur stop old) t ∧ t.mem=initialState.mem := by
  by_cases hc : cur < stop
  · obtain ⟨mid,he,live',memory'⟩ := body p r scale o hBN initialState s old live memory cur
    obtain ⟨t,ht,livem,memorym⟩ := range p r scale o hBN initialState mid
      (advance p o scale (inputs r initialState) cur old) live' memory' (cur+p.BN) stop
    refine ⟨t,?_,?_,memorym⟩
    · rw [rangeInt,if_neg (Nat.ne_of_gt hBN),if_pos hc,he]; exact ht
    · rw [iterate,dif_neg (Nat.ne_of_gt hBN),dif_pos hc]; exact livem
  · refine ⟨s,?_,?_,memory⟩
    · rw [rangeInt,if_neg (Nat.ne_of_gt hBN),if_neg hc]
    · rw [iterate,dif_neg (Nat.ne_of_gt hBN),dif_neg hc]; exact live
termination_by (stop-cur).toNat
decreasing_by omega

theorem loop (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α)
    (hBN : 0 < p.BN) (initialState s : State α)
    (live : Matches p r o.algebra (inputs r initialState) (initial p o.algebra) s)
    (memory : s.mem=initialState.mem) :
    ∃ t,step (engine o.algebra)
      (.forRangeInt "start_n" (FPSymbolic.parts p r scale).start (FPSymbolic.parts p r scale).stop
        (FPSymbolic.parts p r scale).stride (FPSymbolic.parts p r scale).body) s=some t ∧
      Matches p r o.algebra (inputs r initialState)
        (iterate p o scale (inputs r initialState) 0
          ((blockMask p (inputs r initialState)*((inputs r initialState).block+1)*p.BM : Nat) : Int)
          (initial p o.algebra)) t ∧ t.mem=initialState.mem := by
  have hb := live.2.1
  have hm := live.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨t,ht,hr,hmem⟩ := range p r scale o hBN initialState s (initial p o.algebra) live memory 0
    ((blockMask p (inputs r initialState)*((inputs r initialState).block+1)*p.BM : Nat) : Int)
  refine ⟨t,?_,hr,hmem⟩
  simpa (config := {maxSteps := 300000}) [FPSymbolic.parts,Program.parts,Program.mathematical,Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnNopad.original,Source.intRange?,ComputeKernel.surfaceBody,
    step,evalOp_unfold,Option.bind,numeric,bop,State.setReg,hb,hm] using ht

def active (p : Layout) (a : Inputs α) (i : TileIndex [p.BM,p.D]) : Prop :=
  (row p a i.1.val : Int) < a.length

def offset (p : Layout) (a : Inputs α) (i : TileIndex [p.BM,p.D]) : Nat :=
  ((a.start+row p a i.1.val)*p.outRow+(a.head*p.outHead : Nat)+(i.2.1.val*p.outDim : Nat)).toNat

def scatter (p : Layout) (r : Regions) (a : Inputs α) (s : State α)
    (values : Values α .real [p.BM,p.D]) : State α := by
  classical
  exact (TileShape.allIndices [p.BM,p.D]).foldl (fun t i =>
    if active p a i then t.write r.out (offset p a i) (.mk .real (values i)) else t) s

theorem scatter_mem_congr (p : Layout) (r : Regions) (a : Inputs α) (s t : State α)
    (values : Values α .real [p.BM,p.D]) (memory : s.mem=t.mem) :
    (scatter p r a s values).mem=(scatter p r a t values).mem := by
  classical
  exact VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.masked_fold_mem_congr _ _ _ _ _ _ _ memory

theorem epilogue (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α)
    (initialState s : State α) (acc : Acc p α) (live : Matches p r o.algebra (inputs r initialState) acc s) :
    ∃ t,run (engine o.algebra) (FPSymbolic.parts p r scale).after s=some t ∧
      t.mem=(scatter p r (inputs r initialState) s
        (fun i => o.algebra.cast (some .fp32) .real .real (acc.weighted i))).mem := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13⟩
  simp [FPSymbolic.parts,Program.parts,Program.mathematical,Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnNopad.original,ComputeKernel.surfaceBody,Source.intRange?]
  simp (config := {maxSteps := 100000}) [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,Algebra.withDefaultPrecision,engine,FPBackend.Ops.algebra,
    resolvePrecision,numeric,numericLt,natLt,ofFloat,toFloat,Region.cast,
    h0,h2,h3,h4,h6,h12,store,scatter,active,offset,inputs,row,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  apply VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.masked_fold_mem_congr
  rfl

theorem program (p : Layout) (r : Regions) (scale : ℝ) (o : FPBackend.Ops α)
    (hBN : 0 < p.BN) (s : State α) :
    ∃ t,FP.Structural.exec (engine o.algebra) (FPSymbolic.program p r scale) s=some t ∧
      t.mem=(scatter p r (inputs r s) s (output p o scale (inputs r s))).mem := by
  obtain ⟨pre,hpre,live,memory⟩ := prelude p r scale o s
  obtain ⟨post,hloop,live',memory'⟩ := loop p r scale o hBN s pre live memory
  obtain ⟨final,hfinal,frame⟩ := epilogue p r scale o s post _ live'
  refine ⟨final,?_,?_⟩
  · change run _ (FPSymbolic.program p r scale).surfaceBody s=some final
    rw [FPSymbolic.source_split,List.append_assoc,run_append,hpre]
    simpa only [Option.bind_some,List.singleton_append,run,hloop] using hfinal
  · rw [frame]
    exact scatter_mem_congr p r (inputs r s) post s _ memory'

#specNonCircular output avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.step,FPSymbolic.program,FPSymbolic.parts]
#axiomsClean program
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.FPReference
