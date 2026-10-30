import bench.repairs.tritonbench_g.context_attn_bloom.Whole001.FPProgram
import bench.repairs.tritonbench_g.context_attn_bloom.Whole001.FPScatter

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.FPSuccess
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem loop_success (p : Layout) (r : Regions) (scale : ℝ)
    (base : Algebra α) (D : Dot α) (hBN : 0 < p.BN)
    (s : State α) (hs : Ready p s) :
    ∃ t, step ((withDot base D).withDefaultPrecision .fp32)
      (.forRangeInt "start_n" (parts p r scale).start (parts p r scale).stop
        (parts p r scale).stride (parts p r scale).body) s = some t ∧
      Ready p t ∧ t.mem = s.mem ∧ Fixed p s t := by
  have he := hs.2.2.1
  have hm := hs.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  rcases Option.isSome_iff_exists.mp he with ⟨e,he⟩
  rcases Option.isSome_iff_exists.mp hm with ⟨m,hm⟩
  obtain ⟨t,ht,hrt,hmt,hft⟩ := range_success p r scale base D hBN 0
    ((m PUnit.unit : Int) * e PUnit.unit) s hs
  refine ⟨t,?_,hrt,hmt,hft⟩
  have hstart : evalOp ((withDot base D).withDefaultPrecision .fp32) none (parts p r scale).start s =
      some (fun _ => (0 : Int)) := rfl
  have hstride : evalOp ((withDot base D).withDefaultPrecision .fp32) none (parts p r scale).stride s =
      some (fun _ => p.BN) := rfl
  have hstop : evalOp ((withDot base D).withDefaultPrecision .fp32) none (parts p r scale).stop s =
      some (fun _ => (m PUnit.unit : Int) * e PUnit.unit) := by
    change evalOp ((withDot base D).withDefaultPrecision .fp32) none
      (.mul .int .nil (.castNatToInt (.ref .nat [] "block_mask"))
        (.ref .int [] "block_end_loc")) s = _
    simp only [evalOp_unfold,hm,he,Option.bind,numeric]
    rfl
  simpa only [step,hstart,hstride,hstop,Option.bind_some] using ht

/-- The actual typed epilogue produces values with the opaque division and cast
callbacks. Its only writes are exactly the source's active output lanes. -/
theorem epilogue_success (p : Layout) (r : Regions) (scale : ℝ)
    (M : Algebra α) (initial s : State α) (hs : Ready p s)
    (hc : Controls p r initial s) :
    ∃ v t, run (M.withDefaultPrecision .fp32) (parts p r scale).after s = some t ∧
      t.mem = (scatter p r initial s v).mem := by
  rcases hs with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  rcases Option.isSome_iff_exists.mp h9 with ⟨l,h9⟩
  rcases Option.isSome_iff_exists.mp h10 with ⟨a,h10⟩
  rcases hc with ⟨hs,hh,hm,hd,hl⟩
  refine ⟨(fun i => M.cast (some .fp32) .real .real
    (a i)),?_⟩
  simp [parts,kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnBloom.Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnBloom.original,ComputeKernel.surfaceBody,
    Source.intRange?]
  simp (config := {maxSteps := 100000}) [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,Algebra.withDefaultPrecision,
    resolvePrecision,numeric,numericLt,natLt,ofFloat,toFloat,Region.cast,
    h9,h10,hs,hh,hm,hd,hl,store,scatter,active,offset,row,bop,
    Broadcast.leftIndex,Broadcast.rightIndex]
  apply masked_fold_mem_congr
  rfl

/-- Successful execution of the actual typed FP32 source. The only floating
interface premises are the two local dot implementations; all other floating
results, including NaN/infinity and reductions, remain arbitrary. -/
theorem program_success (p : Layout) (r : Regions) (scale : ℝ)
    (base : Algebra α) (D : Dot α) (hBN : 0 < p.BN) (s : State α) :
    ∃ v t, exec ((withDot base D).withDefaultPrecision .fp32) (kernel p r scale) s = some t ∧
      t.mem = (scatter p r s s v).mem := by
  obtain ⟨a,ha,hra,hma,hca⟩ := prelude_success p r scale (withDot base D) s
  obtain ⟨b,hb,hrb,hmb,hfb⟩ := loop_success p r scale base D hBN a hra
  obtain ⟨v,t,ht,hmt⟩ := epilogue_success p r scale (withDot base D) s b hrb (hca.fixed hfb)
  refine ⟨v,t,?_,?_⟩
  · change run _ (kernel p r scale).surfaceBody s = some t
    rw [source_split,run_append,run_append,ha]
    simp only [Option.bind_some,run,hb]
    exact ht
  · rw [hmt]
    exact scatter_mem_congr p r s b s v (hmb.trans hma)

end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.FPSuccess
