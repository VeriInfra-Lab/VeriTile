import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.FPLoop
namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false

theorem scatterMemory_congr {α ι : Type} (l : List ι) (s t : State α)
    (R : RegionName) (offset : ι → Nat) (value : ι → Cell α) (h : s.mem=t.mem) :
    (l.foldl (fun u i => u.write R (offset i) (value i)) s).mem =
    (l.foldl (fun u i => u.write R (offset i) (value i)) t).mem := by
  induction l generalizing s t with
  | nil => exact h
  | cons i tail ih =>
    simp only [List.foldl_cons]
    exact ih _ _ (by simp only [State.write,h])

theorem suffix_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (a : Accumulator α p.dimensions) (h : Ready p s t a) :
    ∃ final, run M (suffix p) t = some final ∧
      final.mem = (written p s (outputValues M p a)).mem := by
  simp [suffix,source,sourceBody,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
    Region.cast,ofFloat,toFloat,store,State.setReg,
    h.batch,h.head,h.dimensions,h.output,h.normalizer,written,outputOffset,outputValues]
  apply scatterMemory_congr
  exact h.mem

/-- Complete source success and exact memory effect, including every loop
iteration and every masked logit/gather lane. No arithmetic law is assumed. -/
theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (hN : 0 < p.tokens) :
    ∃ final, FP.Structural.exec M (source p) s = some final ∧
      final.mem = (written p s (outputValues M p (finalAccumulator M p s))).mem := by
  obtain ⟨a,ha,har⟩ := prefix_run M p s
  obtain ⟨b,hb,hbr⟩ := loop_run M p s a 0 (sequenceLength p s) p.tokens _ har hN
  obtain ⟨final,hf,hm⟩ := suffix_run M p s b _ hbr
  refine ⟨final,?_,hm⟩
  change run M (source p).surfaceBody s = _
  rw [source_split]
  simp only [run_append,ha]
  simp only [Option.bind_some,run_append,run,step,evalOp_unfold,har.seqlen,Option.bind]
  simpa only [bind,pure,Option.bind,hb] using hf

def Post {α : Type} (p : Parameters) (s final : State α) (v : Index p → α) : Prop :=
  (∀ i, final.mem p.output (outputOffset p s i) = .mk .fp32 (v i)) ∧
  (∀ r o, (r ≠ p.output ∨ ∀ i, o ≠ outputOffset p s i) → final.mem r o = s.mem r o)

theorem correctness {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (hN : 0 < p.tokens) (hinj : Function.Injective (outputOffset p s)) :
    ∃ final, FP.Structural.exec M (source p) s = some final ∧
      Post p s final (outputValues M p (finalAccumulator M p s)) := by
  obtain ⟨final,he,hm⟩ := execution_memory M p s hN
  refine ⟨final,he,?_,?_⟩
  · intro i
    rw [hm]
    exact State.scatter_readback _ _ _ _ i hinj
  · intro r o h
    rw [hm]
    exact State.scatter_frame _ _ _ _ r o (h.imp_right fun hh i _ => hh i) s
#axiomsClean scatterMemory_congr
#axiomsClean suffix_run
#axiomsClean execution_memory
#axiomsClean correctness
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPValue
