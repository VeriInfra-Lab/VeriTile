import bench.tritonbench_g.sgmv_expand_slice.Proofs.FPLoop
namespace VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false

theorem fold_memory_congr {ι : Type} (l : List ι) (output : RegionName) (offset : ι → Nat)
    (active : ι → Prop) [DecidablePred active] (v : ι → Cell Word) (s t : State Word) (h : s.mem=t.mem) :
    (l.foldl (fun u i => if active i then u.write output (offset i) (v i) else u) s).mem =
    (l.foldl (fun u i => if active i then u.write output (offset i) (v i) else u) t).mem := by
  induction l generalizing s t with
  | nil => exact h
  | cons i tail ih =>
    simp only [List.foldl_cons]
    split <;> apply ih
    · simp only [State.write,h]
    · exact h

theorem suffix_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot)
    (fmt : Storage) (out : Output) (p : Parameters) (s t : State Word) (cur : Nat) (a : Accumulator p)
    (h : Ready p s t cur a) :
    ∃ final, run (backend rest cfg D) (suffix fmt out p) t=some final ∧
      final.mem=(written out p s (outputValues cfg fmt out p a)).mem := by
  cases fmt <;> cases out <;>
    simp [suffix,source,halfTyped,bfloatTyped,outputType,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
      Region.cast,ofFloat,toFloat,State.setReg,store,
      h.mem,h.pids,h.undef,h.pidm,h.pidn,h.batch,h.length,h.start,h.adapter,h.offm,h.offn,h.offk,
      h.ram,h.rbn,h.aptr,h.bptr,h.acc,backend,LoraExpandGemv.FPValue.backend,
      LoraExpandGemv.FPValue.formatOf]
  all_goals simp only [written,active,outputOffset,outputCell,outputValues,storageFormat,outputFormat,
    Convert.cast,decide_eq_true_eq,Bool.and_eq_true]
  all_goals apply fold_memory_congr
  all_goals rfl

theorem execution_memory [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot)
    (fmt : Storage) (out : Output) (p : Parameters) (s : State Word) :
    ∃ final, FP.Structural.exec (backend rest cfg D) (source fmt out p) s=some final ∧
      final.mem=(written out p s (outputValues cfg fmt out p (accumulated cfg D fmt p s))).mem := by
  obtain ⟨t,ht,htr⟩ := prelude_run rest cfg D fmt out p s
  obtain ⟨u,hu,last,hur⟩ := loop_run rest cfg D fmt out p s t 0 (initial p) htr
  obtain ⟨final,hf,hm⟩ := suffix_run rest cfg D fmt out p s u last _ hur
  refine ⟨final,?_,hm⟩
  change run (backend rest cfg D) (source fmt out p).surfaceBody s=_
  rw [source_split]
  simp only [run_append,ht,Option.bind_some,run,step,hu]
  exact hf

theorem correctness [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot)
    (fmt : Storage) (out : Output) (p : Parameters) (s : State Word)
    (hinj : Function.Injective (outputOffset p s)) :
    ∃ final, FP.Structural.exec (backend rest cfg D) (source fmt out p) s=some final ∧
      Post out p s final (outputValues cfg fmt out p (accumulated cfg D fmt p s)) := by
  obtain ⟨final,he,hm⟩ := execution_memory rest cfg D fmt out p s
  refine ⟨final,he,?_,?_⟩
  · intro i hi
    rw [hm]
    exact State.masked_scatter_readback _ _ _ (fun i => active p s i=Bool.true) s i hi hinj
  · intro r o h
    rw [hm]
    exact State.masked_scatter_frame _ _ _ (fun i => active p s i=Bool.true) _ r o
      (h.imp_right fun hh i _ => hh i) s
#axiomsClean suffix_run
#axiomsClean execution_memory
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
