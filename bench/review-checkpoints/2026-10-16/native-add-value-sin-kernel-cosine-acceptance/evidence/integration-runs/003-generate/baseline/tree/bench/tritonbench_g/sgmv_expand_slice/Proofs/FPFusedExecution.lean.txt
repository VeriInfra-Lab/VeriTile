import bench.tritonbench_g.sgmv_expand_slice.Proofs.FPFusedSource
namespace VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false

theorem fused_body_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (F : DotAcc)
    (fmt : Storage) (out : Output) (p : Parameters) (s t : State Word) (cur : Nat) (a : Accumulator p)
    (h : Ready p s t cur a) :
    ∃ final, run (fusedBackend rest cfg F) (fusedBody fmt out p)
      (t.setReg "kk" .nat [] (fun _ => cur))=some final ∧
      Ready p s final (cur+p.blockInner) (fusedUpdate F fmt p s cur a) := by
  cases fmt <;>
    simp [fusedBody,fusedSource,fusedHalfTyped,fusedBfloatTyped,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
      integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,State.setReg,
      h.mem,h.pids,h.undef,h.pidm,h.pidn,h.batch,h.length,h.start,h.adapter,h.offm,h.offn,h.offk,
      h.ram,h.rbn,h.aptr,h.bptr,h.acc,fusedBackend,backend,installAcc,LoraExpandGemv.FPValue.backend,
      LoraExpandGemv.FPValue.formatOf,LoraExpandGemv.FPValue.addWord]
  all_goals constructor <;>
    simp [State.setReg,h.mem,h.pids,h.undef,h.pidm,h.pidn,h.batch,h.length,h.start,h.adapter,h.offm,h.offn,h.offk,
      h.ram,h.rbn,h.aptr,h.bptr,h.acc,fusedUpdate,aValues,bValues,readWord,storageType,aPointer,bPointer,
      aOffset,bOffset,bop,ofFloat,toFloat,Region.cast,Convert.cast,
      LoraExpandGemv.FPValue.castWord]
  all_goals first | rfl | (funext i; simp [aPointer,bPointer,aOffset,bOffset,bop,Nat.add_mul,Nat.add_assoc])

theorem fused_loop_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (F : DotAcc)
    (fmt : Storage) (out : Output) (p : Parameters) (s t : State Word) (cur : Nat) (a : Accumulator p)
    (h : Ready p s t cur a) :
    ∃ final, FP.Structural.range (fusedBackend rest cfg F) "kk" cur p.inner p.blockInner (fusedBody fmt out p) t=some final ∧
      ∃ last, Ready p s final last (foldRange (fusedUpdate F fmt p s) cur p.inner p.blockInner a) := by
  apply range_values (fusedBackend rest cfg F) "kk" (fusedBody fmt out p) (fusedUpdate F fmt p s)
    (fun n a u => Ready p s u n a) (fun a u => ∃ n, Ready p s u n a) cur p.inner p.blockInner a t
  · intro n a u hu _
    exact fused_body_run rest cfg F fmt out p s u n a hu
  · intro n a u hu
    exact ⟨n,hu⟩
  · exact h
theorem fused_execution_memory [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (F : DotAcc)
    (fmt : Storage) (out : Output) (p : Parameters) (s : State Word) :
    ∃ final, FP.Structural.exec (fusedBackend rest cfg F) (fusedSource fmt out p) s=some final ∧
      final.mem=(written out p s (outputValues cfg fmt out p (fusedAccumulated F fmt p s))).mem := by
  obtain ⟨t,ht,htr⟩ := prelude_run (installAcc rest F) cfg unusedDot fmt out p s
  obtain ⟨u,hu,last,hur⟩ := fused_loop_run rest cfg F fmt out p s t 0 (initial p) htr
  obtain ⟨final,hf,hm⟩ := suffix_run (installAcc rest F) cfg unusedDot fmt out p s u last _ hur
  refine ⟨final,?_,hm⟩
  change run (fusedBackend rest cfg F) (fusedSource fmt out p).surfaceBody s=_
  rw [fused_split]
  change run (backend (installAcc rest F) cfg unusedDot)
    (prelude fmt out p ++ [.forRange "kk" 0 p.inner p.blockInner (fusedBody fmt out p)] ++ suffix fmt out p) s=_
  simp only [run_append,ht,Option.bind_some,run,step]
  change FP.Structural.range (backend (installAcc rest F) cfg unusedDot) "kk" 0 p.inner p.blockInner (fusedBody fmt out p) t=some u at hu
  rw [hu]
  exact hf

theorem fused_correctness [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (F : DotAcc)
    (fmt : Storage) (out : Output) (p : Parameters) (s : State Word)
    (hinj : Function.Injective (outputOffset p s)) :
    ∃ final, FP.Structural.exec (fusedBackend rest cfg F) (fusedSource fmt out p) s=some final ∧
      Post out p s final (outputValues cfg fmt out p (fusedAccumulated F fmt p s)) := by
  obtain ⟨final,he,hm⟩ := fused_execution_memory rest cfg F fmt out p s
  refine ⟨final,he,?_,?_⟩
  · intro i hi
    rw [hm]
    exact State.masked_scatter_readback _ _ _ (fun i => active p s i=Bool.true) s i hi hinj
  · intro r o h
    rw [hm]
    exact State.masked_scatter_frame _ _ _ (fun i => active p s i=Bool.true) _ r o
      (h.imp_right fun hh i _ => hh i) s
#axiomsClean fused_body_run
#axiomsClean fused_loop_run
#axiomsClean fused_execution_memory
#axiomsClean fused_correctness
end
end VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
