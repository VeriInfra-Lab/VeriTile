import bench.tritonbench_g.matmul_dequantize_int4.Proofs.FPSourceBits
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.SourceBits
open VeriTile.Triton FP.Structural FPTyped
open Int4Matmul.FPSoftware (Word)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000

theorem correctness_memory {α : Type} [Inhabited α] (M : Algebra α) (D : FPTyped.Dot α)
    (p : Parameters) (initial : State α) :
    ∃ final, FP.Structural.exec (backend M D) (source p) initial = some final ∧
      final.mem = (written M D p initial).mem := by
  let start := preludeExecution M D p initial
  obtain ⟨mid, hm, hi⟩ := loopExecution M D p initial start.val 0 (by omega)
    (preludeInvariant M D p initial)
  let finish := epilogueExecution M D p initial mid hi
  refine ⟨finish.val, ?_, epilogueMemory M D p initial mid hi⟩
  change run (backend M D) (source p).surfaceBody initial = some finish.val
  rw [source_decomposition, run_append, run_append, start.property]
  have hsingle (s : State α) (hn : s.regs .nat [] "num_pid_k" = some (fun _ => steps p)) :
      run (backend M D) [.forRangeDyn "k" (.constNat 0) (.ref .nat [] "num_pid_k") (.constNat 1) (body p)] s =
        FP.Structural.range (backend M D) "k" 0 (steps p) 1 (body p) s := by
    simp [run, step, evalOp_unfold, hn]
  simp only [Option.bind_some]
  rw [hsingle start.val (preludeInvariant M D p initial).numk, hm]
  simp only [Option.bind_some]
  exact finish.property

theorem source_memory_correctness (rest : Algebra Word) (p : Parameters) (s : State Word) :
    ∃ final, FP.Structural.exec (FPTyped.backend (FPSoftware.scalar rest) FPSoftware.softwareDot)
      (FPTyped.source p) s = some final ∧ final.mem = (sourceReference p s).mem := by
  simpa only [FPTyped.written,FPSoftware.output_scalar,output_eq,sourceReference] using
    correctness_memory (FPSoftware.scalar rest) FPSoftware.softwareDot p s

theorem source_readback_correctness (rest : Algebra Word) (p : Parameters) (s : State Word)
    (hw : ∀ i j, active p s i → active p s j → outputOffset p s i=outputOffset p s j → i=j) :
    ∃ final, FP.Structural.exec (FPTyped.backend (FPSoftware.scalar rest) FPSoftware.softwareDot)
      (FPTyped.source p) s = some final ∧ FPTyped.Post p s final (sourceOutput p s) := by
  simpa only [output_eq] using FPSoftware.correctness rest FPSoftware.softwareDot p s hw

theorem source_rest_independence (a b : Algebra Word) (p : Parameters) (s : State Word) :
    ∃ t u,
      FP.Structural.exec (FPTyped.backend (FPSoftware.scalar a) FPSoftware.softwareDot) (FPTyped.source p) s=some t ∧
      FP.Structural.exec (FPTyped.backend (FPSoftware.scalar b) FPSoftware.softwareDot) (FPTyped.source p) s=some u ∧
      t.mem=u.mem := by
  obtain ⟨t,ht,hm⟩ := source_memory_correctness a p s
  obtain ⟨u,hu,hn⟩ := source_memory_correctness b p s
  exact ⟨t,u,ht,hu,hm.trans hn.symm⟩

#axiomsClean correctness_memory
#axiomsClean source_memory_correctness
#axiomsClean source_readback_correctness
#axiomsClean source_rest_independence
end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.SourceBits
